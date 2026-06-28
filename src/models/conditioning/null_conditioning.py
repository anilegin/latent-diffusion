from __future__ import annotations

import random

import torch
from torch import nn

from src.models.conditioning.clip_text import FrozenCLIPTextEncoder


class ClassifierFreeGuidanceConditioner(nn.Module):
    """
    During training, with probability `cond_drop_prob`, captions are replaced
    with the empty string:

        "a dog on grass" -> ""

    Later at sampling time, CFG uses:

        pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    """

    def __init__(
        self,
        text_encoder: FrozenCLIPTextEncoder,
        cond_drop_prob: float = 0.1,
        empty_text: str = "",
    ):
        super().__init__()

        if cond_drop_prob < 0.0 or cond_drop_prob > 1.0:
            raise ValueError("cond_drop_prob must be between 0 and 1.")

        self.text_encoder = text_encoder
        self.cond_drop_prob = cond_drop_prob
        self.empty_text = empty_text

    @property
    def context_dim(self) -> int:
        return self.text_encoder.context_dim

    @property
    def max_length(self) -> int:
        return self.text_encoder.max_length

    def apply_conditioning_dropout(
        self,
        captions: list[str] | tuple[str, ...],
        force_drop_ids: torch.Tensor | None = None,
    ) -> tuple[list[str], torch.Tensor]:
        """
        Replace some captions with empty text
        """
        captions = list(captions)
        batch_size = len(captions)

        if force_drop_ids is not None:
            drop_mask = force_drop_ids.bool().cpu()
        else:
            drop_mask = torch.zeros(batch_size, dtype=torch.bool)

            for i in range(batch_size):
                if random.random() < self.cond_drop_prob:
                    drop_mask[i] = True

        dropped_captions = []

        for caption, drop in zip(captions, drop_mask):
            if bool(drop):
                dropped_captions.append(self.empty_text)
            else:
                dropped_captions.append(caption)

        return dropped_captions, drop_mask

    def forward(
        self,
        captions: list[str] | tuple[str, ...],
        device: torch.device | str | None = None,
        apply_dropout: bool = True,
        force_drop_ids: torch.Tensor | None = None,
    ):
        """
        Encode captions with optional CFG dropout
        """
        if apply_dropout:
            captions, drop_mask = self.apply_conditioning_dropout(
                captions=captions,
                force_drop_ids=force_drop_ids,
            )
        else:
            captions = list(captions)
            drop_mask = torch.zeros(
                len(captions),
                dtype=torch.bool,
            )

        output = self.text_encoder(
            captions=captions,
            device=device,
        )

        if device is not None:
            drop_mask = drop_mask.to(device)

        return {
            "context": output.hidden_states,
            "attention_mask": output.attention_mask,
            "pooled": output.pooled,
            "drop_mask": drop_mask,
            "captions": captions,
        }

    @torch.no_grad()
    def encode_cond_uncond(
        self,
        captions: list[str] | tuple[str, ...],
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Encode both conditional and unconditional text.

        Used during CFG sampling
        """
        captions = list(captions)
        batch_size = len(captions)

        cond_output = self.text_encoder(
            captions=captions,
            device=device,
        )

        uncond_output = self.text_encoder(
            captions=[self.empty_text] * batch_size,
            device=device,
        )

        return {
            "cond_context": cond_output.hidden_states,
            "cond_attention_mask": cond_output.attention_mask,
            "uncond_context": uncond_output.hidden_states,
            "uncond_attention_mask": uncond_output.attention_mask,
        }