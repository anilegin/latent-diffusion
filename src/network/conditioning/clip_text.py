from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class TextConditioningOutput:
    """
    Output of the CLIP text encoder.

    hidden_states:
        Token-level CLIP embeddings.
        Shape: [B, seq_len, hidden_dim]

    attention_mask:
        Token attention mask.
        Shape: [B, seq_len]

    pooled:
        Optional pooled text embedding.
        Shape: [B, hidden_dim]
    """

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    pooled: torch.Tensor | None = None


class FrozenCLIPTextEncoder(nn.Module):
    """
    Frozen CLIP text encoder for latent diffusion conditioning
    model:
        openai/clip-vit-large-patch14

    This gives:
        context_dim = 768
        max_length = 77

    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        max_length: int = 77,
        freeze: bool = True,
        use_last_hidden_state: bool = True,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.freeze = freeze
        self.use_last_hidden_state = use_last_hidden_state
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        self.text_model = CLIPTextModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        if self.freeze:
            self.text_model.eval()
            for p in self.text_model.parameters():
                p.requires_grad = False

    @property
    def context_dim(self) -> int:
        return int(self.text_model.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    def train(self, mode: bool = True):
        """
        Keep CLIP frozen/eval even if parent model calls .train().
        """
        super().train(mode)

        if self.freeze:
            self.text_model.eval()

        return self

    def tokenize(
        self,
        captions: list[str] | tuple[str, ...],
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize captions into CLIP input tensors.
        """
        tokens = self.tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if device is not None:
            tokens = {
                key: value.to(device)
                for key, value in tokens.items()
            }

        return tokens

    def forward(
        self,
        captions: list[str] | tuple[str, ...],
        device: torch.device | str | None = None,
    ) -> TextConditioningOutput:
        """
        Produces CLIP textual embeddings as diffusion condition.
        """
        if device is None:
            device = next(self.text_model.parameters()).device

        tokens = self.tokenize(
            captions=captions,
            device=device,
        )

        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=not self.use_last_hidden_state,
                return_dict=True,
            )

        if self.use_last_hidden_state:
            hidden_states = outputs.last_hidden_state
        else:
            # Penultimate layer is sometimes used in diffusion models.
            hidden_states = outputs.hidden_states[-2]

        pooled = outputs.pooler_output

        return TextConditioningOutput(
            hidden_states=hidden_states,
            attention_mask=tokens["attention_mask"],
            pooled=pooled,
        )

    @torch.no_grad()
    def encode(
        self,
        captions: list[str] | tuple[str, ...],
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Convenience function.

        Returns only token-level context:

            [B, seq_len, context_dim]
        """
        return self.forward(
            captions=captions,
            device=device,
        ).hidden_states