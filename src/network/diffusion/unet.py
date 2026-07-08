from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.network.diffusion.timestep import TimestepEmbedding
from src.network.diffusion.blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    normalization,
)


class LatentDiffusionUNet(nn.Module):
    """
    Lightweight text-conditioned U-Net for latent diffusion.

    Input:
        z_t:
            noisy latent [B, in_channels, H, W]

        timesteps:
            diffusion timestep [B]

        context:
            CLIP token embeddings [B, seq_len, context_dim]

    Output:
        prediction [B, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        latent_size: int = 32,
        base_channels: int = 128,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 3),
        num_res_blocks: int = 2,
        attention_resolutions: list[int] | tuple[int, ...] = (16, 8),
        context_dim: int = 768,
        num_heads: int = 4,
        head_dim: int = 32,
        transformer_depth: int = 1,
        dropout: float = 0.0,
        time_embedding_dim: int | None = None,
        use_middle_attention: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.base_channels = base_channels
        self.channel_multipliers = list(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.transformer_depth = transformer_depth
        self.dropout = dropout

        if time_embedding_dim is None:
            time_embedding_dim = base_channels * 4

        self.time_embedding_dim = time_embedding_dim

        self.time_embed = TimestepEmbedding(
            embedding_dim=base_channels,
            time_embed_dim=time_embedding_dim,
        )

        self.conv_in = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        channels_per_level = [
            base_channels * multiplier
            for multiplier in self.channel_multipliers
        ]

        # -------------------------
        # Down path
        # -------------------------
        self.down_blocks = nn.ModuleList()

        current_channels = base_channels
        current_resolution = latent_size

        self.down_block_channels: list[int] = []
        self.down_block_resolutions: list[int] = []

        for level, out_channels_level in enumerate(channels_per_level):
            is_last = level == len(channels_per_level) - 1

            use_attention = current_resolution in self.attention_resolutions

            block = DownBlock(
                in_channels=current_channels,
                out_channels=out_channels_level,
                time_embed_dim=time_embedding_dim,
                num_res_blocks=num_res_blocks,
                context_dim=context_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                transformer_depth=transformer_depth,
                dropout=dropout,
                use_attention=use_attention,
                add_downsample=not is_last,
            )

            self.down_blocks.append(block)

            self.down_block_channels.append(out_channels_level)
            self.down_block_resolutions.append(current_resolution)

            current_channels = out_channels_level

            if not is_last:
                current_resolution //= 2

        # -------------------------
        # Middle
        # -------------------------
        self.middle = MiddleBlock(
            channels=current_channels,
            time_embed_dim=time_embedding_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            transformer_depth=transformer_depth,
            dropout=dropout,
            use_attention=use_middle_attention,
        )

        # -------------------------
        # Up path
        # -------------------------
        self.up_blocks = nn.ModuleList()

        reversed_channels = list(reversed(self.down_block_channels))
        reversed_resolutions = list(reversed(self.down_block_resolutions))

        for level, (skip_channels, resolution) in enumerate(
            zip(reversed_channels, reversed_resolutions)
        ):
            is_last = level == len(reversed_channels) - 1

            out_channels_level = skip_channels
            use_attention = resolution in self.attention_resolutions

            block = UpBlock(
                in_channels=current_channels,
                skip_channels=skip_channels,
                out_channels=out_channels_level,
                time_embed_dim=time_embedding_dim,
                num_res_blocks=num_res_blocks,
                context_dim=context_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                transformer_depth=transformer_depth,
                dropout=dropout,
                use_attention=use_attention,
                add_upsample=not is_last,
            )

            self.up_blocks.append(block)

            current_channels = out_channels_level

        self.norm_out = normalization(current_channels)

        self.conv_out = nn.Conv2d(
            current_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Stable start: initially predicts near zero.
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z_t:
                Noisy latent [B, C, H, W]

            timesteps:
                Diffusion timesteps [B]

            context:
                CLIP token embeddings [B, seq_len, context_dim]

            attention_mask:
                CLIP token mask [B, seq_len], optional

        Returns:
            model_output:
                [B, out_channels, H, W]
        """
        if z_t.ndim != 4:
            raise ValueError(f"z_t must be [B, C, H, W], got {z_t.shape}")

        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must be [B], got {timesteps.shape}")

        time_emb = self.time_embed(timesteps)

        x = self.conv_in(z_t)

        skips: list[torch.Tensor] = []

        for block in self.down_blocks:
            x, block_skips = block(
                x=x,
                time_emb=time_emb,
                context=context,
                attention_mask=attention_mask,
            )

            skips.extend(block_skips)

        x = self.middle(
            x=x,
            time_emb=time_emb,
            context=context,
            attention_mask=attention_mask,
        )

        for block in self.up_blocks:
            x = block(
                x=x,
                skips=skips,
                time_emb=time_emb,
                context=context,
                attention_mask=attention_mask,
            )

        if len(skips) != 0:
            raise RuntimeError(
                f"Unused skip connections remain: {len(skips)}. "
                "Check U-Net block construction."
            )

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


def count_parameters(
    model: nn.Module,
    trainable_only: bool = True,
) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def build_latent_diffusion_unet_from_config(cfg: dict) -> LatentDiffusionUNet:
    """
    Build U-Net from config dictionary.

    Expects:

        cfg["model"]
    """
    m = cfg["model"]

    return LatentDiffusionUNet(
        in_channels=int(m["in_channels"]),
        out_channels=int(m["out_channels"]),
        latent_size=int(m.get("latent_size", 32)),
        base_channels=int(m["base_channels"]),
        channel_multipliers=tuple(m["channel_multipliers"]),
        num_res_blocks=int(m["num_res_blocks"]),
        attention_resolutions=tuple(m.get("attention_resolutions", [16, 8])),
        context_dim=int(m["context_dim"]),
        num_heads=int(m["num_heads"]),
        head_dim=int(m["head_dim"]),
        transformer_depth=int(m.get("transformer_depth", 1)),
        dropout=float(m.get("dropout", 0.0)),
        time_embedding_dim=m.get("time_embedding_dim", None),
        use_middle_attention=bool(m.get("use_middle_attention", True)),
    )