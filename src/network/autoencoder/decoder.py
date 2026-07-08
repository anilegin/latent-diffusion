from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.network.autoencoder.blocks import (
    ResBlock,
    Upsample,
    MidBlock,
    normalization,
)


class Decoder(nn.Module):
    """
    VAE decoder.

    Converts latent tensor back into image space.
    Shape path:

        [B, 4,   32,  32]
            -> conv_in
        [B, 512, 32,  32]
            -> mid block
        [B, 512, 32,  32]
            -> upsample
        [B, 512, 64,  64]
            -> upsample
        [B, 256, 128, 128]
            -> upsample
        [B, 128, 256, 256]
            -> conv_out
        [B, 3,   256, 256]

    Output is in [-1, 1] because training images are normalized to [-1, 1].
    """

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_heads: int = 1
    ):
        super().__init__()

        if len(channel_multipliers) < 2:
            raise ValueError("channel_multipliers must contain at least 2 levels.")

        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.channel_multipliers = list(channel_multipliers)
        self.num_res_blocks = num_res_blocks

        # Number of spatial upsampling operations
        # Example:
        #   [1, 2, 4, 4] has 4 levels, so decoder upsamples 3 times:
        #   32 -> 64 -> 128 -> 256
        self.num_upsamples = len(self.channel_multipliers) - 1

        # Start from the deepest encoder channel count
        current_channels = base_channels * self.channel_multipliers[-1]

        self.conv_in = nn.Conv2d(
            latent_channels,
            current_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Bottleneck block at the lowest spatial resolution.
        self.mid = MidBlock(
            channels=current_channels,
            dropout=dropout,
            use_attention=use_attention,
            num_heads=attention_heads,
        )

        self.up_blocks = nn.ModuleList()

        reversed_multipliers = list(reversed(self.channel_multipliers))

        for level, multiplier in enumerate(reversed_multipliers):
            out_stage_channels = base_channels * multiplier

            resblocks = nn.ModuleList()

            # one extra ResBlock per level
            for _ in range(num_res_blocks + 1):
                resblocks.append(
                    ResBlock(
                        in_channels=current_channels,
                        out_channels=out_stage_channels,
                        dropout=dropout,
                    )
                )
                current_channels = out_stage_channels

            # Upsample after every stage except the full-resolution
            if level < len(reversed_multipliers) - 1:
                upsample = Upsample(
                    channels=current_channels
                )
            else:
                upsample = nn.Identity()

            self.up_blocks.append(
                nn.ModuleDict(
                    {
                        "resblocks": resblocks,
                        "upsample": upsample,
                    }
                )
            )

        self.norm_out = normalization(current_channels)

        self.conv_out = nn.Conv2d(
            current_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:
                Latent tensor with shape [B, latent_channels, H/8, W/8].
                For 256x256 images and downsample factor 8:
                    [B, latent_channels, 32, 32]

        Returns:
            x_recon:
                Reconstructed image tensor with shape [B, 3, H, W].
                Values are in [-1, 1].
        """
        h = self.conv_in(z)

        h = self.mid(h)

        for stage in self.up_blocks:
            for block in stage["resblocks"]:
                h = block(h)

            h = stage["upsample"](h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        x_recon = torch.tanh(h)

        return x_recon