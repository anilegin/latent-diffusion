from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.models.autoencoder.blocks import (
    ResBlock,
    Downsample,
    MidBlock,
    normalization,
    SelfAttentionBlock,
)

class Encoder(nn.Module):
    """
    VAE encoder.
    channel_multipliers=[1, 2, 4]: this controls the multiplier of number of feature maps

        [B, 3, 256, 256]
            -> [B, 128, 256, 256]
            -> [B, 128, 128, 128]
            -> [B, 256, 64, 64]
            -> [B, 512, 32, 32]
            -> [B, 2 * latent_channels, 32, 32]

    Output channels are 2 * latent_channels because we predict:
        mu
        logvar
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 8,
        base_channels: int = 128,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 3,
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_heads: int = 4,
        attention_resolutions: tuple[int, ...] = (32,),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.channel_multipliers = list(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)

        # Initial projection
        self.conv_in = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Downsampling
        self.down_blocks = nn.ModuleList()

        current_channels = base_channels
        current_resolution = 256

        for level, multiplier in enumerate(self.channel_multipliers):
            out_channels = base_channels * multiplier

            stage = nn.ModuleDict()
            stage["resblocks"] = nn.ModuleList()

            for _ in range(num_res_blocks):
                stage["resblocks"].append(
                    ResBlock(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        dropout=dropout,
                    )
                )
                current_channels = out_channels


            # This part also adds attention to 64x64 resolution along with bottleneck.
            if use_attention and current_resolution in self.attention_resolutions:
                stage["attention"] = SelfAttentionBlock(
                    channels=current_channels,
                    num_heads=attention_heads,
                )
            else:
                stage["attention"] = nn.Identity()

            # Downsample after each stage except the final one
            if level != len(self.channel_multipliers) - 1:
                stage["downsample"] = Downsample(current_channels)
                next_resolution = current_resolution // 2
            else:
                stage["downsample"] = nn.Identity()
                next_resolution = current_resolution

            self.down_blocks.append(stage)
            current_resolution = next_resolution

        # Bottleneck
        self.mid = MidBlock(
            channels=current_channels,
            dropout=dropout,
            use_attention=use_attention,
            num_heads=attention_heads,
        )

        # Output projection to posterior parameters
        self.norm_out = normalization(current_channels)
        self.conv_out = nn.Conv2d(
            current_channels,
            2 * latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Image tensor with shape [B, 3, H, W]

        Returns:
            moments:
                Tensor with shape [B, 2 * latent_channels, H/8, W/8]
                The first half is mu.
                The second half is logvar.
        """
        h = self.conv_in(x)

        for stage in self.down_blocks:
            for block in stage["resblocks"]:
                h = block(h)

            h = stage["attention"](h)
            h = stage["downsample"](h)

        h = self.mid(h)

        h = self.norm_out(h)
        h = F.silu(h)
        moments = self.conv_out(h)

        return moments