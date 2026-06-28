from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.models.diffusion.attention import SpatialTransformer


def normalization(
    channels: int,
    num_groups: int = 32,
) -> nn.GroupNorm:
    """
    GroupNorm 
    """
    num_groups = min(num_groups, channels)

    while channels % num_groups != 0:
        num_groups -= 1

    return nn.GroupNorm(
        num_groups=num_groups,
        num_channels=channels,
        eps=1e-6,
        affine=True,
    )


class TimeResBlock(nn.Module):
    """
    Residual block conditioned on timestep embedding.
    Time embedding is projected and added after the first conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed_dim = time_embed_dim

        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.time_proj = nn.Linear(
            time_embed_dim,
            out_channels,
        )

        self.norm2 = normalization(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.shortcut(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_out = self.time_proj(
            F.silu(time_emb),
        )

        h = h + time_out[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class Downsample(nn.Module):
    """
    Downsample by factor 2 using strided convolution
    """

    def __init__(
        self,
        channels: int,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsample by factor 2 using nearest-neighbor + conv
    """

    def __init__(
        self,
        channels: int,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2.0,
            mode="nearest",
        )

        x = self.conv(x)

        return x


class AttentionBlock(nn.Module):
    """
    Optional text-conditioned attention block.

    If use_attention=True:
        applies SpatialTransformer.

    If use_attention=False:
        identity.
    """

    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        transformer_depth: int = 1,
        dropout: float = 0.0,
        use_attention: bool = True,
    ):
        super().__init__()

        if use_attention:
            self.block = SpatialTransformer(
                channels=channels,
                context_dim=context_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                depth=transformer_depth,
                dropout=dropout,
            )
        else:
            self.block = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(self.block, nn.Identity):
            return x

        if context is None:
            raise ValueError("AttentionBlock requires context when use_attention=True.")

        return self.block(
            x,
            context=context,
            attention_mask=attention_mask,
        )


class DownBlock(nn.Module):
    """
    U-Net down block.

    Contains:
        ResBlock(s)
        optional SpatialTransformer(s)
        optional downsample

    Returns:
        x
        skip features
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_res_blocks: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        transformer_depth: int = 1,
        dropout: float = 0.0,
        use_attention: bool = False,
        add_downsample: bool = True,
    ):
        super().__init__()

        self.resblocks = nn.ModuleList()
        self.attentions = nn.ModuleList()

        current_channels = in_channels

        for _ in range(num_res_blocks):
            self.resblocks.append(
                TimeResBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                )
            )

            self.attentions.append(
                AttentionBlock(
                    channels=out_channels,
                    context_dim=context_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    transformer_depth=transformer_depth,
                    dropout=dropout,
                    use_attention=use_attention,
                )
            )

            current_channels = out_channels

        if add_downsample:
            self.downsample = Downsample(out_channels)
        else:
            self.downsample = nn.Identity()

        self.out_channels = out_channels
        self.add_downsample = add_downsample

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []

        for resblock, attention in zip(self.resblocks, self.attentions):
            x = resblock(
                x,
                time_emb,
            )

            x = attention(
                x,
                context=context,
                attention_mask=attention_mask,
            )

            skips.append(x)

        x = self.downsample(x)

        return x, skips


class UpBlock(nn.Module):
    """
    U-Net up block.

    Takes skip features from encoder path.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_res_blocks: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        transformer_depth: int = 1,
        dropout: float = 0.0,
        use_attention: bool = False,
        add_upsample: bool = True,
    ):
        super().__init__()

        self.resblocks = nn.ModuleList()
        self.attentions = nn.ModuleList()

        current_channels = in_channels

        for _ in range(num_res_blocks):
            self.resblocks.append(
                TimeResBlock(
                    in_channels=current_channels + skip_channels,
                    out_channels=out_channels,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                )
            )

            self.attentions.append(
                AttentionBlock(
                    channels=out_channels,
                    context_dim=context_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    transformer_depth=transformer_depth,
                    dropout=dropout,
                    use_attention=use_attention,
                )
            )

            current_channels = out_channels

        if add_upsample:
            self.upsample = Upsample(out_channels)
        else:
            self.upsample = nn.Identity()

        self.out_channels = out_channels
        self.add_upsample = add_upsample

    def forward(
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor],
        time_emb: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for resblock, attention in zip(self.resblocks, self.attentions):
            if len(skips) == 0:
                raise RuntimeError("Not enough skip connections for UpBlock.")

            skip = skips.pop()

            x = torch.cat(
                [x, skip],
                dim=1,
            )

            x = resblock(
                x,
                time_emb,
            )

            x = attention(
                x,
                context=context,
                attention_mask=attention_mask,
            )

        x = self.upsample(x)

        return x


class MiddleBlock(nn.Module):
    """
    U-Net bottleneck block
    """

    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        transformer_depth: int = 1,
        dropout: float = 0.0,
        use_attention: bool = True,
    ):
        super().__init__()

        self.res1 = TimeResBlock(
            in_channels=channels,
            out_channels=channels,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
        )

        self.attn = AttentionBlock(
            channels=channels,
            context_dim=context_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            transformer_depth=transformer_depth,
            dropout=dropout,
            use_attention=use_attention,
        )

        self.res2 = TimeResBlock(
            in_channels=channels,
            out_channels=channels,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.res1(
            x,
            time_emb,
        )

        x = self.attn(
            x,
            context=context,
            attention_mask=attention_mask,
        )

        x = self.res2(
            x,
            time_emb,
        )

        return x