from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def init_conv_kaiming(module: nn.Module) -> None:
    """
    Kaiming initialization for convolutional layers used with SiLU activations.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_linear_xavier(module: nn.Module) -> None:
    """
    Xavier initialization for attention-style projection layers.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)


def normalization(num_channels: int, num_groups: int = 32):
    """
    GroupNorm used in VAE blocks
    """
    num_groups = min(num_groups, num_channels)

    while num_channels % num_groups != 0:
        num_groups -= 1

    return nn.GroupNorm(
        num_groups=num_groups,
        num_channels=num_channels,
        eps=1e-6,
        affine=True,
    )


class ResBlock(nn.Module):
    """
    Simple residual block:

        x -> GroupNorm -> SiLU -> Conv
          -> GroupNorm -> SiLU -> Conv
          + shortcut

    Used both in encoder and decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
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

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_conv_kaiming(self.conv1)
        init_conv_kaiming(self.conv2)

        if isinstance(self.shortcut, nn.Conv2d):
            init_conv_kaiming(self.shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class Downsample(nn.Module):
    """
    Downsample by factor 2 using strided convolution.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_conv_kaiming(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsample by factor 2 using nearest-neighbor interpolation + convolution instead of ConvTranspose2d.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_conv_kaiming(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class SelfAttentionBlock(nn.Module):
    """
    Spatial self-attention block for feature maps.

    Input:
        x: [B, C, H, W]

    then get:
        [B, C, H, W] -> [B, H*W, C]
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}"
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = normalization(channels)

        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.proj_out = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_linear_xavier(self.qkv)
        init_linear_xavier(self.proj_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        # [B, C, H, W] -> [B, num_heads, H*W, head_dim]
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        # Output: [B, num_heads, H*W, head_dim]
        out = F.scaled_dot_product_attention(q, k, v)

        # [B, num_heads, H*W, head_dim] -> [B, C, H, W]
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(b, c, h, w)

        out = self.proj_out(out)

        return residual + out


class AttnResBlock(nn.Module):
    """
    Optional attention block:

        ResBlock -> SelfAttentionBlock
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        use_attention: bool = False,
        num_heads: int = 1,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.resblock = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

        if use_attention:
            self.attn = SelfAttentionBlock(
                channels=out_channels,
                num_heads=num_heads,
            )
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resblock(x)
        x = self.attn(x)
        return x


class MidBlock(nn.Module):
    """
    Bottleneck block:

        ResBlock -> SelfAttentionBlock -> ResBlock
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        use_attention: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()

        self.block1 = ResBlock(
            in_channels=channels,
            out_channels=channels,
            dropout=dropout,
        )

        if use_attention:
            self.attn = SelfAttentionBlock(
                channels=channels,
                num_heads=num_heads,
            )
        else:
            self.attn = nn.Identity()

        self.block2 = ResBlock(
            in_channels=channels,
            out_channels=channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.attn(x)
        x = self.block2(x)
        return x


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero-initialize a module.
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module