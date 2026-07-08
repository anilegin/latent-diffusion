from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class GEGLU(nn.Module):
    """
    Gated GELU feed-forward projection.

    Used in transformer-style attention blocks.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
    ):
        super().__init__()

        self.proj = nn.Linear(
            dim_in,
            dim_out * 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """
    Transformer feed-forward block
    """

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        inner_dim = dim * mult

        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """
    Multi-head attention.

    If context is None:
        self-attention

    If context is provided:
        cross-attention from x to context.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        inner_dim = num_heads * head_dim
        context_dim = query_dim if context_dim is None else context_dim

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = inner_dim

        self.to_q = nn.Linear(
            query_dim,
            inner_dim,
            bias=False,
        )

        self.to_k = nn.Linear(
            context_dim,
            inner_dim,
            bias=False,
        )

        self.to_v = nn.Linear(
            context_dim,
            inner_dim,
            bias=False,
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape

        if context is None:
            context = x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # q: [B, heads, N, head_dim]
        # k: [B, heads, M, head_dim]
        # v: [B, heads, M, head_dim]

        attn_mask = None

        if attention_mask is not None:
            # attention_mask: [B, M], 1 for valid tokens, 0 for padding.
            # scaled_dot_product_attention expects True where attention is allowed
            # or additive mask depending on dtype. Bool mask is okay.
            attn_mask = attention_mask.bool()
            attn_mask = attn_mask[:, None, None, :]  # [B, 1, 1, M]

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
        )

        out = out.transpose(1, 2).contiguous()
        out = out.view(b, n, self.inner_dim)

        out = self.to_out(out)

        return out


class BasicTransformerBlock(nn.Module):
    """
    Transformer block used inside spatial U-Net feature maps

    it has:

        self-attention
        cross-attention
        feed-forward
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(
            query_dim=dim,
            context_dim=None,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(
            dim=dim,
            mult=4,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.norm1(x),
            context=None,
        )

        x = x + self.cross_attn(
            self.norm2(x),
            context=context,
            attention_mask=attention_mask,
        )

        x = x + self.ff(
            self.norm3(x),
        )

        return x


class SpatialTransformer(nn.Module):
    """
    Applies transformer attention on 2D feature maps.
    This is where text conditioning enters the U-Net.
    """

    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        depth: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.depth = depth

        self.norm = nn.GroupNorm(
            num_groups=32,
            num_channels=channels,
            eps=1e-6,
            affine=True,
        )

        inner_dim = num_heads * head_dim

        self.proj_in = nn.Conv2d(
            channels,
            inner_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=inner_dim,
                    context_dim=context_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.proj_out = nn.Conv2d(
            inner_dim,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Stable residual start.
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape

        residual = x

        x = self.norm(x)
        x = self.proj_in(x)

        inner_dim = x.shape[1]

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(b, h * w, inner_dim)

        for block in self.transformer_blocks:
            x = block(
                x,
                context=context,
                attention_mask=attention_mask,
            )

        x = x.view(b, h, w, inner_dim)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.proj_out(x)

        return x + residual