from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings
    """
    half_dim = dim // 2

    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(
            start=0,
            end=half_dim,
            dtype=torch.float32,
            device=timesteps.device,
        )
        / half_dim
    )

    args = timesteps.float()[:, None] * frequencies[None]

    embedding = torch.cat(
        [
            torch.cos(args),
            torch.sin(args),
        ],
        dim=-1,
    )

    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))

    return embedding


class TimestepEmbedding(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        time_embed_dim: int,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.time_embed_dim = time_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_timestep_embedding(
            timesteps=timesteps,
            dim=self.embedding_dim,
        )

        emb = self.mlp(emb)

        return emb