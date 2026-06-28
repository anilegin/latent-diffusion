from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class NoiseSchedule:
    """
    Precomputed DDPM noise schedule.

    Main variables:

        beta_t:
            amount of noise added at timestep t

        alpha_t:
            1 - beta_t

        alpha_bar_t:
            cumulative product of alphas up to t

        q(z_t | z_0):
            z_t = sqrt(alpha_bar_t) * z_0
                + sqrt(1 - alpha_bar_t) * eps
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor

    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor

    log_one_minus_alphas_cumprod: torch.Tensor

    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor

    posterior_variance: torch.Tensor
    posterior_log_variance_clipped: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor

    num_timesteps: int
    schedule_type: str

    def to(self, device: torch.device | str) -> "NoiseSchedule":
        device = torch.device(device)

        return NoiseSchedule(
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alphas_cumprod=self.alphas_cumprod.to(device),
            alphas_cumprod_prev=self.alphas_cumprod_prev.to(device),
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod.to(device),
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod.to(device),
            log_one_minus_alphas_cumprod=self.log_one_minus_alphas_cumprod.to(device),
            sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod.to(device),
            sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod.to(device),
            posterior_variance=self.posterior_variance.to(device),
            posterior_log_variance_clipped=self.posterior_log_variance_clipped.to(device),
            posterior_mean_coef1=self.posterior_mean_coef1.to(device),
            posterior_mean_coef2=self.posterior_mean_coef2.to(device),
            num_timesteps=self.num_timesteps,
            schedule_type=self.schedule_type,
        )


def make_beta_schedule(
    schedule_type: str = "cosine",
    num_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    cosine_s: float = 0.008,
    max_beta: float = 0.999,
) -> torch.Tensor:
    """
    Create beta schedule.

    Supported:
        linear:
            Standard DDPM linear beta schedule.

        cosine:
            Improved DDPM cosine schedule.
            Usually better behaved and good default for v-prediction.

    Returns:
        betas: [num_timesteps], float32
    """
    schedule_type = schedule_type.lower()

    if schedule_type == "linear":
        betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            dtype=torch.float64,
        )

    elif schedule_type == "cosine":
        betas = cosine_beta_schedule(
            num_timesteps=num_timesteps,
            cosine_s=cosine_s,
            max_beta=max_beta,
        )

    else:
        raise ValueError(
            f"Unknown schedule_type={schedule_type}. "
            "Use 'linear' or 'cosine'."
        )

    return betas.float()


def cosine_beta_schedule(
    num_timesteps: int,
    cosine_s: float = 0.008,
    max_beta: float = 0.999,
) -> torch.Tensor:
    """
    Cosine beta schedule
    Instead of directly defining beta_t, we define alpha_bar(t)
    using a cosine curve, then derive beta_t.
    """
    steps = num_timesteps + 1

    x = torch.linspace(
        0,
        num_timesteps,
        steps,
        dtype=torch.float64,
    )

    alphas_cumprod = torch.cos(
        ((x / num_timesteps) + cosine_s)
        / (1.0 + cosine_s)
        * math.pi
        * 0.5
    ) ** 2

    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1.0 - (
        alphas_cumprod[1:] / alphas_cumprod[:-1]
    )

    betas = torch.clamp(
        betas,
        min=1e-8,
        max=max_beta,
    )

    return betas


def create_noise_schedule(
    schedule_type: str = "cosine",
    num_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    cosine_s: float = 0.008,
    max_beta: float = 0.999,
) -> NoiseSchedule:
    """
    all precomputed schedule tensors needed for DDPM training and sampling.
    """
    betas = make_beta_schedule(
        schedule_type=schedule_type,
        num_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        cosine_s=cosine_s,
        max_beta=max_beta,
    )

    alphas = 1.0 - betas

    alphas_cumprod = torch.cumprod(
        alphas,
        dim=0,
    )

    alphas_cumprod_prev = torch.cat(
        [
            torch.ones(1, dtype=alphas_cumprod.dtype),
            alphas_cumprod[:-1],
        ],
        dim=0,
    )

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

    sqrt_one_minus_alphas_cumprod = torch.sqrt(
        1.0 - alphas_cumprod
    )

    log_one_minus_alphas_cumprod = torch.log(
        torch.clamp(
            1.0 - alphas_cumprod,
            min=1e-20,
        )
    )

    sqrt_recip_alphas_cumprod = torch.sqrt(
        1.0 / alphas_cumprod
    )

    sqrt_recipm1_alphas_cumprod = torch.sqrt(
        1.0 / alphas_cumprod - 1.0
    )

    # Posterior q(z_{t-1} | z_t, z_0)
    posterior_variance = (
        betas
        * (1.0 - alphas_cumprod_prev)
        / (1.0 - alphas_cumprod)
    )

    posterior_log_variance_clipped = torch.log(
        torch.clamp(
            posterior_variance,
            min=1e-20,
        )
    )

    posterior_mean_coef1 = (
        betas
        * torch.sqrt(alphas_cumprod_prev)
        / (1.0 - alphas_cumprod)
    )

    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev)
        * torch.sqrt(alphas)
        / (1.0 - alphas_cumprod)
    )

    return NoiseSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        log_one_minus_alphas_cumprod=log_one_minus_alphas_cumprod,
        sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
        posterior_variance=posterior_variance,
        posterior_log_variance_clipped=posterior_log_variance_clipped,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
        num_timesteps=num_timesteps,
        schedule_type=schedule_type,
    )


def extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    broadcast_shape: tuple[int, ...],
) -> torch.Tensor:
    """
    Extract values[t] and reshape for broadcasting.

    Args:
        values:
            Schedule tensor with shape [T].

        timesteps:
            Long tensor with shape [B].

        broadcast_shape:
            Shape of target tensor, e.g. z_t.shape = [B, C, H, W].

    Returns:
        Tensor with shape [B, 1, 1, 1], broadcastable to broadcast_shape.

    Example:
        sqrt_alpha_bar_t = extract(
            schedule.sqrt_alphas_cumprod,
            t,
            z_0.shape,
        )
    """
    if timesteps.dtype != torch.long:
        timesteps = timesteps.long()

    out = values.gather(
        dim=0,
        index=timesteps,
    )

    while len(out.shape) < len(broadcast_shape):
        out = out[..., None]

    return out