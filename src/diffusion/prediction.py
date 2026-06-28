from __future__ import annotations

import torch

from src.diffusion.noise_schedule import NoiseSchedule, extract


def predict_x0_from_eps(
    z_t: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    """
    Recover clean latent z_0 from epsilon prediction
    """
    sqrt_recip_alpha_bar = extract(
        schedule.sqrt_recip_alphas_cumprod,
        t,
        z_t.shape,
    )

    sqrt_recipm1_alpha_bar = extract(
        schedule.sqrt_recipm1_alphas_cumprod,
        t,
        z_t.shape,
    )

    return sqrt_recip_alpha_bar * z_t - sqrt_recipm1_alpha_bar * eps


def predict_eps_from_x0(
    z_t: torch.Tensor,
    t: torch.Tensor,
    z_0: torch.Tensor,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    """
    Recover epsilon from clean latent z_0 and noisy latent z_t.

    eps = (z_t - sqrt(alpha_bar_t) * z_0)
          / sqrt(1 - alpha_bar_t)
    """
    sqrt_alpha_bar = extract(
        schedule.sqrt_alphas_cumprod,
        t,
        z_t.shape,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule.sqrt_one_minus_alphas_cumprod,
        t,
        z_t.shape,
    )

    return (z_t - sqrt_alpha_bar * z_0) / sqrt_one_minus_alpha_bar


def get_v_target(
    z_0: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    """
    Compute v-prediction target.

    v-prediction target:

        v = sqrt(alpha_bar_t) * eps
          - sqrt(1 - alpha_bar_t) * z_0
    """
    sqrt_alpha_bar = extract(
        schedule.sqrt_alphas_cumprod,
        t,
        z_0.shape,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule.sqrt_one_minus_alphas_cumprod,
        t,
        z_0.shape,
    )

    v = sqrt_alpha_bar * eps - sqrt_one_minus_alpha_bar * z_0

    return v


def predict_x0_from_v(
    z_t: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    """
    Recover clean latent z_0 from v prediction

    defs:

        z_t = a * z_0 + b * eps
        v   = a * eps - b * z_0

        a = sqrt(alpha_bar_t)
        b = sqrt(1 - alpha_bar_t)

    it gives

        z_0 = a * z_t - b * v
    """
    sqrt_alpha_bar = extract(
        schedule.sqrt_alphas_cumprod,
        t,
        z_t.shape,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule.sqrt_one_minus_alphas_cumprod,
        t,
        z_t.shape,
    )

    z_0 = sqrt_alpha_bar * z_t - sqrt_one_minus_alpha_bar * v

    return z_0


def predict_eps_from_v(
    z_t: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    """
    Recover epsilon from v prediction

    defs:

        z_t = a * z_0 + b * eps
        v   = a * eps - b * z_0

    it gives

        eps = b * z_t + a * v
    """
    sqrt_alpha_bar = extract(
        schedule.sqrt_alphas_cumprod,
        t,
        z_t.shape,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule.sqrt_one_minus_alphas_cumprod,
        t,
        z_t.shape,
    )

    eps = sqrt_one_minus_alpha_bar * z_t + sqrt_alpha_bar * v

    return eps


def model_output_to_x0_and_eps(
    model_output: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    prediction_type: str = "v",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert model output into both:
        z_0 prediction
        epsilon prediction
    """
    prediction_type = prediction_type.lower()

    if prediction_type in {"v", "v_prediction"}:
        z_0 = predict_x0_from_v(
            z_t=z_t,
            t=t,
            v=model_output,
            schedule=schedule,
        )

        eps = predict_eps_from_v(
            z_t=z_t,
            t=t,
            v=model_output,
            schedule=schedule,
        )

    elif prediction_type in {"eps", "epsilon"}:
        eps = model_output

        z_0 = predict_x0_from_eps(
            z_t=z_t,
            t=t,
            eps=eps,
            schedule=schedule,
        )

    elif prediction_type in {"x0", "sample"}:
        z_0 = model_output

        eps = predict_eps_from_x0(
            z_t=z_t,
            t=t,
            z_0=z_0,
            schedule=schedule,
        )

    else:
        raise ValueError(
            f"Unknown prediction_type={prediction_type}. "
            "Use 'v', 'eps', or 'x0'."
        )

    return z_0, eps


def get_training_target(
    z_0: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    prediction_type: str = "v",
) -> torch.Tensor:
    """
    Return the target the U-Net should learn

    For our project, default is:

        prediction_type = "v"

    Then target is:

        v = sqrt(alpha_bar_t) * eps
          - sqrt(1 - alpha_bar_t) * z_0
    """
    prediction_type = prediction_type.lower()

    if prediction_type in {"v", "v_prediction"}:
        return get_v_target(
            z_0=z_0,
            eps=eps,
            t=t,
            schedule=schedule,
        )

    if prediction_type in {"eps", "epsilon"}:
        return eps

    if prediction_type in {"x0", "sample"}:
        return z_0

    raise ValueError(
        f"Unknown prediction_type={prediction_type}. "
        "Use 'v', 'eps', or 'x0'."
    )