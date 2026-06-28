from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.losses.diffusion_loss import DiffusionLoss
from src.diffusion.noise_schedule import NoiseSchedule, create_noise_schedule, extract
from src.diffusion.prediction import (
    get_training_target,
    model_output_to_x0_and_eps,
)


@dataclass
class DiffusionTrainingOutput:
    loss: torch.Tensor
    simple_loss: torch.Tensor
    model_output: torch.Tensor
    target: torch.Tensor
    z_t: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor


class GaussianDiffusion:
    """
    Core latent diffusion utilities.

    This handles:

        - sampling timesteps
        - adding noise q(z_t | z_0)
        - creating v-prediction targets
        - computing diffusion training loss
        - computing DDPM posterior mean/variance for sampling
    """

    def __init__(
        self,
        schedule: NoiseSchedule | None = None,
        schedule_type: str = "cosine",
        num_timesteps: int = 1000,
        prediction_type: str = "v",
        loss_type: str = "mse",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_s: float = 0.008,
        max_beta: float = 0.999,
    ):
        if schedule is None:
            schedule = create_noise_schedule(
                schedule_type=schedule_type,
                num_timesteps=num_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                cosine_s=cosine_s,
                max_beta=max_beta,
            )

        self.schedule = schedule
        self.prediction_type = prediction_type.lower()
        self.diffusion_loss = DiffusionLoss(
            prediction_type=self.prediction_type
        )
        self.loss_type = loss_type.lower()

        if self.prediction_type not in {"v", "v_prediction", "eps", "epsilon", "x0", "sample"}:
            raise ValueError(
                f"Unknown prediction_type={prediction_type}. "
                "Use 'v', 'eps', or 'x0'."
            )

        if self.loss_type not in {"mse", "l1", "huber"}:
            raise ValueError(
                f"Unknown loss_type={loss_type}. "
                "Use 'mse', 'l1', or 'huber'."
            )

    @property
    def num_timesteps(self) -> int:
        return self.schedule.num_timesteps

    def to(self, device: torch.device | str) -> "GaussianDiffusion":
        self.schedule = self.schedule.to(device)
        return self

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """
        Sample random diffusion timesteps.

        Returns:
            t: [B], values in [0, num_timesteps - 1]
        """
        return torch.randint(
            low=0,
            high=self.num_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def q_sample(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process:

            q(z_t | z_0)

        Formula:

            z_t = sqrt(alpha_bar_t) * z_0
                + sqrt(1 - alpha_bar_t) * eps

        Args:
            z_0:
                Clean latent [B, C, H, W].

            t:
                Timesteps [B].

            noise:
                Optional epsilon noise. If None, sampled from N(0, I).

        Returns:
            z_t:
                Noisy latent.

            noise:
                The epsilon noise used.
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        sqrt_alpha_bar = extract(
            self.schedule.sqrt_alphas_cumprod,
            t,
            z_0.shape,
        )

        sqrt_one_minus_alpha_bar = extract(
            self.schedule.sqrt_one_minus_alphas_cumprod,
            t,
            z_0.shape,
        )

        z_t = sqrt_alpha_bar * z_0 + sqrt_one_minus_alpha_bar * noise

        return z_t, noise

    def training_target(
        self,
        z_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get target for current prediction type
        """
        return get_training_target(
            z_0=z_0,
            eps=noise,
            t=t,
            schedule=self.schedule,
            prediction_type=self.prediction_type,
        )

    def p_losses(
        self,
        model,
        z_0: torch.Tensor,
        context: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        model_kwargs: dict | None = None,
    ) -> DiffusionTrainingOutput:
        """
        Full diffusion training step using  loss module.
        """
        if model_kwargs is None:
            model_kwargs = {}

        batch_size = z_0.shape[0]
        device = z_0.device

        if t is None:
            t = self.sample_timesteps(batch_size, device)

        z_t, noise = self.q_sample(
            z_0=z_0,
            t=t,
            noise=noise,
        )

        if context is None:
            model_output = model(z_t, t, **model_kwargs)
        else:
            model_output = model(z_t, t, context=context, **model_kwargs)

        alpha_t = extract(
            self.schedule.sqrt_alphas_cumprod,
            t,
            z_0.shape,
        )

        sigma_t = extract(
            self.schedule.sqrt_one_minus_alphas_cumprod,
            t,
            z_0.shape,
        )

        loss = self.diffusion_loss(
            model_output=model_output,
            x0=z_0,
            noise=noise,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
        )

        return DiffusionTrainingOutput(
            loss=loss,
            simple_loss=loss.detach(),
            model_output=model_output,
            target=None,  
            z_t=z_t,
            noise=noise,
            timesteps=t,
        )

    def predict_x0_and_eps(
        self,
        model_output: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert model output to:

            z_0 prediction
            epsilon prediction
        """
        return model_output_to_x0_and_eps(
            model_output=model_output,
            z_t=z_t,
            t=t,
            schedule=self.schedule,
            prediction_type=self.prediction_type,
        )

    def q_posterior(
        self,
        z_0: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior:

            q(z_{t-1} | z_t, z_0)

        Returns:
            posterior_mean
            posterior_variance
            posterior_log_variance_clipped
        """
        posterior_mean_coef1 = extract(
            self.schedule.posterior_mean_coef1,
            t,
            z_t.shape,
        )

        posterior_mean_coef2 = extract(
            self.schedule.posterior_mean_coef2,
            t,
            z_t.shape,
        )

        posterior_mean = (
            posterior_mean_coef1 * z_0
            + posterior_mean_coef2 * z_t
        )

        posterior_variance = extract(
            self.schedule.posterior_variance,
            t,
            z_t.shape,
        )

        posterior_log_variance_clipped = extract(
            self.schedule.posterior_log_variance_clipped,
            t,
            z_t.shape,
        )

        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
        )

    @torch.no_grad()
    def p_mean_variance(
        self,
        model,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None = None,
        clip_denoised: bool = False,
        model_kwargs: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        One reverse-process prediction.

        Model predicts v/eps/x0.
        We convert to predicted z_0
        """
        if model_kwargs is None:
            model_kwargs = {}

        if context is None:
            model_output = model(
                z_t,
                t,
                **model_kwargs,
            )
        else:
            model_output = model(
                z_t,
                t,
                context=context,
                **model_kwargs,
            )

        pred_z0, pred_eps = self.predict_x0_and_eps(
            model_output=model_output,
            z_t=z_t,
            t=t,
        )

        if clip_denoised:
            pred_z0 = pred_z0.clamp(-1.0, 1.0)

        (
            posterior_mean,
            posterior_variance,
            posterior_log_variance,
        ) = self.q_posterior(
            z_0=pred_z0,
            z_t=z_t,
            t=t,
        )

        return {
            "mean": posterior_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_z0": pred_z0,
            "pred_eps": pred_eps,
            "model_output": model_output,
        }

    @torch.no_grad()
    def p_sample(
        self,
        model,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None = None,
        clip_denoised: bool = False,
        model_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """
        This is one reverse step
        """
        out = self.p_mean_variance(
            model=model,
            z_t=z_t,
            t=t,
            context=context,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )

        noise = torch.randn_like(z_t)

        # No noise when t == 0.
        nonzero_mask = (t != 0).float()

        while len(nonzero_mask.shape) < len(z_t.shape):
            nonzero_mask = nonzero_mask[..., None]

        z_prev = (
            out["mean"]
            + nonzero_mask
            * torch.exp(0.5 * out["log_variance"])
            * noise
        )

        return z_prev