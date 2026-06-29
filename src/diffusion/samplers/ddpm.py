from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm

from src.diffusion.gaussian_diffusion import GaussianDiffusion


@dataclass
class DDPMSamplerOutput:
    latents: torch.Tensor
    trajectory: list[torch.Tensor] | None = None


class DDPMSampler:
    """
    DDPM ancestral sampler.

    This sampler uses the learned reverse process:

        z_T ~ N(0, I)
        z_T -> z_{T-1} -> ... -> z_0
        
    Supports classifier-free guidance if both conditional and unconditional
    context are provided.
    """

    def __init__(
        self,
        diffusion: GaussianDiffusion,
    ):
        self.diffusion = diffusion

    @torch.no_grad()
    def predict_model_output(
        self,
        model,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        uncond_context: torch.Tensor | None = None,
        uncond_attention_mask: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict model output with optional classifier-free guidance.

        If guidance_scale == 1 or uncond_context is None:
            normal conditional prediction.

        If guidance_scale > 1:
            output = uncond + scale * (cond - uncond)
        """
        if uncond_context is None or guidance_scale == 1.0:
            if context is None:
                return model(
                    z_t,
                    t,
                )

            return model(
                z_t,
                t,
                context=context,
                attention_mask=attention_mask,
            )

        # Conditional prediction
        cond_output = model(
            z_t,
            t,
            context=context,
            attention_mask=attention_mask,
        )

        # Unconditional prediction
        uncond_output = model(
            z_t,
            t,
            context=uncond_context,
            attention_mask=uncond_attention_mask,
        )

        return uncond_output + guidance_scale * (cond_output - uncond_output)

    @torch.no_grad()
    def p_mean_variance_with_cfg(
        self,
        model,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        uncond_context: torch.Tensor | None = None,
        uncond_attention_mask: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        clip_denoised: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Same as GaussianDiffusion.p_mean_variance, but supports CFG.
        """
        model_output = self.predict_model_output(
            model=model,
            z_t=z_t,
            t=t,
            context=context,
            attention_mask=attention_mask,
            uncond_context=uncond_context,
            uncond_attention_mask=uncond_attention_mask,
            guidance_scale=guidance_scale,
        )

        pred_z0, pred_eps = self.diffusion.predict_x0_and_eps(
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
        ) = self.diffusion.q_posterior(
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
    def sample(
        self,
        model,
        shape: tuple[int, int, int, int],
        device: torch.device | str,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        uncond_context: torch.Tensor | None = None,
        uncond_attention_mask: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        clip_denoised: bool = False,
        return_trajectory: bool = False,
        progress: bool = True,
    ) -> DDPMSamplerOutput:
        """
        Generate clean latents from pure noise.

        Args:
            shape:
                Usually [B, 8, 32, 32] for your model.

            context:
                Conditional CLIP text context.

            uncond_context:
                Empty-prompt CLIP context for CFG.

            guidance_scale:
                CFG scale. Common values: 3.0 to 7.5.
        """
        device = torch.device(device)

        model.eval()

        z_t = torch.randn(
            shape,
            device=device,
        )

        trajectory = [] if return_trajectory else None

        timesteps = reversed(range(self.diffusion.num_timesteps))

        if progress:
            timesteps = tqdm(
                timesteps,
                total=self.diffusion.num_timesteps,
                desc="DDPM sampling",
            )

        for step in timesteps:
            t = torch.full(
                (shape[0],),
                step,
                device=device,
                dtype=torch.long,
            )

            out = self.p_mean_variance_with_cfg(
                model=model,
                z_t=z_t,
                t=t,
                context=context,
                attention_mask=attention_mask,
                uncond_context=uncond_context,
                uncond_attention_mask=uncond_attention_mask,
                guidance_scale=guidance_scale,
                clip_denoised=clip_denoised,
            )

            noise = torch.randn_like(z_t)

            nonzero_mask = (t != 0).float()

            while len(nonzero_mask.shape) < len(z_t.shape):
                nonzero_mask = nonzero_mask[..., None]

            z_t = (
                out["mean"]
                + nonzero_mask
                * torch.exp(0.5 * out["log_variance"])
                * noise
            )

            if return_trajectory:
                trajectory.append(z_t.detach().cpu())

        return DDPMSamplerOutput(
            latents=z_t,
            trajectory=trajectory,
        )