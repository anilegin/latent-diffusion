from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm

from src.diffusion.gaussian_diffusion import GaussianDiffusion


@dataclass
class DDIMSamplerOutput:
    latents: torch.Tensor
    trajectory: list[torch.Tensor] | None = None


class DDIMSampler:
    """
    DDIM sampler.

    eta controls stochasticity:

        eta = 0.0 -> deterministic DDIM
        eta > 0.0 -> more stochastic
    """

    def __init__(
        self,
        diffusion: GaussianDiffusion,
    ):
        self.diffusion = diffusion

    def make_timesteps(
        self,
        num_steps: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """
        Select evenly spaced timesteps from the original diffusion schedule.

        Example:
            original T = 1000
            num_steps = 50

        returns 50 timesteps descending from high noise to low noise.
        """
        if num_steps > self.diffusion.num_timesteps:
            raise ValueError(
                f"num_steps={num_steps} cannot be larger than "
                f"num_timesteps={self.diffusion.num_timesteps}"
            )

        timesteps = torch.linspace(
            0,
            self.diffusion.num_timesteps - 1,
            steps=num_steps,
            device=device,
        ).long()

        timesteps = torch.flip(timesteps, dims=[0])

        return timesteps

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
        Predict v/eps/x0 with optional classifier-free guidance.
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

        cond_output = model(
            z_t,
            t,
            context=context,
            attention_mask=attention_mask,
        )

        uncond_output = model(
            z_t,
            t,
            context=uncond_context,
            attention_mask=uncond_attention_mask,
        )

        return uncond_output + guidance_scale * (cond_output - uncond_output)

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
        num_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = False,
        return_trajectory: bool = False,
        progress: bool = True,
    ) -> DDIMSamplerOutput:
        """
        DDIM sampling.

        Returns:
            clean latent estimate z_0 at the final step.
        """
        device = torch.device(device)
        model.eval()

        z_t = torch.randn(
            shape,
            device=device,
        )

        trajectory = [] if return_trajectory else None

        ddim_timesteps = self.make_timesteps(
            num_steps=num_steps,
            device=device,
        )

        if progress:
            iterator = tqdm(
                range(len(ddim_timesteps)),
                desc=f"DDIM sampling ({num_steps} steps)",
            )
        else:
            iterator = range(len(ddim_timesteps))

        for i in iterator:
            step = ddim_timesteps[i]

            t = torch.full(
                (shape[0],),
                int(step.item()),
                device=device,
                dtype=torch.long,
            )

            if i == len(ddim_timesteps) - 1:
                prev_step = torch.tensor(
                    -1,
                    device=device,
                    dtype=torch.long,
                )
            else:
                prev_step = ddim_timesteps[i + 1]

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

            alpha_t = self.diffusion.schedule.alphas_cumprod[t]
            alpha_t = alpha_t.view(shape[0], 1, 1, 1)

            if prev_step.item() < 0:
                alpha_prev = torch.ones_like(alpha_t)
            else:
                alpha_prev = self.diffusion.schedule.alphas_cumprod[
                    torch.full(
                        (shape[0],),
                        int(prev_step.item()),
                        device=device,
                        dtype=torch.long,
                    )
                ]
                alpha_prev = alpha_prev.view(shape[0], 1, 1, 1)

            sigma_t = eta * torch.sqrt(
                (1.0 - alpha_prev)
                / (1.0 - alpha_t)
                * (1.0 - alpha_t / alpha_prev)
            )

            # Direction pointing to z_t.
            dir_xt = torch.sqrt(
                torch.clamp(
                    1.0 - alpha_prev - sigma_t ** 2,
                    min=0.0,
                )
            ) * pred_eps

            noise = sigma_t * torch.randn_like(z_t)

            z_t = torch.sqrt(alpha_prev) * pred_z0 + dir_xt + noise

            if return_trajectory:
                trajectory.append(z_t.detach().cpu())

        return DDIMSamplerOutput(
            latents=z_t,
            trajectory=trajectory,
        )