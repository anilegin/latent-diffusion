from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class VAELossOutput:
    total_loss: torch.Tensor
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    perceptual_loss: torch.Tensor


class VAELoss(nn.Module):
    """
    VAE loss:
        total =
            recon_weight * reconstruction_loss
          + kl_weight * KL
          + perceptual_weight * LPIPS

    Inputs are expected to be in [-1, 1]
    """

    def __init__(
        self,
        recon_loss_type: str = "l1",
        recon_weight: float = 1.0,
        kl_weight: float = 1e-6,
        perceptual_weight: float = 0.0,
        use_lpips: bool = False,
        lpips_net: str = "vgg",
    ):
        super().__init__()

        if recon_loss_type not in {"l1", "mse"}:
            raise ValueError(
                f"Unknown recon_loss_type={recon_loss_type}. "
                "Use 'l1' or 'mse'."
            )

        self.recon_loss_type = recon_loss_type
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.use_lpips = use_lpips

        self.lpips_model = None

        if use_lpips:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError(
                    "LPIPS is enabled but package 'lpips' is not installed. "
                    "Install it with: pip install lpips"
                ) from exc

            self.lpips_model = lpips.LPIPS(net=lpips_net)
            self.lpips_model.eval()

            for p in self.lpips_model.parameters():
                p.requires_grad = False

    def reconstruction_loss(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.recon_loss_type == "l1":
            return F.l1_loss(x_recon, x)

        if self.recon_loss_type == "mse":
            return F.mse_loss(x_recon, x)

        raise RuntimeError("Invalid reconstruction loss type.")

    def perceptual_loss(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_lpips or self.lpips_model is None:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        # LPIPS expects images in [-1, 1], which matches our transform.
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.lpips_model(
                x_recon.float(),
                x.float(),
            ).mean()

        return loss.to(dtype=x.dtype)

    def forward(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        posterior,
        kl_weight: float | None = None,
    ) -> VAELossOutput:
        """
        Args:
            x_recon:
                Reconstructed image [B, 3, H, W], in [-1, 1].

            x:
                Target image [B, 3, H, W], in [-1, 1].

            posterior:
                DiagonalGaussianDistribution from vae.encode(x).

            kl_weight:
                Optional current KL weight. Useful for KL warmup.

        Returns:
            VAELossOutput.
        """
        current_kl_weight = self.kl_weight if kl_weight is None else kl_weight

        recon = self.reconstruction_loss(x_recon, x)

        # posterior.kl() returns [B], already summed over latent dimensions.
        kl = posterior.kl().mean()

        perceptual = self.perceptual_loss(x_recon, x)

        total = (
            self.recon_weight * recon
            + current_kl_weight * kl
            + self.perceptual_weight * perceptual
        )

        return VAELossOutput(
            total_loss=total,
            recon_loss=recon.detach(),
            kl_loss=kl.detach(),
            perceptual_loss=perceptual.detach(),
        )