from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):
    """
    Extract coefficients at timestep t
    a: [T]
    t: [B]
    returns: [B, 1, 1, 1]
    """
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.view(b, *((1,) * (len(x_shape) - 1)))


class DiffusionLoss(nn.Module):
    """
    Diffusion loss supporting:
        - epsilon prediction
        - v prediction 

    v-prediction:
        v = alpha_t * epsilon - sigma_t * x0
    """

    def __init__(
        self,
        prediction_type: str = "v",  # "epsilon" or "v"
        loss_type: str = "mse",
        snr_gamma: float | None = None,
        snr_weighting: str = "none",
        normalize_snr_weights: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()

        prediction_type = prediction_type.lower()
        loss_type = loss_type.lower()
        snr_weighting = snr_weighting.lower()

        if prediction_type in {"eps", "epsilon"}:
            prediction_type = "epsilon"

        elif prediction_type in {"v", "v_prediction"}:
            prediction_type = "v"

        elif prediction_type in {"x0", "sample"}:
            prediction_type = "x0"

        else:
            raise ValueError(
                "prediction_type must be 'epsilon', 'v', or 'x0'"
            )

        if loss_type not in {"mse", "l1", "huber"}:
            raise ValueError(
                "loss_type must be 'mse', 'l1', or 'huber'"
            )

        if snr_weighting not in {"none", "min_snr"}:
            raise ValueError(
                "snr_weighting must be 'none' or 'min_snr'"
            )

        if snr_weighting == "min_snr" and snr_gamma is None:
            raise ValueError(
                "snr_gamma must be set when snr_weighting='min_snr'"
            )

        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.snr_gamma = snr_gamma
        self.snr_weighting = snr_weighting
        self.normalize_snr_weights = normalize_snr_weights
        self.eps = eps

    def v_target(self, x0, noise, alpha, sigma):
        return alpha * noise - sigma * x0

    def epsilon_target(self, x0, noise):
        return noise

    def x0_target(self, x0):
        return x0

    def get_target(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        alpha_t: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            return self.epsilon_target(x0, noise)

        if self.prediction_type == "v":
            return self.v_target(x0, noise, alpha_t, sigma_t)

        if self.prediction_type == "x0":
            return self.x0_target(x0)

        raise RuntimeError("Invalid prediction type.")

    def elementwise_loss(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(
                model_output,
                target,
                reduction="none",
            )

        if self.loss_type == "l1":
            return F.l1_loss(
                model_output,
                target,
                reduction="none",
            )

        if self.loss_type == "huber":
            return F.smooth_l1_loss(
                model_output,
                target,
                reduction="none",
            )

        raise RuntimeError("Invalid loss type.")

    def get_snr_weights(
        self,
        snr: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Returns per-sample SNR weights.

        snr:
            [B]

        For Min-SNR:
            epsilon prediction:
                weight = min(snr, gamma) / snr

            v prediction:
                weight = min(snr, gamma) / (snr + 1)

            x0 prediction:
                weight = min(snr, gamma)
        """
        if self.snr_weighting == "none":
            return None

        if self.snr_weighting == "min_snr":
            if self.snr_gamma is None:
                raise RuntimeError("snr_gamma is required for min_snr weighting.")

            snr = snr.float().clamp(min=self.eps)

            gamma = torch.full_like(
                snr,
                fill_value=float(self.snr_gamma),
            )

            clipped_snr = torch.minimum(
                snr,
                gamma,
            )

            if self.prediction_type == "epsilon":
                weights = clipped_snr / snr

            elif self.prediction_type == "v":
                weights = clipped_snr / (snr + 1.0)

            elif self.prediction_type == "x0":
                weights = clipped_snr

            else:
                raise RuntimeError("Invalid prediction type.")

            if self.normalize_snr_weights:
                weights = weights / weights.mean().clamp(min=self.eps)

            return weights

        raise RuntimeError("Invalid SNR weighting type.")

    def forward(
        self,
        model_output: torch.Tensor,
        x0: torch.Tensor,
        noise: torch.Tensor,
        alpha_t: torch.Tensor,
        sigma_t: torch.Tensor,
        snr: torch.Tensor | None = None,
        return_dict: bool = False,
    ):

        target = self.get_target(
            x0=x0,
            noise=noise,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
        )

        loss = self.elementwise_loss(
            model_output=model_output,
            target=target,
        )

        # [B, C, H, W] -> [B]
        per_sample_loss = loss.mean(
            dim=tuple(range(1, loss.ndim)),
        )

        raw_loss = per_sample_loss.mean()

        weights = None

        if self.snr_weighting != "none":
            if snr is None:
                raise ValueError(
                    "snr must be passed when SNR weighting is enabled."
                )

            weights = self.get_snr_weights(snr)

            if weights is not None:
                per_sample_loss = per_sample_loss * weights.to(per_sample_loss.device)

        weighted_loss = per_sample_loss.mean()

        if return_dict:
            out = {
                "loss": weighted_loss,
                "raw_loss": raw_loss.detach(),
            }

            if weights is not None:
                out["snr_weight_mean"] = weights.mean().detach()
                out["snr_weight_min"] = weights.min().detach()
                out["snr_weight_max"] = weights.max().detach()

            return out

        return weighted_loss