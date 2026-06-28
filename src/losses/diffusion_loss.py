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
    ):
        super().__init__()

        if prediction_type not in ["epsilon", "v"]:
            raise ValueError("prediction_type must be 'epsilon' or 'v'")

        self.prediction_type = prediction_type

    def v_target(self, x0, noise, alpha, sigma):
        return alpha * noise - sigma * x0

    def epsilon_target(self, x0, noise):
        return noise

    def forward(
        self,
        model_output: torch.Tensor,
        x0: torch.Tensor,
        noise: torch.Tensor,
        alpha_t: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            model_output:
                predicted v or epsilon [B, C, H, W]

            x0:
                clean latent

            noise:
                sampled noise

            alpha_t, sigma_t:
                diffusion schedule scalars [B, 1, 1, 1]
        """

        if self.prediction_type == "epsilon":
            target = self.epsilon_target(x0, noise)

        else:  # v-prediction
            target = self.v_target(x0, noise, alpha_t, sigma_t)

        loss = F.mse_loss(model_output, target)

        return loss