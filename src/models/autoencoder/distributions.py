from __future__ import annotations

import torch


class DiagonalGaussianDistribution:
    """
    Diagonal Gaussian posterior used by the VAE.

    The encoder predicts a tensor called `moments`:

        moments: [B, 2 * latent_channels, H, W]

    We split it into:

        mean:   [B, latent_channels, H, W]
        logvar: [B, latent_channels, H, W]

    Then sample:

        z = mean + std * eps

    where:

        std = exp(0.5 * logvar)
        eps ~ N(0, I)
    """

    def __init__(
        self,
        moments: torch.Tensor,
        deterministic: bool = False,
        logvar_min: float = -30.0,
        logvar_max: float = 20.0,
    ):
        self.moments = moments
        self.deterministic = deterministic

        self.mean, self.logvar = torch.chunk(moments, chunks=2, dim=1)

        # Clamp log-variance for numerical stability
        self.logvar = torch.clamp(self.logvar, min=logvar_min, max=logvar_max)

        self.var = torch.exp(self.logvar)
        self.std = torch.exp(0.5 * self.logvar)

        if self.deterministic:
            self.var = torch.zeros_like(self.mean)
            self.std = torch.zeros_like(self.mean)

    def sample(self) -> torch.Tensor:
        """
        Reparameterized sampling:

            z = mean + std * eps
        """
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def mode(self) -> torch.Tensor:
        """
        Most likely latent value
        """
        return self.mean

    def kl(self) -> torch.Tensor:
        """
        KL divergence from posterior q(z|x) to standard normal N(0, I).

        For diagonal Gaussian:

            KL(q || N(0,I))
            = 0.5 * (mean^2 + var - 1 - logvar)

        Returns:
            Per-sample KL with shape [B].
        """
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device)

        kl = 0.5 * (
            torch.pow(self.mean, 2)
            + self.var
            - 1.0
            - self.logvar
        )

        # Sum over latent channels and spatial dimensions.
        return torch.sum(kl, dim=[1, 2, 3])

    def nll(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Negative log likelihood of `sample` under this posterior.

        Mostly useful for debugging, not essential for our VAE training loop.

        Returns:
            Per-sample NLL with shape [B].
        """
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device)

        log_two_pi = torch.log(
            torch.tensor(2.0 * torch.pi, device=sample.device, dtype=sample.dtype)
        )

        nll = 0.5 * (
            log_two_pi
            + self.logvar
            + torch.pow(sample - self.mean, 2) / self.var
        )

        return torch.sum(nll, dim=[1, 2, 3])