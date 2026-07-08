from __future__ import annotations

import torch
from torch import nn

from src.network.autoencoder.encoder import Encoder
from src.network.autoencoder.decoder import Decoder
from src.network.autoencoder.distributions import DiagonalGaussianDistribution


class AutoencoderKL(nn.Module):
    """
    VAE / AutoencoderKL wrapper
        posterior = vae.encode(x)
        z = posterior.sample()
        x_recon = vae.decode(z)

    Or directly:

        x_recon, posterior, z = vae(x)

    Input image range:
        x in [-1, 1]

    Output image range:
        x_recon in [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 8,
        base_channels: int = 128,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 3,
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_heads: int = 4,
        scaling_factor: float = 1.0,
        attention_resolutions: tuple[int, ...] = (32,),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.channel_multipliers = list(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.scaling_factor = scaling_factor

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            use_attention=use_attention,
            attention_heads=attention_heads,
            attention_resolutions= attention_resolutions
        )

        self.decoder = Decoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            use_attention=use_attention,
            attention_heads=attention_heads,
        )

    def encode(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> DiagonalGaussianDistribution:
        """
        Encode image into posterior distribution.
            deterministic:
                If True, posterior.sample() will return mean only.
        Returns:
            DiagonalGaussianDistribution.
        """
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(
            moments=moments,
            deterministic=deterministic,
        )
        return posterior

    def decode(
        self,
        z: torch.Tensor,
        unscale: bool = True,
    ) -> torch.Tensor:
        """
        Decode latent into image
            z:
                Latent tensor [B, latent_channels, H/8, W/8].

            unscale:
                If True, divide by scaling_factor before decoding.

        Returns:
            Reconstructed image in [-1, 1].
        """
        if unscale:
            z = z / self.scaling_factor

        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
    ) -> tuple[torch.Tensor, DiagonalGaussianDistribution, torch.Tensor]:
        """
        Full VAE forward pass.

        Args:
            x:
                Image tensor [B, 3, H, W], normalized to [-1, 1].

            sample_posterior:
                If True:
                    z = posterior.sample()
                If False:
                    z = posterior.mode()

        Returns:
            x_recon:
                Reconstructed image [B, 3, H, W].

            posterior:
                DiagonalGaussianDistribution object.

            z:
                Latent tensor before scaling.
        """
        posterior = self.encode(x)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        x_recon = self.decode(z, unscale=False)

        return x_recon, posterior, z

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        sample_posterior: bool = False,
    ) -> torch.Tensor:
        """
        Convenience method for inference reconstruction.

        By default, uses posterior mode for stable reconstructions.
        """
        x_recon, _, _ = self.forward(
            x,
            sample_posterior=sample_posterior,
        )
        return x_recon

    @torch.no_grad()
    def encode_to_latent(
        self,
        x: torch.Tensor,
        sample_posterior: bool = False,
        scale: bool = True,
    ) -> torch.Tensor:
        """
        Encode image into latent tensor for latent diffusion training.

        Usually for latent caching, use:

            sample_posterior=False

        because the posterior mean is deterministic and stable.

        If scale=True:

            z_scaled = z * scaling_factor

        Stable Diffusion-style LDMs often scale latents before diffusion.
        """
        posterior = self.encode(x)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        if scale:
            z = z * self.scaling_factor

        return z

    @torch.no_grad()
    def decode_from_latent(
        self,
        z: torch.Tensor,
        unscale: bool = True,
    ) -> torch.Tensor:
        """
        Decode latent tensor produced by diffusion model.
        """
        return self.decode(z, unscale=unscale)