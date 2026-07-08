# Latent Diffusion

**Model checkpoints:**
- VAE: [https://huggingface.co/anilegin/lightweight-diffusion-vae](https://huggingface.co/anilegin/lightweight-diffusion-vae)
- Latent Diffusion Model: [https://huggingface.co/anilegin/lightweight-diffusion-ldm](https://huggingface.co/anilegin/lightweight-diffusion-ldm)

This repository implements a custom latent diffusion text-to-image pipeline trained on COCO 2017 captions. It includes a custom VAE, a CLIP-conditioned latent diffusion U-Net, and DDIM/DDPM sampling.

The project is intended as a compact research codebase for understanding and improving latent diffusion systems end to end, rather than as a production Stable Diffusion replacement.

---

## Quick Start

You can easily try out the custom Diffusion model through [Colab](https://colab.research.google.com/drive/1f4ZXEfXSt6tzPHSRIoija2U8gsWwYd-I?usp=sharing](Colab:))

The released lightweight LDM checkpoint can also be used directly from Hugging Face:

```bash
git clone https://huggingface.co/anilegin/lightweight-diffusion-ldm
cd lightweight-diffusion-ldm
pip install -r requirements.txt
```

```bash
python inference.py \
  --prompt "a small dog sitting on a red couch" \
  --sampler ddim \
  --num-steps 150 \
  --guidance-scale 3.0 \
  --precision fp16 \
  --output-dir outputs/example
```

A good first sampling setup is:

```text
sampler: ddim
num_steps: 150
guidance_scale: 3.0
precision: fp16
```

In fp16 inference, this released checkpoint was measured at approximately **3.17 GB peak VRAM**.

For this custom COCO-trained model, moderate classifier-free guidance values usually work better than very high values. Useful values to try are:

```text
1.0, 2.0, 3.0
```

---

## Local Repository Sampling

Inside this repository, use `scripts/sample_text2image.py` to generate images from a local trained latent diffusion checkpoint:

```bash
python scripts/sample_text2image.py \
  --config configs/experiments/sample_coco_256.yaml \
  --ldm-checkpoint outputs/ldm/ldm_coco_256_vae8_strong_vpred_ft/checkpoints/best.pt \
  --vae-checkpoint outputs/vae/vae_coco_256_small/checkpoints/best.pt \
  --prompt "a dog running through grass" \
  --num-steps 100 \
  --guidance-scale 2.0 \
  --seed 42
```

Generated images are saved under:

```text
outputs/samples/ldm_text2image/
```

---

## Overview

The pipeline follows the standard latent diffusion structure:

```text
COCO image
  -> VAE encoder
  -> latent z
  -> diffusion model learns denoising in latent space
  -> VAE decoder
  -> generated image
```

The model is trained on 256×256 COCO images. The VAE downsamples images by a factor of 8, so the diffusion model operates on 32×32 latent tensors.

---

## VAE

The custom VAE is an `AutoencoderKL` trained from scratch on COCO 2017 images.

Its role is to compress images:

```text
image [3, 256, 256] -> latent [8, 32, 32]
```

and reconstruct images:

```text
latent [8, 32, 32] -> image [3, 256, 256]
```

The trained custom VAE uses 8 latent channels. The estimated latent scaling factor is:

```text
scaling_factor = 1.0320695526096337
```

During diffusion training, encoded latents are multiplied by this scaling factor. During sampling, generated latents must be divided by the same scaling factor before VAE decoding.

---

## VAE Results

Evaluation was performed on the same subset setting: 5,000 COCO `val2017` images resized to 256×256.

| Model | Params | Param memory | rFID ↓ | PSNR ↑ | SSIM ↑ | Perceptual ↓ | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Custom COCO VAE | 98.19M | 187.3 MiB fp16 | 8.91 | 30.1 ± 3.5 | 0.88 ± 0.07 | LPIPS 0.13 ± 0.04 | Trained from scratch for this project; 8 latent channels |
| LDM text2image VQ-VAE | 83.65M | 159.6 MiB fp16| **4.51** | 25.55 ± 3.62 | 0.757 ± 0.114 | LPIPS 0.143 ± 0.042 | `CompVis/ldm-text2im-large-256`, subfolder `vqvae` |
| SD VAE original | 83.65M | 159.6 MiB fp16 | 4.99 | 23.4 ± 3.8 | 0.69 ± 0.14 | PSIM 1.01 ± 0.28 | Original KL-f4 VAE used in Stable Diffusion 2 |

Reference for the Stable Diffusion VAE and ldm-text2im-large:

```text
https://huggingface.co/stabilityai/sd-vae-ft-mse-original
https://huggingface.co/CompVis/ldm-text2im-large-256 
```

Note: the SD VAE row reports PSIM, while the other rows report LPIPS, so those perceptual values should not be treated as exactly the same metric.

---

## Latent Diffusion Model

The latent diffusion model is a text-conditioned U-Net trained with v-prediction. It receives:

```text
noisy latent z_t
diffusion timestep t
CLIP text embeddings
```

and predicts the denoising target.

Text conditioning is provided by a frozen CLIP text encoder. Classifier-free guidance is supported by randomly replacing some captions with the empty string during training.

Several U-Net sizes were tested:

| Model name | Parameters | Pipeline Memory | Description |
|---|---:|---:|---|
| Light | ~40M | - | Smallest custom U-Net |
| Medium | ~160M | - | Mid-size custom U-Net |
| Strong | ~450M | ~2.23 GB | Best custom U-Net in this evaluation |
| LDM text2image | ~872M | ~6.15 GB  | Reference `CompVis/ldm-text2im-large-256` model |

---

## Diffusion Results

The following evaluations compare generated images against COCO `val2017`.

Evaluation protocol:

```text
num_real = 500
num_fake = 500
resolution = 256x256
metrics = FID, KID, Inception Score, CLIP score
generation_batch_size = 32
num_batches = 16
```

Lower FID/KID is better. Higher Inception Score and CLIP score are better.

Timing is reported as end-to-end evaluation time averaged over 16 generation batches. `Avg time / 32 imgs` is the average time for one batch of 32 generated images, and `Approx time / img` is derived from the 500-image evaluation total. These are approximate single-A100 evaluation timings and may vary with implementation details, caching, and I/O.

### Overall comparison

| Model | Parameters | Best-FID setting | FID ↓ | KID ↓ | IS ↑ | CLIP ↑ | Time / 32 imgs | Time / img | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Light | ~40M | DDIM, 100 steps, CFG 5 | 135.13 | 0.0154 | 8.99 | 21.53 | 3.81 s | 0.122 s | Smallest custom U-Net; fastest custom model |
| Medium | ~160M | DDIM, 100 steps, CFG 5 | 117.35 | 0.0120 | 10.33 | 23.04 | 6.20 s | 0.198 s | Mid-size custom U-Net |
| Strong | ~450M | DDIM, 100 steps, CFG 5 | 105.85 | 0.0076 | 12.92 | 24.45 | 8.59 s | 0.275 s | Best custom model in this evaluation |
| LDM text2image | ~872M | DDIM, 100 steps, CFG 5 | 113.94 | 0.0117 | 12.69 | 25.19 | 10.95 s | 0.351 s | Reference CompVis LDM text-to-image model |

The Strong custom model gives the best FID among the custom models and is also slightly better than the 872M-parameter LDM text2image reference on FID in this 500-image evaluation. The LDM text2image reference still obtains the strongest CLIP scores and Inception Scores, especially with DDPM.


### Timing overview

| Model | Params | Sampler | Steps | Avg time / 32 imgs | Approx time / img |
|---|---:|---|---:|---:|---:|
| Light | ~40M | DDIM | 50 | 2.32 s | 0.074 s |
| Light | ~40M | DDIM | 100 | 3.81 s | 0.122 s |
| Light | ~40M | DDIM | 150 | 5.32 s | 0.170 s |
| Light | ~40M | DDPM | 1000 | 29.74 s | 0.952 s |
| Medium | ~160M | DDIM | 50 | 5.51 s | 0.176 s |
| Medium | ~160M | DDIM | 100 | 10.19 s | 0.326 s |
| Medium | ~160M | DDIM | 150 | 14.88 s | 0.476 s |
| Medium | ~160M | DDPM | 1000 | 93.99 s | 3.008 s |
| Strong | ~450M | DDIM | 50 | 4.54 s | 0.145 s |
| Strong | ~450M | DDIM | 100 | 8.59 s | 0.275 s |
| Strong | ~450M | DDIM | 150 | 12.66 s | 0.405 s |
| Strong | ~450M | DDPM | 1000 | 81.44 s | 2.606 s |
| LDM text2image | ~872M | DDIM | 50 | 5.85 s | 0.187 s |
| LDM text2image | ~872M | DDIM | 100 | 10.95 s | 0.351 s |
| LDM text2image | ~872M | DDIM | 150 | 16.07 s | 0.514 s |
| LDM text2image | ~872M | DDPM | 1000 | 103.06 s | 3.298 s |

### Best metric summary

| Model | Best KID ↓ | Best IS ↑ | Best CLIP ↑ |
|---|---:|---:|---:|
| Light | 0.0138 (DDPM, 1000 steps, CFG 3) | 10.01 (DDIM, 150 steps, CFG 5) | 21.78 (DDIM, 150 steps, CFG 7.5) |
| Medium | 0.0115 (DDIM, 50 steps, CFG 5) | 10.93 (DDIM, 100 steps, CFG 7.5) | 23.33 (DDIM, 50 steps, CFG 7.5) |
| Strong | 0.0061 (DDIM, 50 steps, CFG 3) | 13.37 (DDPM, 1000 steps, CFG 7.5) | 25.25 (DDPM, 1000 steps, CFG 7.5) |
| LDM text2image | 0.0091 (DDIM, 150 steps, CFG 3) | 14.75 (DDPM, 1000 steps, CFG 7.5) | 25.80 (DDPM, 1000 steps, CFG 7.5) |

### Detailed results: Light (~40M parameters)

| Sampler | Steps | CFG | FID ↓ | KID ↓ | IS ↑ | CLIP ↑ | Avg time / 32 imgs | Approx time / img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DDIM | 50 | 1 | 154.40 | 0.0379 ± 0.0050 | 7.33 ± 0.95 | 18.93 | 2.32 s | 0.074 s |
| DDIM | 50 | 3 | **133.21** | 0.0148 ± 0.0021 | 9.32 ± 1.34 | 21.36 | 2.32 s | 0.074 s |
| DDIM | 50 | 5 | 135.99 | 0.0156 ± 0.0023 | 9.48 ± 1.21 | 21.53 | 2.32 s | 0.074 s |
| DDIM | 50 | 7.5 | 136.01 | 0.0167 ± 0.0021 | 9.69 ± 1.03 | 21.55 | 2.32 s | 0.074 s |
| DDIM | 100 | 1 | 154.20 | 0.0392 ± 0.0046 | 7.15 ± 0.64 | 19.10 | 3.81 s | 0.122 s |
| DDIM | 100 | 3 | 134.80 | 0.0147 ± 0.0024 | 9.14 ± 0.67 | 21.04 | 3.81 s | 0.122 s |
| DDIM | 100 | 5 | 135.13 | 0.0154 ± 0.0024 | 8.99 ± 0.69 | 21.53 | 3.81 s | 0.122 s |
| DDIM | 100 | 7.5 | 139.33 | 0.0169 ± 0.0022 | 9.67 ± 1.07 | 21.60 | 3.81 s | 0.122 s |
| DDIM | 150 | 1 | 154.62 | 0.0379 ± 0.0043 | 7.39 ± 0.80 | 18.90 | 5.32 s | 0.170 s |
| DDIM | 150 | 3 | 134.55 | 0.0149 ± 0.0021 | 9.52 ± 0.94 | 21.23 | 5.32 s | 0.170 s |
| DDIM | 150 | 5 | 134.54 | 0.0143 ± 0.0019 | 10.01 ± 1.25 | 21.47 | 5.32 s | 0.170 s |
| DDIM | 150 | 7.5 | 137.29 | 0.0170 ± 0.0022 | 9.65 ± 0.95 | 21.78 | 5.32 s | 0.170 s |
| DDPM | 1000 | 1 | 153.62 | 0.0315 ± 0.0040 | 7.49 ± 0.84 | 19.01 | 29.74 s | 0.952 s |
| DDPM | 1000 | 3 | 134.41 | 0.0138 ± 0.0019 | 9.34 ± 0.98 | 21.15 | 29.74 s | 0.952 s |
| DDPM | 1000 | 5 | 135.27 | 0.0151 ± 0.0021 | 8.99 ± 0.96 | 21.41 | 29.74 s | 0.952 s |
| DDPM | 1000 | 7.5 | 142.39 | 0.0193 ± 0.0019 | 9.01 ± 1.25 | 21.32 | 29.74 s | 0.952 s |

### Detailed results: Medium (~160M parameters)

| Sampler | Steps | CFG | FID ↓ | KID ↓ | IS ↑ | CLIP ↑ | Avg time / 32 imgs | Approx time / img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DDIM | 50 | 1 | 146.18 | 0.0375 ± 0.0051 | 7.79 ± 1.14 | 19.35 | ~3.08 s | ~0.099 s |
| DDIM | 50 | 3 | 120.58 | 0.0123 ± 0.0018 | 10.48 ± 1.23 | 22.36 | ~3.43 s | ~0.110 s |
| DDIM | 50 | 5 | 118.83 | 0.0115 ± 0.0017 | 10.51 ± 0.58 | 22.94 | ~3.43 s | ~0.110 s |
| DDIM | 50 | 7.5 | 118.01 | 0.0124 ± 0.0019 | 10.76 ± 1.10 | 23.33 | ~3.43 s | ~0.110 s |
| DDIM | 100 | 1 | 145.97 | 0.0366 ± 0.0048 | 7.26 ± 0.61 | 19.41 | ~5.48 s | ~0.175 s |
| DDIM | 100 | 3 | 119.36 | 0.0120 ± 0.0018 | 10.23 ± 1.02 | 22.33 | ~6.20 s | ~0.198 s |
| DDIM | 100 | 5 | **117.35** | 0.0120 ± 0.0021 | 10.33 ± 1.26 | 23.04 | ~6.20 s | ~0.198 s |
| DDIM | 100 | 7.5 | 118.53 | 0.0130 ± 0.0020 | 10.93 ± 1.07 | 23.18 | ~6.20 s | ~0.198 s |
| DDPM | 1000 | 1 | 147.01 | 0.0323 ± 0.0043 | 7.70 ± 0.66 | 19.48 | ~48.48 s | ~1.551 s |
| DDPM | 1000 | 3 | 120.35 | 0.0117 ± 0.0015 | 10.55 ± 0.91 | 22.34 | ~55.57 s | ~1.778 s |
| DDPM | 1000 | 5 | 119.38 | 0.0118 ± 0.0021 | 10.76 ± 1.94 | 22.99 | ~55.57 s | ~1.778 s |
| DDPM | 1000 | 7.5 | 118.99 | 0.0128 ± 0.0017 | 10.45 ± 0.87 | 23.24 | ~55.57 s | ~1.778 s |

### Detailed results: Strong (~450M parameters)

| Sampler | Steps | CFG | FID ↓ | KID ↓ | IS ↑ | CLIP ↑ | Avg time / 32 imgs | Approx time / img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DDIM | 50 | 1 | 129.41 | 0.0249 ± 0.0046 | 8.44 ± 0.88 | 19.89 | 4.54 s | 0.145 s |
| DDIM | 50 | 3 | 108.18 | 0.0061 ± 0.0012 | 12.34 ± 0.93 | 23.56 | 4.54 s | 0.145 s |
| DDIM | 50 | 5 | 106.49 | 0.0086 ± 0.0017 | 12.42 ± 1.51 | 24.57 | 4.54 s | 0.145 s |
| DDIM | 50 | 7.5 | 108.41 | 0.0095 ± 0.0018 | 11.75 ± 0.80 | 24.71 | 4.54 s | 0.145 s |
| DDIM | 100 | 1 | 128.20 | 0.0246 ± 0.0047 | 8.71 ± 0.66 | 20.36 | 8.59 s | 0.275 s |
| DDIM | 100 | 3 | 107.76 | 0.0067 ± 0.0013 | 12.39 ± 1.20 | 23.47 | 8.59 s | 0.275 s |
| DDIM | 100 | 5 | **105.85** | 0.0076 ± 0.0019 | 12.92 ± 0.75 | 24.45 | 8.59 s | 0.275 s |
| DDIM | 100 | 7.5 | 109.54 | 0.0093 ± 0.0021 | 12.98 ± 1.09 | 24.71 | 8.59 s | 0.275 s |
| DDIM | 150 | 1 | 130.50 | 0.0257 ± 0.0054 | 8.26 ± 0.83 | 20.00 | 12.66 s | 0.405 s |
| DDIM | 150 | 3 | 108.33 | 0.0070 ± 0.0014 | 11.61 ± 1.40 | 23.68 | 12.66 s | 0.405 s |
| DDIM | 150 | 5 | 106.72 | 0.0082 ± 0.0019 | 12.41 ± 1.25 | 24.68 | 12.66 s | 0.405 s |
| DDIM | 150 | 7.5 | 109.19 | 0.0101 ± 0.0023 | 12.86 ± 0.85 | 24.70 | 12.66 s | 0.405 s |
| DDPM | 1000 | 1 | 126.15 | 0.0178 ± 0.0039 | 9.38 ± 1.13 | 20.17 | 81.44 s | 2.606 s |
| DDPM | 1000 | 3 | 106.73 | 0.0066 ± 0.0017 | 12.93 ± 1.15 | 24.09 | 81.44 s | 2.606 s |
| DDPM | 1000 | 5 | 106.97 | 0.0083 ± 0.0021 | 12.84 ± 1.08 | 24.69 | 81.44 s | 2.606 s |
| DDPM | 1000 | 7.5 | 108.33 | 0.0095 ± 0.0019 | 13.37 ± 1.03 | 25.25 | 81.44 s | 2.606 s |

### Detailed results: LDM text2image (~872M parameters)

| Sampler | Steps | CFG | FID ↓ | KID ↓ | IS ↑ | CLIP ↑ | Avg time / 32 imgs | Approx time / img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DDIM | 50 | 1 | 130.53 | 0.0212 ± 0.0031 | 9.54 ± 0.51 | 20.77 | 5.85 s | 0.187 s |
| DDIM | 50 | 3 | 112.84 | 0.0108 ± 0.0014 | 12.05 ± 0.93 | 24.65 | 5.85 s | 0.187 s |
| DDIM | 50 | 5 | 113.79 | 0.0115 ± 0.0017 | 12.76 ± 1.36 | 25.21 | 5.85 s | 0.187 s |
| DDIM | 50 | 7.5 | 114.56 | 0.0127 ± 0.0018 | 12.91 ± 1.74 | 25.45 | 5.85 s | 0.187 s |
| DDIM | 100 | 1 | 131.66 | 0.0229 ± 0.0044 | 9.50 ± 0.99 | 20.66 | 10.95 s | 0.351 s |
| DDIM | 100 | 3 | 113.79 | 0.0111 ± 0.0016 | 12.16 ± 0.99 | 24.63 | 10.95 s | 0.351 s |
| DDIM | 100 | 5 | 113.94 | 0.0117 ± 0.0017 | 12.69 ± 1.38 | 25.19 | 10.95 s | 0.351 s |
| DDIM | 100 | 7.5 | 114.49 | 0.0127 ± 0.0018 | 12.88 ± 1.36 | 25.52 | 10.95 s | 0.351 s |
| DDIM | 150 | 1 | 132.29 | 0.0224 ± 0.0041 | 9.18 ± 0.69 | 20.77 | 16.07 s | 0.514 s |
| DDIM | 150 | 3 | 109.28 | 0.0091 ± 0.0015 | 12.08 ± 1.17 | 24.80 | 16.07 s | 0.514 s |
| DDIM | 150 | 5 | 109.96 | 0.0095 ± 0.0015 | 12.44 ± 1.15 | 25.30 | 16.07 s | 0.514 s |
| DDIM | 150 | 7.5 | 111.04 | 0.0104 ± 0.0017 | 12.93 ± 1.16 | 25.61 | 16.07 s | 0.514 s |
| DDPM | 1000 | 1 | 136.40 | 0.0238 ± 0.0031 | 9.03 ± 1.13 | 21.04 | 103.06 s | 3.298 s |
| DDPM | 1000 | 3 | **108.54** | 0.0096 ± 0.0014 | 13.11 ± 1.42 | 25.42 | 103.06 s | 3.298 s |
| DDPM | 1000 | 5 | 110.75 | 0.0101 ± 0.0019 | 13.80 ± 0.85 | 25.74 | 103.06 s | 3.298 s |
| DDPM | 1000 | 7.5 | 113.20 | 0.0111 ± 0.0020 | 14.75 ± 1.52 | 25.80 | 103.06 s | 3.298 s |

---

## Dataset

Expected COCO structure:

```text
coco2017/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

Latents are cached once before diffusion training. Each image has one latent and all available COCO captions.

Example latent cache paths:

```text
outputs/latents/coco_train2017_vae8_scaled_allcaptions
outputs/latents/coco_val2017_vae8_scaled_allcaptions
```

---

## Main Files

```text
configs/
├── model/
│   ├── vae_small.yaml
│   ├── ldm_unet_light.yaml
│   ├── ldm_unet_medium.yaml
│   └── ldm_unet_strong.yaml
└── experiments/
    ├── vae_coco_256.yaml
    ├── ldm_coco_256_vpred.yaml
    ├── ldm_coco_256_vpred_strong_ddp.yaml
    └── sample_coco_256.yaml

scripts/
├── train_vae.py
├── estimate_vae_scaling_factor.py
├── encode_latents.py
├── train_ldm.py
├── train_ldm_multigpu.py
└── sample_text2image.py

src/
├── network/autoencoder/
├── network/diffusion/
├── network/conditioning/
├── diffusion/
├── losses/
└── data/
```

---

## Training

### 1. Train the VAE

```bash
python scripts/train_vae.py \
  --config configs/experiments/vae_coco_256.yaml
```

### 2. Estimate VAE Scaling Factor

```bash
python scripts/estimate_vae_scaling_factor.py \
  --config outputs/vae/vae_coco_256_small/config_used.yaml \
  --checkpoint outputs/vae/vae_coco_256_small/checkpoints/best.pt \
  --split train2017 \
  --num-images 10000 \
  --batch-size 32 \
  --output-json outputs/vae/vae_coco_256_small/scaling_factor_10k.json
```

The estimated scaling factor is used in diffusion training to normalize VAE latents to roughly unit variance, so the diffusion noise schedule works stably instead of seeing latents with arbitrary scale.

### 3. Encode COCO Latents

```bash
python scripts/encode_latents.py \
  --config outputs/vae/vae_coco_256_small/config_used.yaml \
  --checkpoint outputs/vae/vae_coco_256_small/checkpoints/best.pt \
  --split train2017 \
  --num-images -1 \
  --batch-size 32 \
  --shard-size 10000 \
  --scaling-factor 1.0320695526096337 \
  --output-dir outputs/latents/coco_train2017_vae8_scaled_allcaptions \
  --overwrite
```

### 4. Train the LDM

Single GPU:

```bash
python scripts/train_ldm.py \
  --config configs/experiments/ldm_coco_256_vpred.yaml
```

Multi-GPU:

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=3 \
  scripts/train_ldm_multigpu.py \
  --config configs/experiments/ldm_coco_256_vpred_strong_ddp.yaml
```

Fine-tuning from an existing LDM checkpoint:

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=3 \
  scripts/train_ldm_multigpu.py \
  --config configs/experiments/ldm_coco_256_vpred_strong_ft.yaml \
  --finetune-from outputs/ldm/YOUR_OLD_RUN/checkpoints/last.pt
```

Fine-tuning loads only model weights and starts a new optimizer state.

---

## Training Features

The project supports:

- v-prediction
- cosine noise schedule
- classifier-free guidance dropout
- DDPM and DDIM samplers
- multi-GPU DDP training
- warmup/cosine learning-rate scheduling
- optional Min-SNR loss weighting
- timestep-bucket validation loss

Example Min-SNR configuration:

```yaml
diffusion:
  prediction_type: v
  loss_type: mse
  snr_weighting: min_snr
  snr_gamma: 5.0
  normalize_snr_weights: false
```

To disable SNR weighting:

```yaml
diffusion:
  snr_weighting: none
  snr_gamma: null
```

Min-SNR loss weighting downweights very high-SNR timesteps and can help balance learning across the diffusion trajectory.

---

## Model Size Notes

```text
Light:          ~40M parameters
Medium:         ~160M parameters
Strong:         ~450M parameters
LDM text2image: ~872M parameters
Original Stable Diffusion 2 U-Net: ~860M parameters
```

Measured memory notes:

```text
Released custom LDM fp16 peak VRAM: 3244 MB / 3.168 GB
Strong custom LDM comparison memory: ~2.23 GB
LDM text2image total memory: ~6.15 GB
```

---

## Disclaimer

This repository is intended as a custom latent diffusion research codebase for COCO-scale text-to-image experiments. It is not a production Stable Diffusion replacement. The main goal is to understand and improve each component of a latent diffusion system end to end.
