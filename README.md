# Latent Diffusion

**Model checkpoints:**  
- VAE: `https://huggingface.co/anilegin/lightweight-diffusion-vae`  
- Latent Diffusion Model: `-`

This repository implements a custom latent diffusion text-to-image pipeline trained on COCO 2017 captions. It includes a custom VAE, a CLIP-conditioned latent diffusion U-Net, multi-GPU training and DDIM/DDPM sampling.

---

## Quick Start: Text-to-Image Sampling

Use `scripts/sample_text2image.py` to generate images from a trained latent diffusion checkpoint. For early custom-trained models, lower classifier-free guidance can work better.

```bash
python scripts/sample_text2image.py \
  --config configs/experiments/sample_coco_256.yaml \
  --ldm-checkpoint outputs/ldm/ldm_coco_256_vae8_strong_vpred_ddp/checkpoints/best.pt \
  --vae-checkpoint outputs/vae/vae_coco_256_small/checkpoints/best.pt \
  --prompt "a dog running through grass" \
  --num-steps 100 \
  --guidance-scale 2.0 \
  --seed 42
```

Recommended first sampling settings:

```yaml
sampler:
  type: ddim
  num_steps: 100
  eta: 0.0
  guidance_scale: 2.0
  clip_denoised: false
```

Try CFG values:

```text
1.0, 1.5, 2.0, 3.0
```

Very high CFG values such as `5.0` or `7.5` can make weak or undertrained custom models worse.

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

The VAE is an `AutoencoderKL` trained from scratch on COCO 2017 images.

Its role is to compress images:

```text
image [3, 256, 256] -> latent [8, 32, 32]
```

and reconstruct images:

```text
latent [8, 32, 32] -> image [3, 256, 256]
```

The trained VAE uses 8 latent channels. The estimated latent scaling factor is:

```text
scaling_factor = 1.0320695526096337
```

During diffusion training, encoded latents are multiplied by this scaling factor. During sampling, generated latents must be divided by the same scaling factor before VAE decoding.

---

## VAE Results

Evaluation was performed on the same subset setting: 5,000 COCO images at 256×256.

| Model | rFID ↓ | PSNR ↑ | SSIM ↑ | LPIPS / PSIM ↓ | Notes |
|---|---:|---:|---:|---:|---|
| Custom COCO VAE | 8.91 | 30.1 ± 3.5 | 0.88 ± 0.07 | LPIPS 0.13 ± 0.04 | Trained from scratch for this project; 8 latent channels |
| SD VAE original | 4.99 | 23.4 ± 3.8 | 0.69 ± 0.14 | PSIM 1.01 ± 0.28 | Original KL-f4 VAE used in Stable Diffusion |

Reference for the Stable Diffusion VAE:  
`https://huggingface.co/stabilityai/sd-vae-ft-mse-original`

The custom VAE reconstructs images with much higher PSNR and SSIM on this setup, while the SD VAE original has better rFID.

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

---

## Diffusion Results


| Model | Parameters | Training setup | Loss / metric | Sampling result | Notes |
|---|---:|---|---|---|---|
| Light LDM | ~40M | `[ADD_DETAILS]` | `[ADD_RESULTS]` | `[ADD_RESULTS]` | `[ADD_NOTES]` |
| Medium LDM | ~165M | `[ADD_DETAILS]` | `[ADD_RESULTS]` | `[ADD_RESULTS]` | `[ADD_NOTES]` |
| Strong LDM | ~450M | `[ADD_DETAILS]` | `[ADD_RESULTS]` | `[ADD_RESULTS]` | `[ADD_NOTES]` |

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
│   ├── ldm_unet_strong.yaml
│   └── ldm_unet_xl_480m.yaml
└── experiments/
    ├── vae_coco_256.yaml
    ├── ldm_coco_256_vpred.yaml
    ├── ldm_coco_256_vpred_strong_ddp.yaml
    └── sample_coco_256.yaml

scripts/
├── train_vae.py
├── encode_latents.py
├── train_ldm.py
├── train_ldm_multigpu.py
└── sample_text2image.py

src/
├── models/autoencoder/
├── models/diffusion/
├── models/conditioning/
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

Min-SNR allows model to treat differently for each timestep, weighting less for the higher and more for the later steps.

---

## Notes on Model Sizes

Several U-Net sizes were tested:

```text
light:   ~40M parameters
medium:  ~160M parameters
strong:  ~450M parameters
Original Stable Diffusion 2:      ~860M parameters
```

---

## Disclaimer

This repository is intended as a custom latent diffusion research codebase for COCO-scale text-to-image experiments. It is not a production Stable Diffusion replacement. The main goal is to understand and improve each component of a latent diffusion system end to end.
