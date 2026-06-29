from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image, make_grid

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.diffusion.gaussian_diffusion import GaussianDiffusion
from src.diffusion.samplers import DDPMSampler, DDIMSampler
from src.models.autoencoder.vae import AutoencoderKL
from src.models.conditioning.clip_text import FrozenCLIPTextEncoder
from src.models.conditioning.null_conditioning import ClassifierFreeGuidanceConditioner
from src.models.diffusion.unet import build_latent_diffusion_unet_from_config
from src.utils.config import load_config, resolve_path_key, save_yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/sample_coco_256.yaml",
        help="Sampling config path.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt override.",
    )

    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional text file with one prompt per line.",
    )

    parser.add_argument(
        "--ldm-checkpoint",
        type=str,
        default=None,
        help="Override LDM checkpoint path.",
    )

    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default=None,
        help="Override VAE checkpoint path.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed.",
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override CFG guidance scale.",
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override number of sampling steps.",
    )

    parser.add_argument(
        "--num-images-per-prompt",
        type=int,
        default=None,
        help="Override images per prompt.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_state_dict_flexible(module: torch.nn.Module, checkpoint_path: str | Path, key: str = "model") -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and key in checkpoint:
        state_dict = checkpoint[key]
    else:
        state_dict = checkpoint

    module.load_state_dict(state_dict, strict=True)


def build_vae_from_config(vae_cfg: dict) -> AutoencoderKL:
    m = vae_cfg["model"]

    model = AutoencoderKL(
        in_channels=int(m["in_channels"]),
        out_channels=int(m["out_channels"]),
        latent_channels=int(m["latent_channels"]),
        base_channels=int(m["base_channels"]),
        channel_multipliers=tuple(m["channel_multipliers"]),
        num_res_blocks=int(m["num_res_blocks"]),
        dropout=float(m.get("dropout", 0.0)),
        use_attention=bool(m.get("use_attention", True)),
        attention_heads=int(m.get("attention_heads", 1)),
        scaling_factor=float(m.get("scaling_factor", 1.0)),
    )

    return model


def build_conditioner(cfg: dict) -> ClassifierFreeGuidanceConditioner:
    text_cfg = cfg["text_encoder"]
    cond_cfg = cfg["conditioning"]

    text_encoder = FrozenCLIPTextEncoder(
        model_name=str(text_cfg.get("model_name", "openai/clip-vit-large-patch14")),
        max_length=int(text_cfg.get("max_length", 77)),
        freeze=bool(text_cfg.get("freeze", True)),
        use_last_hidden_state=bool(text_cfg.get("use_last_hidden_state", True)),
    )

    conditioner = ClassifierFreeGuidanceConditioner(
        text_encoder=text_encoder,
        cond_drop_prob=0.0,  # no training-time dropout in inference
        empty_text=str(cond_cfg.get("empty_text", "")),
    )

    return conditioner


def build_diffusion(cfg: dict) -> GaussianDiffusion:
    d = cfg["diffusion"]

    diffusion = GaussianDiffusion(
        schedule_type=str(d.get("schedule_type", "cosine")),
        num_timesteps=int(d.get("num_timesteps", 1000)),
        prediction_type=str(d.get("prediction_type", "v")),
        loss_type=str(d.get("loss_type", "mse")),
        beta_start=float(d.get("beta_start", 1e-4)),
        beta_end=float(d.get("beta_end", 2e-2)),
        cosine_s=float(d.get("cosine_s", 0.008)),
        max_beta=float(d.get("max_beta", 0.999)),
    )

    return diffusion


def get_autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False)

    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=True)

    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16, enabled=True)

    return torch.autocast("cuda", enabled=False)


def gather_prompts(cfg: dict, args) -> list[str]:
    prompts = []

    if args.prompt is not None:
        prompts = [args.prompt]

    elif args.prompts_file is not None:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    else:
        prompts = list(cfg.get("prompts", []))

    if len(prompts) == 0:
        raise ValueError("No prompts provided. Use config prompts, --prompt, or --prompts-file.")

    return prompts


def repeat_prompts(prompts: list[str], num_images_per_prompt: int) -> tuple[list[str], list[int]]:
    repeated_prompts = []
    prompt_indices = []

    for i, prompt in enumerate(prompts):
        for _ in range(num_images_per_prompt):
            repeated_prompts.append(prompt)
            prompt_indices.append(i)

    return repeated_prompts, prompt_indices


def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    keep = []

    for ch in text:
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_"}:
            keep.append("_")

    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")

    out = out.strip("_")

    if not out:
        out = "sample"

    return out[:max_len]


@torch.no_grad()
def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    """
    Decode scaled diffusion latents back into image space.

    During training, latents were scaled before diffusion:
        z_scaled = z * scaling_factor

    So here we unscale before decoding.
    """
    scaling_factor = float(getattr(vae, "scaling_factor", 1.0))

    # Prefer model's own decode API if it supports unscale=True.
    try:
        images = vae.decode(latents, unscale=True)
        return images
    except TypeError:
        pass

    z = latents / scaling_factor
    images = vae.decode(z)
    return images


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.ldm_checkpoint is not None:
        cfg["ldm"]["checkpoint"] = args.ldm_checkpoint

    if args.vae_checkpoint is not None:
        cfg["vae"]["checkpoint"] = args.vae_checkpoint

    if args.output_dir is not None:
        cfg["generation"]["output_dir"] = args.output_dir

    if args.seed is not None:
        cfg["generation"]["seed"] = args.seed

    if args.guidance_scale is not None:
        cfg["sampler"]["guidance_scale"] = args.guidance_scale

    if args.num_steps is not None:
        cfg["sampler"]["num_steps"] = args.num_steps

    if args.num_images_per_prompt is not None:
        cfg["generation"]["num_images_per_prompt"] = args.num_images_per_prompt

    prompts = gather_prompts(cfg, args)

    seed = int(cfg["generation"].get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(cfg["generation"].get("precision", "bf16"))

    output_dir = Path(cfg["generation"]["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = resolve_path_key(cfg, "project.root") / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, output_dir / "config_used.yaml")

    vae_cfg_path = Path(cfg["vae"]["config"]).expanduser()
    if not vae_cfg_path.is_absolute():
        vae_cfg_path = resolve_path_key(cfg, "project.root") / vae_cfg_path
    vae_cfg = load_config(str(vae_cfg_path))

    vae_ckpt = Path(cfg["vae"]["checkpoint"]).expanduser()
    if not vae_ckpt.is_absolute():
        vae_ckpt = resolve_path_key(cfg, "project.root") / vae_ckpt

    ldm_ckpt = Path(cfg["ldm"]["checkpoint"]).expanduser()
    if not ldm_ckpt.is_absolute():
        ldm_ckpt = resolve_path_key(cfg, "project.root") / ldm_ckpt

    num_images_per_prompt = int(cfg["generation"].get("num_images_per_prompt", 1))
    batch_size = int(cfg["generation"].get("batch_size", 4))

    sampler_type = str(cfg["sampler"].get("type", "ddim")).lower()
    num_steps = int(cfg["sampler"].get("num_steps", 50))
    eta = float(cfg["sampler"].get("eta", 0.0))
    guidance_scale = float(cfg["sampler"].get("guidance_scale", 5.0))
    clip_denoised = bool(cfg["sampler"].get("clip_denoised", False))

    # Build models
    vae = build_vae_from_config(vae_cfg)
    load_state_dict_flexible(vae, vae_ckpt, key="model")
    vae.to(device)
    vae.eval()

    unet = build_latent_diffusion_unet_from_config(cfg)
    load_state_dict_flexible(unet, ldm_ckpt, key="model")
    unet.to(device)
    unet.eval()

    conditioner = build_conditioner(cfg)
    conditioner.to(device)
    conditioner.eval()

    diffusion = build_diffusion(cfg).to(device)

    if sampler_type == "ddim":
        sampler = DDIMSampler(diffusion)
    elif sampler_type == "ddpm":
        sampler = DDPMSampler(diffusion)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

    repeated_prompts, prompt_indices = repeat_prompts(prompts, num_images_per_prompt)

    print("=============================================")
    print("Text-to-image sampling")
    print("Config:", args.config)
    print("VAE checkpoint:", vae_ckpt)
    print("LDM checkpoint:", ldm_ckpt)
    print("Output dir:", output_dir)
    print("Device:", device)
    print("Precision:", precision)
    print("Sampler:", sampler_type)
    print("Steps:", num_steps if sampler_type == "ddim" else diffusion.num_timesteps)
    print("Guidance scale:", guidance_scale)
    print("Seed:", seed)
    print("Prompts:", len(prompts))
    print("Images per prompt:", num_images_per_prompt)
    print("Total images:", len(repeated_prompts))
    print("=============================================")

    latent_channels = int(cfg["model"]["in_channels"])
    latent_size = int(cfg["model"].get("latent_size", 32))

    all_images = []
    saved_paths = []

    for batch_start in range(0, len(repeated_prompts), batch_size):
        batch_prompts = repeated_prompts[batch_start: batch_start + batch_size]
        batch_prompt_indices = prompt_indices[batch_start: batch_start + batch_size]

        with get_autocast_context(device, precision):
            cond = conditioner.encode_cond_uncond(
                batch_prompts,
                device=device,
            )

            if sampler_type == "ddim":
                sample_out = sampler.sample(
                    model=unet,
                    shape=(len(batch_prompts), latent_channels, latent_size, latent_size),
                    device=device,
                    context=cond["cond_context"],
                    attention_mask=cond["cond_attention_mask"],
                    uncond_context=cond["uncond_context"],
                    uncond_attention_mask=cond["uncond_attention_mask"],
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    eta=eta,
                    clip_denoised=clip_denoised,
                    return_trajectory=False,
                    progress=True,
                )
            else:
                sample_out = sampler.sample(
                    model=unet,
                    shape=(len(batch_prompts), latent_channels, latent_size, latent_size),
                    device=device,
                    context=cond["cond_context"],
                    attention_mask=cond["cond_attention_mask"],
                    uncond_context=cond["uncond_context"],
                    uncond_attention_mask=cond["uncond_attention_mask"],
                    guidance_scale=guidance_scale,
                    clip_denoised=clip_denoised,
                    return_trajectory=False,
                    progress=True,
                )

            latents = sample_out.latents
            images = decode_latents(vae, latents)

        images = ((images + 1.0) / 2.0).clamp(0.0, 1.0)
        all_images.append(images.cpu())

        # Save individual images
        for i, image in enumerate(images):
            prompt = batch_prompts[i]
            prompt_idx = batch_prompt_indices[i]
            file_stub = sanitize_filename(prompt)
            img_count_for_prompt = sum(1 for pidx in prompt_indices[: batch_start + i + 1] if pidx == prompt_idx)

            out_path = output_dir / f"{prompt_idx:03d}_{img_count_for_prompt:02d}_{file_stub}.png"
            save_image(image.cpu(), out_path)
            saved_paths.append(out_path)

    all_images = torch.cat(all_images, dim=0)

    grid_nrow = num_images_per_prompt
    grid = make_grid(all_images, nrow=grid_nrow)
    save_image(grid, output_dir / "grid.png")

    with open(output_dir / "prompts.txt", "w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts):
            f.write(f"[{idx:03d}] {prompt}\n")

    print("=============================================")
    print("Sampling finished")
    print("Saved images:", len(saved_paths))
    print("Grid:", output_dir / "grid.png")
    print("Output dir:", output_dir)
    print("=============================================")


if __name__ == "__main__":
    main()