from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.diffusion.gaussian_diffusion import GaussianDiffusion
from src.diffusion.samplers import DDPMSampler, DDIMSampler
from src.network.conditioning.clip_text import FrozenCLIPTextEncoder
from src.network.diffusion.unet import build_latent_diffusion_unet_from_config
from src.utils.config import load_config
from src.utils.timer import Timer
from src.utils.gpu_monitor import GPUMonitor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--monitor-gpu", action="store_true", help="Enable per-section GPU memory/utilization monitoring.")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25, help="Seconds between nvidia-smi utilization samples.")
    parser.add_argument("--no-nvidia-smi", action="store_true", help="Disable nvidia-smi utilization sampling and keep only PyTorch CUDA memory stats.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dtype(name: str):
    name = name.lower()
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown precision={name}")


def sanitize_float(x: float) -> str:
    return str(x).replace(".", "p")


def safe_torch_load(path: str | Path, map_location="cpu"):
    """Load tensors/checkpoints safely on newer PyTorch, with fallback for older files."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def save_image_grid(images: list[Image.Image], path: str | Path, nrow: int | None = None, padding: int = 2):
    if not images:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if nrow is None:
        nrow = int(math.ceil(math.sqrt(len(images))))
    ncol = int(math.ceil(len(images) / nrow))

    widths, heights = zip(*(img.size for img in images))
    cell_w, cell_h = max(widths), max(heights)
    grid_w = nrow * cell_w + padding * (nrow - 1)
    grid_h = ncol * cell_h + padding * (ncol - 1)

    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        x = col * (cell_w + padding)
        y = row * (cell_h + padding)
        grid.paste(img.convert("RGB"), (x, y))

    grid.save(path)


def model_forward_autocast(device: torch.device, dtype: torch.dtype):

    enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)

    return torch.autocast(device_type="cpu", enabled=False)


def load_first_val_captions(latent_dir: str | Path, max_images: int) -> list[str]:
    """
    max_images > 0: return the first max_images captions.
    max_images == 0: return all captions found in the latent shards.
    """
    latent_dir = Path(latent_dir)
    shard_paths = sorted(latent_dir.glob("*.pt"))

    if not shard_paths:
        raise RuntimeError(f"No .pt latent shards found in {latent_dir}")

    if max_images < 0:
        raise ValueError(f"evaluation.max_images must be >= 0, got {max_images}")

    captions: list[str] = []
    limit = None if max_images == 0 else max_images

    for shard_path in shard_paths:
        payload = safe_torch_load(shard_path, map_location="cpu")
        shard_caps = payload["captions"]

        for cap in shard_caps:
            if isinstance(cap, list):
                cap = cap[0]

            captions.append(str(cap))

            if limit is not None and len(captions) >= limit:
                return captions

    return captions


def load_vae_from_config(config_path: str | Path, checkpoint_path: str | Path, device, dtype):
    cfg = load_config(config_path)
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)

    try:
        from src.network.autoencoder.vae import AutoencoderKL
        vae = AutoencoderKL(**model_cfg)
    except Exception as exc:
        raise RuntimeError(
            "Could not build custom VAE. If your VAE builder has a different name, "
            "edit load_vae_from_config() in this script."
        ) from exc

    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint

    vae.load_state_dict(state_dict, strict=True)

    # Minimal important change:
    # Keep VAE weights in fp32. Use autocast during decode instead.
    vae.to(device=device)
    vae.eval()

    return vae


def load_ldm_model(config_path: str | Path, checkpoint_path: str | Path, device, dtype):
    cfg = load_config(config_path)
    model = build_latent_diffusion_unet_from_config(cfg)

    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)

    # Minimal important change:
    # Keep UNet weights in fp32. Use autocast during forward instead.
    model.to(device=device)
    model.eval()

    return model, cfg


def build_diffusion(cfg: dict) -> GaussianDiffusion:
    d = cfg["diffusion"]

    return GaussianDiffusion(
        schedule_type=str(d.get("schedule_type", "cosine")),
        num_timesteps=int(d.get("num_timesteps", 1000)),
        prediction_type=str(d.get("prediction_type", "v")),
        loss_type=str(d.get("loss_type", "mse")),
        beta_start=float(d.get("beta_start", 1e-4)),
        beta_end=float(d.get("beta_end", 2e-2)),
        cosine_s=float(d.get("cosine_s", 0.008)),
        max_beta=float(d.get("max_beta", 0.999)),
    )


def build_text_encoder(cfg: dict, device, dtype):
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*", category=FutureWarning)

    text_cfg = cfg["text_encoder"]

    text_encoder = FrozenCLIPTextEncoder(
        model_name=str(text_cfg.get("model_name", "openai/clip-vit-large-patch14")),
        max_length=int(text_cfg.get("max_length", 77)),
        freeze=True,
        use_last_hidden_state=bool(text_cfg.get("use_last_hidden_state", True)),

        # Keeps usage local.
        local_files_only=bool(text_cfg.get("local_files_only", True)),
    )

    text_encoder.to(device=device)
    text_encoder.eval()

    return text_encoder


@torch.no_grad()
def encode_contexts(
    text_encoder,
    prompts: list[str],
    empty_text: str,
    device,
    dtype,
    timer: Timer,
    gpu_monitor: GPUMonitor,
):
    context_dtype = dtype if device.type == "cuda" else torch.float32

    with timer("encode_text_cond"), gpu_monitor.track("encode_text_cond"):
        cond_context = text_encoder.encode(prompts, device=device).to(dtype=context_dtype)

    with timer("encode_text_uncond"), gpu_monitor.track("encode_text_uncond"):
        uncond_context = text_encoder.encode([empty_text] * len(prompts), device=device).to(dtype=context_dtype)

    return cond_context, uncond_context


@torch.no_grad()
def model_predict_cfg(model, z_t, t, cond_context, uncond_context, guidance_scale: float):
    z_in = torch.cat([z_t, z_t], dim=0)
    t_in = torch.cat([t, t], dim=0)
    context = torch.cat([uncond_context, cond_context], dim=0)


    amp_dtype = context.dtype

    with model_forward_autocast(z_t.device, amp_dtype):
        model_out = model(z_in, t_in, context=context)

    model_out = model_out.to(dtype=z_t.dtype)

    uncond, cond = model_out.chunk(2, dim=0)
    return uncond + guidance_scale * (cond - uncond)


@torch.no_grad()
def sample_ddpm(
    model,
    diffusion,
    shape,
    cond_context,
    uncond_context,
    guidance_scale,
    device,
    dtype,
    timer: Timer,
    gpu_monitor: GPUMonitor,
):
    sampler = DDPMSampler(diffusion)

    with timer("sample_ddpm_loop"), gpu_monitor.track("sample_ddpm_loop"):
        with model_forward_autocast(device, dtype):
            sample_out = sampler.sample(
                model=model,
                shape=shape,
                device=device,
                context=cond_context,
                attention_mask=None,
                uncond_context=uncond_context,
                uncond_attention_mask=None,
                guidance_scale=guidance_scale,
                clip_denoised=False,
                return_trajectory=False,
                progress=True,
            )

    if hasattr(sample_out, "latents"):
        return sample_out.latents

    return sample_out

@torch.no_grad()
def sample_ddim(
    model,
    diffusion,
    shape,
    cond_context,
    uncond_context,
    guidance_scale,
    num_steps,
    eta,
    device,
    dtype,
    timer: Timer,
    gpu_monitor: GPUMonitor,
):

    sampler = DDIMSampler(diffusion)

    with timer("sample_ddim_loop"), gpu_monitor.track("sample_ddim_loop"):
        with model_forward_autocast(device, dtype):
            sample_out = sampler.sample(
                model=model,
                shape=shape,
                device=device,
                context=cond_context,
                attention_mask=None,
                uncond_context=uncond_context,
                uncond_attention_mask=None,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                eta=eta,
                clip_denoised=False,
                return_trajectory=False,
                progress=True,
            )

    if hasattr(sample_out, "latents"):
        return sample_out.latents

    return sample_out


@torch.no_grad()
def decode_latents(
    vae,
    latents: torch.Tensor,
    scaling_factor: float,
    dtype,
    timer: Timer,
    gpu_monitor: GPUMonitor,
):
    with timer("vae_decode"), gpu_monitor.track("vae_decode"):
        vae_param_dtype = next(vae.parameters()).dtype

        # Match your working script: prefer VAE's own unscale=True path if available.
        try:
            vae_input = latents.float()

            if latents.device.type != "cuda":
                vae_input = vae_input.to(dtype=vae_param_dtype)

            with model_forward_autocast(latents.device, dtype):
                out = vae.decode(vae_input, unscale=True)

        except TypeError:
            z = latents.float() / scaling_factor

            if latents.device.type != "cuda":
                z = z.to(dtype=vae_param_dtype)

            with model_forward_autocast(latents.device, dtype):
                out = vae.decode(z)

        if hasattr(out, "sample"):
            images = out.sample
        else:
            images = out

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")

    return [Image.fromarray(image) for image in images]


def save_prompt_file(prompts: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "prompts.json", "w", encoding="utf-8") as f:
        json.dump([{"index": i, "prompt": p} for i, p in enumerate(prompts)], f, indent=2)


def save_image_batch(
    images: list[Image.Image],
    output_dir: Path,
    start_index: int,
    timer: Timer,
    gpu_monitor: GPUMonitor,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with timer("save_images"), gpu_monitor.track("save_images"):
        for offset, image in enumerate(images):
            image.save(output_dir / f"sample_{start_index + offset:04d}.png")


def iter_batches(items: list[str], batch_size: int):
    if batch_size <= 0:
        raise ValueError(f"generation_batch_size must be > 0, got {batch_size}")

    for start in range(0, len(items), batch_size):
        yield start, items[start: start + batch_size]


def run_setting(
    setting_name,
    sampler_name,
    steps,
    guidance_scale,
    eta,
    model,
    diffusion,
    vae,
    prompts,
    latent_channels,
    latent_size,
    generation_batch_size,
    scaling_factor,
    text_encoder,
    empty_text,
    device,
    dtype,
    output_root,
    monitor_gpu: bool,
    gpu_sample_interval: float,
    use_nvidia_smi: bool,
    grid_max_images: int = 64,
):
    setting_dir = output_root / setting_name
    setting_dir.mkdir(parents=True, exist_ok=True)
    save_prompt_file(prompts, setting_dir)

    timer = Timer(sync_cuda=True)
    gpu_monitor = GPUMonitor(
        enabled=monitor_gpu,
        device=device,
        sync_cuda=True,
        sample_interval_s=gpu_sample_interval,
        use_nvidia_smi=use_nvidia_smi,
    )

    grid_images: list[Image.Image] = []
    num_batches = math.ceil(len(prompts) / generation_batch_size)

    with timer("total"), gpu_monitor.track("total"):
        for batch_id, (start_index, batch_prompts) in enumerate(
            tqdm(
                iter_batches(prompts, generation_batch_size),
                total=num_batches,
                desc=f"{setting_name} batches",
            )
        ):
            batch_size = len(batch_prompts)
            shape = (batch_size, latent_channels, latent_size, latent_size)

            cond_context, uncond_context = encode_contexts(
                text_encoder=text_encoder,
                prompts=batch_prompts,
                empty_text=empty_text,
                device=device,
                dtype=dtype,
                timer=timer,
                gpu_monitor=gpu_monitor,
            )

            if sampler_name == "ddpm":
                latents = sample_ddpm(
                    model=model,
                    diffusion=diffusion,
                    shape=shape,
                    cond_context=cond_context,
                    uncond_context=uncond_context,
                    guidance_scale=guidance_scale,
                    device=device,
                    dtype=dtype,
                    timer=timer,
                    gpu_monitor=gpu_monitor,
                )

            elif sampler_name == "ddim":
                latents = sample_ddim(
                    model=model,
                    diffusion=diffusion,
                    shape=shape,
                    cond_context=cond_context,
                    uncond_context=uncond_context,
                    guidance_scale=guidance_scale,
                    num_steps=steps,
                    eta=eta,
                    device=device,
                    dtype=dtype,
                    timer=timer,
                    gpu_monitor=gpu_monitor,
                )

            else:
                raise ValueError(f"Unknown sampler_name={sampler_name}")

            images = decode_latents(
                vae=vae,
                latents=latents,
                scaling_factor=scaling_factor,
                dtype=dtype,
                timer=timer,
                gpu_monitor=gpu_monitor,
            )

            save_image_batch(images, setting_dir, start_index, timer, gpu_monitor)

            if grid_max_images > 0 and len(grid_images) < grid_max_images:
                remaining = grid_max_images - len(grid_images)
                grid_images.extend(images[:remaining])

            del cond_context, uncond_context, latents, images

            if device.type == "cuda":
                torch.cuda.empty_cache()

        if grid_images:
            with timer("save_grid"), gpu_monitor.track("save_grid"):
                try:
                    save_image_grid(grid_images, setting_dir / "grid.png")
                except Exception:
                    pass

    timer.print_summary(title=f"Timing: {setting_name}")
    timer.save_json(setting_dir / "timing.json")

    gpu_monitor.print_summary(title=f"GPU usage: {setting_name}")
    gpu_monitor.save_json(setting_dir / "gpu_usage.json")

    return {
        "setting": setting_name,
        "sampler": sampler_name,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "eta": eta,
        "output_dir": str(setting_dir),
        "num_images": len(prompts),
        "generation_batch_size": generation_batch_size,
        "num_batches": num_batches,
        "grid_max_images": grid_max_images,
        "timing": timer.summary(),
        "gpu_usage": gpu_monitor.summary(),
    }


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg["evaluation"].get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(str(cfg["evaluation"].get("precision", "bf16")))

    output_root = Path(cfg["evaluation"]["output_dir"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    eval_cfg = cfg["evaluation"]

    max_images = int(eval_cfg.get("max_images", eval_cfg.get("num_prompts", 16)))
    prompts = load_first_val_captions(cfg["data"]["val_latent_dir"], max_images=max_images)

    if len(prompts) == 0:
        raise RuntimeError("No validation captions found.")

    if max_images > 0 and len(prompts) < max_images:
        print(f"Warning: requested max_images={max_images}, but only found {len(prompts)} captions in validation latents.")

    generation_batch_size = int(eval_cfg.get("generation_batch_size", eval_cfg.get("batch_size", 8)))

    if generation_batch_size <= 0:
        raise ValueError(f"evaluation.generation_batch_size must be > 0, got {generation_batch_size}")

    generation_batch_size = min(generation_batch_size, len(prompts))
    grid_max_images = int(eval_cfg.get("grid_max_images", 64))

    total_desc = "all" if max_images == 0 else str(max_images)

    print(
        f"Using {len(prompts)} validation prompts/images "
        f"(requested max_images={total_desc}); "
        f"generation_batch_size={generation_batch_size}; "
        f"expected batches per setting={math.ceil(len(prompts) / generation_batch_size)}."
    )

    ldm_cfg_path = cfg["ldm"]["config"]
    ldm_ckpt = cfg["ldm"]["checkpoint"]

    vae_cfg_path = cfg["vae"]["config"]
    vae_ckpt = cfg["vae"]["checkpoint"]

    scaling_factor = float(cfg["vae"].get("scaling_factor", 1.0))

    print("Loading LDM...")
    model, ldm_cfg = load_ldm_model(ldm_cfg_path, ldm_ckpt, device, dtype)
    diffusion = build_diffusion(ldm_cfg).to(device)

    print("Loading VAE...")
    vae = load_vae_from_config(vae_cfg_path, vae_ckpt, device, dtype)

    # If main config does not define scaling_factor, use VAE attribute when present.
    scaling_factor = float(cfg["vae"].get("scaling_factor", getattr(vae, "scaling_factor", scaling_factor)))

    print("Loading text encoder...")
    text_encoder = build_text_encoder(ldm_cfg, device, dtype)

    latent_channels = int(ldm_cfg["model"].get("in_channels", 8))
    image_size = int(cfg["evaluation"].get("resolution", 256))
    latent_size = image_size // 8

    empty_text = str(cfg.get("conditioning", {}).get("empty_text", ""))
    eta = float(cfg.get("ddim", {}).get("eta", 0.0))

    with open(output_root / "prompts.json", "w", encoding="utf-8") as f:
        json.dump([{"index": i, "prompt": p} for i, p in enumerate(prompts)], f, indent=2)

    reports = []

    ddpm_cfg = cfg.get("ddpm", {})

    if bool(ddpm_cfg.get("enabled", True)):
        for guidance_scale in ddpm_cfg.get("guidance_scales", [1.0]):
            setting_name = f"ddpm_steps{diffusion.num_timesteps:04d}_cfg{sanitize_float(float(guidance_scale))}"

            reports.append(
                run_setting(
                    setting_name=setting_name,
                    sampler_name="ddpm",
                    steps=diffusion.num_timesteps,
                    guidance_scale=float(guidance_scale),
                    eta=0.0,
                    model=model,
                    diffusion=diffusion,
                    vae=vae,
                    prompts=prompts,
                    latent_channels=latent_channels,
                    latent_size=latent_size,
                    generation_batch_size=generation_batch_size,
                    scaling_factor=scaling_factor,
                    text_encoder=text_encoder,
                    empty_text=empty_text,
                    device=device,
                    dtype=dtype,
                    output_root=output_root,
                    monitor_gpu=bool(cfg.get("evaluation", {}).get("monitor_gpu", args.monitor_gpu)),
                    gpu_sample_interval=float(cfg.get("evaluation", {}).get("gpu_sample_interval", args.gpu_sample_interval)),
                    use_nvidia_smi=not bool(cfg.get("evaluation", {}).get("no_nvidia_smi", args.no_nvidia_smi)),
                    grid_max_images=grid_max_images,
                )
            )

    ddim_cfg = cfg.get("ddim", {})

    if bool(ddim_cfg.get("enabled", True)):
        for steps in ddim_cfg.get("steps", [50, 100, 150]):
            for guidance_scale in ddim_cfg.get("guidance_scales", [1.0, 2.0, 3.0]):
                setting_name = f"ddim_steps{int(steps):04d}_cfg{sanitize_float(float(guidance_scale))}"
                reports.append(
                    run_setting(
                        setting_name=setting_name,
                        sampler_name="ddim",
                        steps=int(steps),
                        guidance_scale=float(guidance_scale),
                        eta=eta,
                        model=model,
                        diffusion=diffusion,
                        vae=vae,
                        prompts=prompts,
                        latent_channels=latent_channels,
                        latent_size=latent_size,
                        generation_batch_size=generation_batch_size,
                        scaling_factor=scaling_factor,
                        text_encoder=text_encoder,
                        empty_text=empty_text,
                        device=device,
                        dtype=dtype,
                        output_root=output_root,
                        monitor_gpu=bool(cfg.get("evaluation", {}).get("monitor_gpu", args.monitor_gpu)),
                        gpu_sample_interval=float(cfg.get("evaluation", {}).get("gpu_sample_interval", args.gpu_sample_interval)),
                        use_nvidia_smi=not bool(cfg.get("evaluation", {}).get("no_nvidia_smi", args.no_nvidia_smi)),
                        grid_max_images=grid_max_images,
                    )
                )

    with open(output_root / "timing_report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print("Saved timing report:", output_root / "timing_report.json")


if __name__ == "__main__":
    main()