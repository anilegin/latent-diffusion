from __future__ import annotations

import argparse
import inspect
import json
import math
import os
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

from src.utils.timer import Timer
from src.utils.gpu_monitor import GPUMonitor


DEFAULT_LDM_TEXT2IM_MODEL = "CompVis/ldm-text2im-large-256"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--monitor-gpu", action="store_true", help="Enable per-section GPU memory/utilization monitoring.")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25, help="Seconds between nvidia-smi utilization samples.")
    parser.add_argument("--no-nvidia-smi", action="store_true", help="Disable nvidia-smi utilization sampling and keep only PyTorch CUDA memory stats.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise RuntimeError(f"Empty YAML config: {path}")
    return data


def get_dtype(name: str):
    name = name.lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown precision={name}")


def sanitize_float(x: float) -> str:
    return str(x).replace(".", "p")


def safe_torch_load(path: str | Path, map_location="cpu"):
    """Load local latent shards safely on newer PyTorch, with fallback for older files."""
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


def _normalize_caption_item(item: Any) -> str | None:
    """Extract one caption string from a shard/json/txt item."""
    if item is None:
        return None

    if isinstance(item, str):
        cap = item.strip()
        return cap if cap else None

    if isinstance(item, (list, tuple)):
        if len(item) == 0:
            return None
        return _normalize_caption_item(item[0])

    if isinstance(item, dict):
        for key in ("prompt", "caption", "text", "captions"):
            if key in item:
                return _normalize_caption_item(item[key])
        return None

    cap = str(item).strip()
    return cap if cap else None


def _apply_max_images(captions: list[str], max_images: int) -> list[str]:
    if max_images < 0:
        raise ValueError(f"evaluation.max_images must be >= 0, got {max_images}")
    if max_images == 0:
        return captions
    return captions[:max_images]


def load_captions_from_latent_shards(latent_dir: str | Path, max_images: int, caption_key: str = "captions") -> list[str]:
    """
    max_images > 0: return first max_images captions.
    max_images == 0: return all captions found in the shards.
    """
    latent_dir = Path(latent_dir).expanduser()
    shard_paths = sorted(latent_dir.glob("*.pt"))
    if not shard_paths:
        raise RuntimeError(f"No .pt latent shards found in {latent_dir}")

    if max_images < 0:
        raise ValueError(f"evaluation.max_images must be >= 0, got {max_images}")

    captions: list[str] = []
    limit = None if max_images == 0 else max_images

    for shard_path in shard_paths:
        payload = safe_torch_load(shard_path, map_location="cpu")
        if caption_key not in payload:
            raise KeyError(f"Caption key '{caption_key}' not found in {shard_path}. Available keys: {list(payload.keys())}")
        shard_caps = payload[caption_key]

        for item in shard_caps:
            cap = _normalize_caption_item(item)
            if cap is None:
                continue
            captions.append(cap)
            if limit is not None and len(captions) >= limit:
                return captions

    return captions


def load_captions_from_json(path: str | Path, max_images: int) -> list[str]:
    path = Path(path).expanduser()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Supported formats:
    #   ["caption", ...]
    #   [{"prompt": "..."}, ...]
    #   {"prompts": [...]}
    #   {"captions": [...]}
    if isinstance(data, dict):
        for key in ("prompts", "captions", "data", "items"):
            if key in data:
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"Unsupported captions JSON format in {path}. Expected a list or dict containing prompts/captions.")

    captions: list[str] = []
    for item in data:
        cap = _normalize_caption_item(item)
        if cap is not None:
            captions.append(cap)
    return _apply_max_images(captions, max_images)


def load_captions_from_txt(path: str | Path, max_images: int) -> list[str]:
    path = Path(path).expanduser()
    captions: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cap = line.strip()
            if cap:
                captions.append(cap)
    return _apply_max_images(captions, max_images)


def load_eval_captions(data_cfg: dict[str, Any], max_images: int) -> tuple[list[str], dict[str, Any]]:

    if max_images < 0:
        raise ValueError(f"evaluation.max_images must be >= 0, got {max_images}")

    if data_cfg.get("captions_file"):
        path = Path(str(data_cfg["captions_file"])).expanduser()
        suffix = path.suffix.lower()
        if suffix in {".json"}:
            captions = load_captions_from_json(path, max_images=max_images)
        elif suffix in {".jsonl"}:
            captions = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cap = _normalize_caption_item(json.loads(line))
                    if cap is not None:
                        captions.append(cap)
            captions = _apply_max_images(captions, max_images)
        else:
            captions = load_captions_from_txt(path, max_images=max_images)
        source = {"type": "captions_file", "path": str(path)}
        return captions, source

    if data_cfg.get("val_latent_dir"):
        caption_key = str(data_cfg.get("caption_key", "captions"))
        captions = load_captions_from_latent_shards(
            latent_dir=data_cfg["val_latent_dir"],
            max_images=max_images,
            caption_key=caption_key,
        )
        source = {"type": "latent_shards_captions_only", "path": str(data_cfg["val_latent_dir"]), "caption_key": caption_key}
        return captions, source

    raise KeyError("Config must provide either data.captions_file or data.val_latent_dir")


def batched_indices(total: int, batch_size: int):
    if batch_size <= 0:
        raise ValueError(f"evaluation.generation_batch_size must be > 0, got {batch_size}")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield start, end


def save_prompts_json(prompts: list[str], path: Path, start_index: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [{"index": start_index + i, "prompt": p} for i, p in enumerate(prompts)],
            f,
            indent=2,
        )


def make_generators(device: torch.device, seed: int, start: int, count: int) -> list[torch.Generator]:
    gens: list[torch.Generator] = []
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    for offset in range(count):
        g = torch.Generator(device=gen_device)
        g.manual_seed(seed + start + offset)
        gens.append(g)
    return gens


def _module_eval_if_present(pipe, name: str):
    module = getattr(pipe, name, None)
    if module is not None and hasattr(module, "eval"):
        module.eval()


def build_pipeline(cfg: dict[str, Any], device: torch.device, dtype: torch.dtype, timer: Timer, gpu_monitor: GPUMonitor):
    try:
        from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline
    except Exception as exc:
        raise RuntimeError(
            "This script requires diffusers. Install it in your environment before running on the GPU node."
        ) from exc

    model_cfg = cfg.get("ldm", cfg.get("sd", {}))
    model_name = str(model_cfg.get("model_name", model_cfg.get("model_id", DEFAULT_LDM_TEXT2IM_MODEL)))
    local_files_only = bool(model_cfg.get("local_files_only", True))
    use_safetensors = model_cfg.get("use_safetensors", None)
    variant = model_cfg.get("variant", None)

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("DIFFUSERS_OFFLINE", "1")

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
    }
    if use_safetensors is not None:
        kwargs["use_safetensors"] = bool(use_safetensors)
    if variant is not None:
        kwargs["variant"] = str(variant)

    with timer("load_ldm_text2im_pipeline"), gpu_monitor.track("load_ldm_text2im_pipeline"):
        pipe = DiffusionPipeline.from_pretrained(model_name, **kwargs)
        pipe = pipe.to(device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        for module_name in ("unet", "vqvae", "vae", "bert", "text_encoder", "tokenizer"):
            _module_eval_if_present(pipe, module_name)

        if bool(model_cfg.get("enable_attention_slicing", False)) and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if bool(model_cfg.get("enable_xformers_memory_efficient_attention", False)) and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                print(f"Warning: could not enable xformers memory efficient attention: {exc}")

    pipe._eval_ddim_scheduler_cls = DDIMScheduler
    pipe._eval_ddpm_scheduler_cls = DDPMScheduler
    return pipe, model_name, local_files_only

def _build_scheduler_from_current_config(pipe, scheduler_cls):
    try:
        scheduler = scheduler_cls.from_config(pipe.scheduler.config, steps_offset=0)
    except TypeError:
        scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        try:
            scheduler.register_to_config(steps_offset=0)
        except Exception:
            pass
    except Exception:
        scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        try:
            scheduler.register_to_config(steps_offset=0)
        except Exception:
            pass

    # Keep this robust across Diffusers versions. Some configs are FrozenDict-like,
    # so register_to_config is the safest public way to override config values.
    try:
        scheduler.register_to_config(steps_offset=0)
    except Exception:
        pass
    return scheduler


def set_scheduler(pipe, sampler_name: str):
    if sampler_name == "ddim":
        pipe.scheduler = _build_scheduler_from_current_config(pipe, pipe._eval_ddim_scheduler_cls)
    elif sampler_name == "ddpm":
        pipe.scheduler = _build_scheduler_from_current_config(pipe, pipe._eval_ddpm_scheduler_cls)
    else:
        raise ValueError(f"Unknown sampler_name={sampler_name}")


def _filter_pipe_call_kwargs(pipe, call_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by the active Diffusers pipeline."""
    try:
        sig = inspect.signature(pipe.__call__)
    except Exception:
        return call_kwargs

    parameters = sig.parameters
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    if accepts_var_kwargs:
        return call_kwargs

    filtered: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in call_kwargs.items():
        if key in parameters:
            filtered[key] = value
        else:
            dropped.append(key)
    if dropped:
        print(f"Info: pipeline call does not accept {dropped}; skipping those kwargs.")
    return filtered


def generate_batch(
    pipe,
    prompts: list[str],
    negative_prompt: str,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    eta: float,
    seed: int,
    start_index: int,
    device: torch.device,
    timer: Timer,
    gpu_monitor: GPUMonitor,
) -> list[Image.Image]:
    train_steps = int(getattr(pipe.scheduler.config, "num_train_timesteps", 1000))
    if steps > train_steps:
        raise ValueError(
            f"Requested num_inference_steps={steps}, but scheduler has only "
            f"num_train_timesteps={train_steps}. Use steps <= {train_steps}."
        )

    generators = make_generators(device, seed=seed, start=start_index, count=len(prompts))

    call_kwargs: dict[str, Any] = {
        "prompt": prompts,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generators,
        "output_type": "pil",
    }

    # eta is a DDIM-specific stochasticity parameter. Do not pass it for DDPM.
    if "DDIM" in pipe.scheduler.__class__.__name__.upper():
        call_kwargs["eta"] = eta

    # Stable Diffusion accepts `negative_prompt`; the original LDM text-to-image
    # pipeline usually does not. Keep it only when the active pipeline supports it.
    if negative_prompt:
        call_kwargs["negative_prompt"] = [negative_prompt] * len(prompts)

    call_kwargs = _filter_pipe_call_kwargs(pipe, call_kwargs)

    with timer("ldm_text2im_generate_batch"), gpu_monitor.track("ldm_text2im_generate_batch"):
        out = pipe(**call_kwargs)
    return out.images

def run_setting(
    setting_name: str,
    sampler_name: str,
    steps: int,
    guidance_scale: float,
    eta: float,
    pipe,
    prompts: list[str],
    seed: int,
    resolution: int,
    generation_batch_size: int,
    grid_max_images: int,
    empty_text: str,
    device: torch.device,
    output_root: Path,
    monitor_gpu: bool,
    gpu_sample_interval: float,
    use_nvidia_smi: bool,
):
    setting_dir = output_root / setting_name
    setting_dir.mkdir(parents=True, exist_ok=True)

    timer = Timer(sync_cuda=True)
    gpu_monitor = GPUMonitor(
        enabled=monitor_gpu,
        device=device,
        sync_cuda=True,
        sample_interval_s=gpu_sample_interval,
        use_nvidia_smi=use_nvidia_smi,
    )

    saved_images_for_grid: list[Image.Image] = []
    batch_reports: list[dict[str, Any]] = []

    set_scheduler(pipe, sampler_name)
    scheduler_train_steps = int(getattr(pipe.scheduler.config, "num_train_timesteps", 1000))
    scheduler_steps_offset = int(getattr(pipe.scheduler.config, "steps_offset", 0))
    if steps > scheduler_train_steps:
        raise ValueError(
            f"{setting_name}: requested steps={steps}, but scheduler has "
            f"num_train_timesteps={scheduler_train_steps}."
        )
    if scheduler_steps_offset != 0:
        print(f"Warning: {setting_name} scheduler steps_offset={scheduler_steps_offset}; expected 0 after patch.")

    with timer("total"), gpu_monitor.track("total"):
        save_prompts_json(prompts, setting_dir / "prompts.json")

        batches = list(batched_indices(len(prompts), generation_batch_size))
        pbar = tqdm(batches, desc=f"{setting_name}", leave=True)
        for batch_id, (start, end) in enumerate(pbar):
            batch_prompts = prompts[start:end]
            images = generate_batch(
                pipe=pipe,
                prompts=batch_prompts,
                negative_prompt=empty_text,
                height=resolution,
                width=resolution,
                steps=steps,
                guidance_scale=guidance_scale,
                eta=eta,
                seed=seed,
                start_index=start,
                device=device,
                timer=timer,
                gpu_monitor=gpu_monitor,
            )

            with timer("save_images"), gpu_monitor.track("save_images"):
                for local_idx, image in enumerate(images):
                    global_idx = start + local_idx
                    image.save(setting_dir / f"sample_{global_idx:04d}.png")
                    if grid_max_images > 0 and len(saved_images_for_grid) < grid_max_images:
                        saved_images_for_grid.append(image.copy())

            batch_reports.append({
                "batch_id": batch_id,
                "start_index": start,
                "end_index_exclusive": end,
                "num_images": end - start,
            })

        if grid_max_images > 0:
            with timer("save_grid"), gpu_monitor.track("save_grid"):
                save_image_grid(saved_images_for_grid, setting_dir / "grid.png")

    timer.print_summary(title=f"Timing: {setting_name}")
    timer.save_json(setting_dir / "timing.json")
    gpu_monitor.print_summary(title=f"GPU usage: {setting_name}")
    gpu_monitor.save_json(setting_dir / "gpu_usage.json")

    with open(setting_dir / "batches.json", "w", encoding="utf-8") as f:
        json.dump(batch_reports, f, indent=2)

    return {
        "setting": setting_name,
        "sampler": sampler_name,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "eta": eta,
        "num_images": len(prompts),
        "generation_batch_size": generation_batch_size,
        "scheduler_class": pipe.scheduler.__class__.__name__,
        "scheduler_num_train_timesteps": int(getattr(pipe.scheduler.config, "num_train_timesteps", 1000)),
        "scheduler_steps_offset": int(getattr(pipe.scheduler.config, "steps_offset", 0)),
        "output_dir": str(setting_dir),
        "timing": timer.summary(),
        "gpu_usage": gpu_monitor.summary(),
        "batches": batch_reports,
    }


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    eval_cfg = cfg.get("evaluation", {})

    seed = int(eval_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*", category=FutureWarning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(str(eval_cfg.get("precision", "bf16")))

    output_root = Path(eval_cfg["output_dir"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Backward compatible fallback: num_prompts still works, but max_images is preferred.
    max_images = int(eval_cfg.get("max_images", eval_cfg.get("num_prompts", 16)))
    generation_batch_size = int(eval_cfg.get("generation_batch_size", 16))
    resolution = int(eval_cfg.get("resolution", 256))
    grid_max_images = int(eval_cfg.get("grid_max_images", 64))

    prompts, caption_source = load_eval_captions(cfg.get("data", {}), max_images=max_images)
    if len(prompts) == 0:
        raise RuntimeError("No captions found.")

    # LDM text-to-image requires dimensions divisible by 8.
    if resolution % 8 != 0:
        raise ValueError(f"evaluation.resolution must be divisible by 8, got {resolution}")
    if generation_batch_size <= 0:
        raise ValueError(f"evaluation.generation_batch_size must be > 0, got {generation_batch_size}")

    empty_text = str(cfg.get("conditioning", {}).get("empty_text", ""))
    monitor_gpu = bool(eval_cfg.get("monitor_gpu", args.monitor_gpu))
    gpu_sample_interval = float(eval_cfg.get("gpu_sample_interval", args.gpu_sample_interval))
    use_nvidia_smi = not bool(eval_cfg.get("no_nvidia_smi", args.no_nvidia_smi))

    save_prompts_json(prompts, output_root / "prompts.json")

    load_timer = Timer(sync_cuda=True)
    load_gpu_monitor = GPUMonitor(
        enabled=monitor_gpu,
        device=device,
        sync_cuda=True,
        sample_interval_s=gpu_sample_interval,
        use_nvidia_smi=use_nvidia_smi,
    )

    print("Loading LDM text-to-image pipeline...")
    pipe, model_name, local_files_only = build_pipeline(
        cfg=cfg,
        device=device,
        dtype=dtype,
        timer=load_timer,
        gpu_monitor=load_gpu_monitor,
    )
    load_timer.print_summary(title="Timing: model load")
    load_timer.save_json(output_root / "load_timing.json")
    load_gpu_monitor.print_summary(title="GPU usage: model load")
    load_gpu_monitor.save_json(output_root / "load_gpu_usage.json")

    autoencoder = getattr(pipe, "vqvae", getattr(pipe, "vae", None))
    autoencoder_cfg = getattr(autoencoder, "config", None)
    autoencoder_scaling_factor = getattr(autoencoder_cfg, "scaling_factor", None)
    autoencoder_sample_size = getattr(autoencoder_cfg, "sample_size", None)
    autoencoder_latent_channels = getattr(autoencoder_cfg, "latent_channels", None)

    print(f"Model: {model_name}")
    print(f"local_files_only: {local_files_only}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Caption source: {caption_source}")
    print(f"Total prompts/images: {len(prompts)}")
    print(f"Generation batch size: {generation_batch_size}")
    print(f"Pipeline class: {pipe.__class__.__name__}")
    print(f"Scheduler: {pipe.scheduler.__class__.__name__}")
    print(f"Autoencoder class: {autoencoder.__class__.__name__ if autoencoder is not None else None}")
    print(f"Autoencoder scaling_factor: {autoencoder_scaling_factor}")
    print("Note: the LDM text-to-image pipeline decodes generated latents; it does not encode input images during text-to-image evaluation.")

    reports: list[dict[str, Any]] = []

    ddpm_cfg = cfg.get("ddpm", {})
    if bool(ddpm_cfg.get("enabled", True)):
        default_ddpm_steps = int(getattr(pipe.scheduler.config, "num_train_timesteps", 1000))
        ddpm_steps = int(ddpm_cfg.get("steps", default_ddpm_steps))
        for guidance_scale in ddpm_cfg.get("guidance_scales", [1.0]):
            setting_name = f"ldm_text2im_ddpm_steps{ddpm_steps:04d}_cfg{sanitize_float(float(guidance_scale))}"
            reports.append(
                run_setting(
                    setting_name=setting_name,
                    sampler_name="ddpm",
                    steps=ddpm_steps,
                    guidance_scale=float(guidance_scale),
                    eta=0.0,
                    pipe=pipe,
                    prompts=prompts,
                    seed=seed,
                    resolution=resolution,
                    generation_batch_size=generation_batch_size,
                    grid_max_images=grid_max_images,
                    empty_text=empty_text,
                    device=device,
                    output_root=output_root,
                    monitor_gpu=monitor_gpu,
                    gpu_sample_interval=gpu_sample_interval,
                    use_nvidia_smi=use_nvidia_smi,
                )
            )

    ddim_cfg = cfg.get("ddim", {})
    eta = float(ddim_cfg.get("eta", 0.0))
    for steps in ddim_cfg.get("steps", [50, 100, 150]):
        for guidance_scale in ddim_cfg.get("guidance_scales", [1.0, 3.0, 5.0]):
            setting_name = f"ldm_text2im_ddim_steps{int(steps):04d}_cfg{sanitize_float(float(guidance_scale))}"
            reports.append(
                run_setting(
                    setting_name=setting_name,
                    sampler_name="ddim",
                    steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    eta=eta,
                    pipe=pipe,
                    prompts=prompts,
                    seed=seed,
                    resolution=resolution,
                    generation_batch_size=generation_batch_size,
                    grid_max_images=grid_max_images,
                    empty_text=empty_text,
                    device=device,
                    output_root=output_root,
                    monitor_gpu=monitor_gpu,
                    gpu_sample_interval=gpu_sample_interval,
                    use_nvidia_smi=use_nvidia_smi,
                )
            )

    metadata = {
        "model_name": model_name,
        "local_files_only": local_files_only,
        "seed": seed,
        "precision": str(eval_cfg.get("precision", "bf16")),
        "resolution": resolution,
        "num_images": len(prompts),
        "max_images_config": max_images,
        "generation_batch_size": generation_batch_size,
        "caption_source": caption_source,
        "autoencoder_scaling_factor": autoencoder_scaling_factor,
        "autoencoder_sample_size": autoencoder_sample_size,
        "autoencoder_latent_channels": autoencoder_latent_channels,
        "note": "LDM text-to-image mode decodes generated latents. It does not encode input images during evaluation.",
        "load_timing": load_timer.summary(),
        "load_gpu_usage": load_gpu_monitor.summary(),
        "settings": reports,
    }

    with open(output_root / "ldm_text2im_timing_report.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved LDM text-to-image timing report:", output_root / "ldm_text2im_timing_report.json")


if __name__ == "__main__":
    main()
