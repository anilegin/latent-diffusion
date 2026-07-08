# python evaluate_ldm_metrics.py \
#   --config path/to/your_eval_config.yaml \
#   --output-dir outputs/ldm_metrics \
#   --metrics fid kid is \
#   --force-rebuild-reference

# python evaluate_ldm_metrics.py \
#   --config path/to/your_eval_config.yaml \
#   --generated-root /leonardo_scratch/large/userexternal/aegin000/latent-diffusion-eval/custom_timing \
#   --output-dir outputs/ldm_metrics \
#   --metrics fid kid is clip \
#   --force-rebuild-reference



from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Make local project imports work when this file is placed in scripts/ or run from project root.
_THIS_FILE = Path(__file__).resolve()
_CWD = Path.cwd().resolve()
if (_CWD / "src").exists():
    sys.path.insert(0, str(_CWD))
elif len(_THIS_FILE.parents) >= 2 and (_THIS_FILE.parents[1] / "src").exists():
    sys.path.insert(0, str(_THIS_FILE.parents[1]))
else:
    # Fallback: allows running from unusual locations if PYTHONPATH is already set.
    sys.path.insert(0, str(_CWD))

from src.utils.evaluation import (
    compute_clip_score_local,
    compute_fid,
    compute_inception_score,
    compute_kid,
    ensure_dir,
    list_reference_images,
    list_sample_images,
    load_prompts,
    load_yaml,
    safe_torch_load,
)


# -----------------------------
# Reference image construction
# -----------------------------

def find_payload_key(payload: dict[str, Any], candidate_keys: list[str]) -> str | None:
    for key in candidate_keys:
        if key in payload:
            return key
    return None


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x) if not isinstance(x, (str, bytes)) and hasattr(x, "__iter__") else [x]


def resolve_image_path(raw: Any, image_root: Path | None, shard_dir: Path) -> Path | None:
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]

    s = str(raw)
    if not s:
        return None

    p = Path(s)
    candidates: list[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        if image_root is not None:
            candidates.append(image_root / p)
            candidates.append(image_root / p.name)
        candidates.append(shard_dir / p)
        candidates.append(Path.cwd() / p)

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    return None


def pil_save_rgb(src: Path, dst: Path, resolution: int | None) -> None:
    img = Image.open(src).convert("RGB")
    if resolution is not None:
        img = img.resize((resolution, resolution), Image.BICUBIC)
    img.save(dst)


def tensor_image_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu()

    if x.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape={tuple(x.shape)}")

    # CHW -> HWC if needed.
    if x.shape[0] in (1, 3, 4):
        x = x.permute(1, 2, 0)

    if x.shape[-1] == 1:
        x = x.repeat(1, 1, 3)
    if x.shape[-1] == 4:
        x = x[..., :3]

    if x.dtype.is_floating_point:
        # Common conventions: [-1, 1] or [0, 1].
        if float(x.min()) < -0.05:
            x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)
        x = (x * 255.0).round().to(torch.uint8)
    else:
        x = x.clamp(0, 255).to(torch.uint8)

    arr = x.numpy()
    return Image.fromarray(arr, mode="RGB")


def load_vae_from_config(config_path: str | Path, checkpoint_path: str | Path, device: torch.device):
    from src.utils.config import load_config
    from src.network.autoencoder.vae import AutoencoderKL

    cfg = load_config(config_path)
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)

    vae = AutoencoderKL(**model_cfg)

    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint

    vae.load_state_dict(state_dict, strict=True)
    vae.to(device)
    vae.eval()
    return vae


@torch.no_grad()
def decode_latents_to_reference_images(
    latents: torch.Tensor,
    vae,
    scaling_factor: float,
    out_dir: Path,
    start_index: int,
    device: torch.device,
    batch_size: int,
    resolution: int | None,
) -> int:
    written = 0

    for start in range(0, latents.shape[0], batch_size):
        batch = latents[start: start + batch_size].to(device=device, dtype=torch.float32)

        try:
            images = vae.decode(batch, unscale=True)
        except TypeError:
            images = vae.decode(batch / scaling_factor)

        if hasattr(images, "sample"):
            images = images.sample

        images = ((images + 1.0) / 2.0).clamp(0.0, 1.0)

        for i in range(images.shape[0]):
            img = tensor_image_to_pil(images[i])
            if resolution is not None:
                img = img.resize((resolution, resolution), Image.BICUBIC)
            img.save(out_dir / f"ref_{start_index + written:06d}.png")
            written += 1

    return written


def build_reference_from_latent_shards(
    latent_dir: Path,
    out_dir: Path,
    max_images: int,
    resolution: int | None,
    image_root: Path | None,
    vae_config: Path | None,
    vae_checkpoint: Path | None,
    scaling_factor: float,
    device: torch.device,
    decode_batch_size: int,
    force: bool,
) -> tuple[list[Path], dict[str, Any]]:
    out_dir = Path(out_dir)

    # If the user requested a rebuild, remove the stale cache first.
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)

    # Create the generated reference-image cache on first run.
    ensure_dir(out_dir)

    existing = list_reference_images(out_dir)
    if existing and not force:
        info = {
            "reference_dir": str(out_dir),
            "source": "existing_reference_dir",
            "num_images": len(existing),
            "note": "Using existing reference images. Pass --force-rebuild-reference to recreate them.",
        }
        return existing[:max_images], info

    shard_paths = sorted(latent_dir.glob("*.pt"))
    if not shard_paths:
        raise RuntimeError(f"No .pt latent shards found in {latent_dir}")

    path_keys = [
        "image_paths", "image_path", "paths", "path", "file_paths", "file_path",
        "filenames", "filename", "file_names", "file_name", "images_paths",
    ]
    image_tensor_keys = ["images", "image", "pixels", "pixel_values", "original_images"]
    latent_keys = ["latents", "latent", "z", "zs", "encoded_latents", "vae_latents"]

    written = 0
    source = None
    vae = None
    first_payload_keys: list[str] = []

    for shard_path in tqdm(shard_paths, desc="Building reference images"):
        if written >= max_images:
            break

        payload = safe_torch_load(shard_path, map_location="cpu")
        if not isinstance(payload, dict):
            continue

        if not first_payload_keys:
            first_payload_keys = list(payload.keys())

        # 1) Best case: shard stores paths to original images.
        key = find_payload_key(payload, path_keys)
        if key is not None:
            source = "original_image_paths_from_latent_shards"
            vals = as_list(payload[key])

            for raw in vals:
                if written >= max_images:
                    break

                src = resolve_image_path(raw, image_root=image_root, shard_dir=latent_dir)
                if src is None:
                    continue

                dst = out_dir / f"ref_{written:06d}.png"
                pil_save_rgb(src, dst, resolution=resolution)
                written += 1

            continue

        # 2) Shard directly stores image tensors.
        key = find_payload_key(payload, image_tensor_keys)
        if key is not None:
            source = "image_tensors_from_latent_shards"
            imgs = payload[key]
            if not torch.is_tensor(imgs):
                imgs = torch.as_tensor(imgs)

            for i in range(imgs.shape[0]):
                if written >= max_images:
                    break

                img = tensor_image_to_pil(imgs[i])
                if resolution is not None:
                    img = img.resize((resolution, resolution), Image.BICUBIC)
                img.save(out_dir / f"ref_{written:06d}.png")
                written += 1

            continue

        # 3) Last resort: decode stored VAE latents into reference reconstructions.
        key = find_payload_key(payload, latent_keys)
        if key is not None:
            source = "vae_decoded_latents_from_latent_shards"

            if vae is None:
                if vae_config is None or vae_checkpoint is None:
                    raise RuntimeError(
                        "Latent shards do not contain original image paths or image tensors. "
                        "They contain latents, so --vae-config and --vae-checkpoint are required "
                        "to build reference images."
                    )
                vae = load_vae_from_config(vae_config, vae_checkpoint, device=device)

            latents = payload[key]
            if not torch.is_tensor(latents):
                latents = torch.as_tensor(latents)

            remaining = max_images - written
            latents = latents[:remaining]

            n = decode_latents_to_reference_images(
                latents=latents,
                vae=vae,
                scaling_factor=scaling_factor,
                out_dir=out_dir,
                start_index=written,
                device=device,
                batch_size=decode_batch_size,
                resolution=resolution,
            )
            written += n
            continue

    refs = list_reference_images(out_dir)
    if not refs:
        raise RuntimeError(
            "Could not create reference images from latent shards. "
            f"First shard keys seen: {first_payload_keys}. "
            "If your shard stores original file paths under a different key, add it to path_keys in this script."
        )

    info = {
        "reference_dir": str(out_dir),
        "source": source,
        "num_images": len(refs),
        "latent_dir": str(latent_dir),
        "image_root": str(image_root) if image_root is not None else None,
        "vae_config": str(vae_config) if vae_config is not None else None,
        "vae_checkpoint": str(vae_checkpoint) if vae_checkpoint is not None else None,
        "scaling_factor": scaling_factor,
        "resolution": resolution,
        "note": (
            "Standard FID/KID should ideally use true original image files. "
            "If source is vae_decoded_latents_from_latent_shards, report the metric as FID/KID versus VAE reconstructions, not canonical dataset FID."
        ),
    }

    with open(out_dir / "reference_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return refs[:max_images], info


def evaluate_setting(
    setting_dir: Path,
    real_paths: list[Path],
    args,
    device: torch.device,
) -> dict[str, Any]:
    fake_paths = list_sample_images(setting_dir)
    if not fake_paths:
        raise RuntimeError(f"No generated sample images found in {setting_dir}")

    n = min(len(real_paths), len(fake_paths), args.max_images)
    real_eval = real_paths[:n]
    fake_eval = fake_paths[:n]

    result: dict[str, Any] = {
        "setting": setting_dir.name,
        "setting_dir": str(setting_dir),
        "num_real": len(real_eval),
        "num_fake": len(fake_eval),
    }

    requested = set(args.metrics)

    if "fid" in requested:
        try:
            result["fid"] = compute_fid(real_eval, fake_eval, args.batch_size, args.eval_resolution, device)
        except Exception as exc:
            result["fid"] = None
            result["fid_error"] = repr(exc)

    if "kid" in requested:
        try:
            kid_mean, kid_std = compute_kid(real_eval, fake_eval, args.batch_size, args.eval_resolution, device)
            result["kid_mean"] = kid_mean
            result["kid_std"] = kid_std
        except Exception as exc:
            result["kid_mean"] = None
            result["kid_std"] = None
            result["kid_error"] = repr(exc)

    if "is" in requested:
        try:
            is_mean, is_std = compute_inception_score(fake_eval, args.batch_size, args.eval_resolution, device)
            result["inception_score_mean"] = is_mean
            result["inception_score_std"] = is_std
        except Exception as exc:
            result["inception_score_mean"] = None
            result["inception_score_std"] = None
            result["inception_score_error"] = repr(exc)

    if "clip" in requested:
        prompts = load_prompts(setting_dir, n)
        if prompts is None or len(prompts) < n:
            result["clip_score"] = None
            result["clip_score_error"] = "Missing or incomplete prompts.json in setting folder."
        else:
            try:
                result["clip_score"] = compute_clip_score_local(
                    fake_paths=fake_eval,
                    prompts=prompts[:n],
                    batch_size=args.batch_size,
                    device=device,
                    clip_model_name=args.clip_model_name,
                    local_files_only=args.local_files_only,
                )
            except Exception as exc:
                result["clip_score"] = None
                result["clip_score_error"] = repr(exc)

    return result


def write_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    json_path = out_dir / "metrics_summary.json"
    csv_path = out_dir / "metrics_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "setting",
        "num_real",
        "num_fake",
        "fid",
        "kid_mean",
        "kid_std",
        "inception_score_mean",
        "inception_score_std",
        "clip_score",
        "setting_dir",
    ]

    # Include error columns if any exist.
    error_keys = sorted({k for r in results for k in r.keys() if k.endswith("_error")})
    fieldnames.extend(error_keys)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LDM generated folders with common image-generation metrics."
    )

    parser.add_argument("--config", type=str, default=None, help="Your timing/evaluation YAML config. Used to infer paths.")
    parser.add_argument("--generated-root", type=str, default=None, help="Folder containing ddpm_steps*/ddim_steps* folders. Defaults to evaluation.output_dir from --config.")
    parser.add_argument("--output-dir", type=str, default="outputs/ldm_metrics", help="Where metric reports and reference images are written.")

    parser.add_argument("--reference-dir", type=str, default=None, help="Existing true reference image folder. If omitted, the script builds one.")
    parser.add_argument("--val-latent-dir", type=str, default=None, help="Latent shard folder. Defaults to data.val_latent_dir from --config.")
    parser.add_argument("--image-root", type=str, default=None, help="Optional root to resolve relative original image paths stored in latent shards.")

    parser.add_argument("--vae-config", type=str, default=None, help="Needed only if reference images must be decoded from latent tensors. Defaults to vae.config from --config.")
    parser.add_argument("--vae-checkpoint", type=str, default=None, help="Needed only if reference images must be decoded from latent tensors. Defaults to vae.checkpoint from --config.")
    parser.add_argument("--scaling-factor", type=float, default=None, help="VAE latent scaling factor. Defaults to vae.scaling_factor from --config, else 1.0.")

    parser.add_argument("--max-images", type=int, default=None, help="Maximum images per setting/reference. Defaults to evaluation.max_images or 500.")
    parser.add_argument("--eval-resolution", type=int, default=None, help="Resize both real/fake images to this size before metrics. Defaults to evaluation.resolution from --config.")
    parser.add_argument("--batch-size", type=int, default=32, help="Metric batch size.")
    parser.add_argument("--decode-batch-size", type=int, default=32, help="VAE decode batch size when building references from latents.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["fid", "kid", "is"],
        choices=["fid", "kid", "is", "clip"],
        help="Metrics to compute. Default: fid kid is. Add clip for CLIPScore if CLIPModel is cached locally.",
    )
    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Use only locally cached CLIP files for CLIPScore.")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow transformers to download CLIP files if --metrics includes clip.")

    parser.add_argument("--force-rebuild-reference", action="store_true", help="Delete and recreate output-dir/reference_images.")

    args = parser.parse_args()
    if args.allow_downloads:
        args.local_files_only = False
    return args


def main():
    args = parse_args()
    cfg: dict[str, Any] = {}

    if args.config is not None:
        cfg = load_yaml(args.config)

    generated_root = Path(args.generated_root or cfg.get("evaluation", {}).get("output_dir", ""))
    if not str(generated_root):
        raise ValueError("Provide --generated-root or set evaluation.output_dir in --config.")
    generated_root = generated_root.expanduser().resolve()

    output_dir = ensure_dir(Path(args.output_dir).expanduser())

    val_latent_dir = args.val_latent_dir or cfg.get("data", {}).get("val_latent_dir", None)
    if val_latent_dir is not None:
        val_latent_dir = Path(val_latent_dir).expanduser()

    vae_config = args.vae_config or cfg.get("vae", {}).get("config", None)
    vae_checkpoint = args.vae_checkpoint or cfg.get("vae", {}).get("checkpoint", None)
    vae_config = Path(vae_config).expanduser() if vae_config is not None else None
    vae_checkpoint = Path(vae_checkpoint).expanduser() if vae_checkpoint is not None else None

    scaling_factor = args.scaling_factor
    if scaling_factor is None:
        scaling_factor = float(cfg.get("vae", {}).get("scaling_factor", 1.0))

    if args.max_images is None:
        args.max_images = int(cfg.get("evaluation", {}).get("max_images", cfg.get("evaluation", {}).get("num_prompts", 500)))

    if args.eval_resolution is None:
        args.eval_resolution = int(cfg.get("evaluation", {}).get("resolution", 256))

    image_root = Path(args.image_root).expanduser() if args.image_root is not None else None
    device = torch.device(args.device)

    print("=============================================")
    print("LDM metrics evaluation")
    print("Generated root:", generated_root)
    print("Output dir:", output_dir)
    print("Metrics:", args.metrics)
    print("Max images:", args.max_images)
    print("Eval resolution:", args.eval_resolution)
    print("Device:", device)
    print("=============================================")

    # Reference images.
    if args.reference_dir is not None:
        reference_dir = Path(args.reference_dir).expanduser().resolve()
        real_paths = list_reference_images(reference_dir)[:args.max_images]
        ref_info = {
            "reference_dir": str(reference_dir),
            "source": "provided_reference_dir",
            "num_images": len(real_paths),
        }
    else:
        if val_latent_dir is None:
            raise ValueError("Provide --reference-dir, or provide --val-latent-dir / data.val_latent_dir so a reference set can be built.")
        reference_dir = output_dir / "reference_images"
        real_paths, ref_info = build_reference_from_latent_shards(
            latent_dir=Path(val_latent_dir),
            out_dir=reference_dir,
            max_images=args.max_images,
            resolution=args.eval_resolution,
            image_root=image_root,
            vae_config=vae_config,
            vae_checkpoint=vae_checkpoint,
            scaling_factor=float(scaling_factor),
            device=device,
            decode_batch_size=args.decode_batch_size,
            force=args.force_rebuild_reference,
        )

    if len(real_paths) == 0:
        raise RuntimeError("No reference images available.")

    with open(output_dir / "reference_info.json", "w", encoding="utf-8") as f:
        json.dump(ref_info, f, indent=2)

    print(f"Reference images: {len(real_paths)}")
    print("Reference source:", ref_info.get("source"))

    # Generated setting folders.
    setting_dirs = []
    for d in sorted(generated_root.iterdir()):
        if not d.is_dir():
            continue
        if list_sample_images(d):
            setting_dirs.append(d)

    if not setting_dirs:
        raise RuntimeError(f"No generated setting folders with sample images found under {generated_root}")

    print(f"Found {len(setting_dirs)} generated setting folders.")

    results: list[dict[str, Any]] = []

    for setting_dir in setting_dirs:
        print("\n---------------------------------------------")
        print("Evaluating:", setting_dir.name)
        result = evaluate_setting(setting_dir, real_paths, args, device)
        results.append(result)

        with open(output_dir / f"{setting_dir.name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(json.dumps(result, indent=2))

    write_summary(results, output_dir)

    notes = {
        "metrics": {
            "fid": "Frechet Inception Distance; lower is better.",
            "kid": "Kernel Inception Distance; lower is better.",
            "is": "Inception Score on generated images only; higher is usually better, but it is less meaningful for COCO-style caption alignment.",
            "clip": "Mean 100*cosine(image,text) using CLIP; higher indicates better text-image alignment.",
        },
        "important_note": (
            "For paper-style reporting, FID/KID should be computed against true dataset images. "
            "If the reference source is vae_decoded_latents_from_latent_shards, label the result as comparison against VAE reconstructions."
        ),
        "generated_root": str(generated_root),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "metric_notes.json", "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)

    print("\n=============================================")
    print("Done.")
    print("Summary JSON:", output_dir / "metrics_summary.json")
    print("Summary CSV:", output_dir / "metrics_summary.csv")
    print("Reference info:", output_dir / "reference_info.json")
    print("=============================================")


if __name__ == "__main__":
    main()
