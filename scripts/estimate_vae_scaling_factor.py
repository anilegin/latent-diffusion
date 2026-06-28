from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.coco_captions import CocoCaptionsDataset
from src.data.image_transforms import build_image_transform
from src.models.autoencoder.vae import AutoencoderKL
from src.utils.config import load_config, resolve_path_key


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="VAE config path, usually outputs/vae/.../config_used.yaml.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="VAE checkpoint path.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train2017",
        choices=["train2017", "val2017"],
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=10000,
        help="Number of images used to estimate latent std.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--sample-posterior",
        action="store_true",
        help="Use posterior.sample() instead of posterior.mode().",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
    )

    return parser.parse_args()


def build_model(cfg: dict) -> AutoencoderKL:
    m = cfg["model"]

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


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def main():
    args = parse_args()

    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(cfg.get("train", {}).get("precision", "bf16"))

    coco_root = resolve_path_key(cfg, cfg["dataset"]["root_key"])
    resolution = int(cfg["dataset"]["resolution"])

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(cfg["dataset"].get("num_workers", 8))
    )

    transform = build_image_transform(resolution)

    dataset = CocoCaptionsDataset(
        root=coco_root,
        split=args.split,
        transform=transform,
    )

    dataset_size = min(args.num_images, len(dataset))

    subset = Subset(
        dataset,
        list(range(dataset_size)),
    )

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg["dataset"].get("pin_memory", True)),
        drop_last=False,
    )

    model = build_model(cfg)
    model = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    if device.type == "cuda" and precision == "bf16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    elif device.type == "cuda" and precision == "fp16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
    elif device.type == "cuda":
        autocast_ctx = torch.autocast("cuda", enabled=False)
    else:
        autocast_ctx = torch.autocast("cpu", enabled=False)

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    channel_sum = None
    channel_sq_sum = None
    channel_count = 0

    print("=============================================")
    print("Estimating VAE latent scaling factor")
    print("Config:", args.config)
    print("Checkpoint:", args.checkpoint)
    print("Split:", args.split)
    print("Images:", dataset_size)
    print("Batch size:", args.batch_size)
    print("Use posterior sample:", args.sample_posterior)
    print("Device:", device)
    print("Precision:", precision)
    print("=============================================")

    for batch in tqdm(loader, desc="Encoding"):
        x = batch["image"].to(device, non_blocking=True)

        with autocast_ctx:
            posterior = model.encode(x)

            if args.sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()

        z = z.float()

        total_sum += z.sum().item()
        total_sq_sum += (z ** 2).sum().item()
        total_count += z.numel()

        # Per-channel stats.
        # z shape: [B, C, H, W]
        z_channel_sum = z.sum(dim=[0, 2, 3]).detach().cpu()
        z_channel_sq_sum = (z ** 2).sum(dim=[0, 2, 3]).detach().cpu()

        if channel_sum is None:
            channel_sum = z_channel_sum
            channel_sq_sum = z_channel_sq_sum
        else:
            channel_sum += z_channel_sum
            channel_sq_sum += z_channel_sq_sum

        channel_count += z.shape[0] * z.shape[2] * z.shape[3]

    mean = total_sum / total_count
    variance = total_sq_sum / total_count - mean**2
    std = variance**0.5
    scaling_factor = 1.0 / std

    channel_mean = channel_sum / channel_count
    channel_var = channel_sq_sum / channel_count - channel_mean**2
    channel_std = torch.sqrt(channel_var)
    channel_scaling = 1.0 / channel_std

    results = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_images": dataset_size,
        "sample_posterior": bool(args.sample_posterior),
        "latent_global_mean": float(mean),
        "latent_global_std": float(std),
        "scaling_factor": float(scaling_factor),
        "latent_channel_mean": channel_mean.tolist(),
        "latent_channel_std": channel_std.tolist(),
        "latent_channel_scaling": channel_scaling.tolist(),
    }

    print("=============================================")
    print("Results")
    print("=============================================")
    print("latent_global_mean:", results["latent_global_mean"])
    print("latent_global_std:", results["latent_global_std"])
    print("scaling_factor:", results["scaling_factor"])
    print("latent_channel_std:", results["latent_channel_std"])
    print("=============================================")

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("Saved:", output_path)


if __name__ == "__main__":
    main()