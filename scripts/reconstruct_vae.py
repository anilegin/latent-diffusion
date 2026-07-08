from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.coco_captions import CocoCaptionsDataset
from src.data.image_transforms import build_image_transform
from src.network.autoencoder.vae import AutoencoderKL
from src.utils.config import load_config, resolve_path_key


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/vae_coco_256.yaml",
        help="Path to VAE experiment config.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to VAE checkpoint, e.g. outputs/vae/.../checkpoints/best.pt",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val2017",
        choices=["train2017", "val2017"],
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path.",
    )

    parser.add_argument(
        "--sample-posterior",
        action="store_true",
        help="Use posterior sampling instead of posterior mean.",
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
        upsample_type=str(m.get("upsample_type", "nearest_conv")),
        scaling_factor=float(m.get("scaling_factor", 1.0)),
    )

    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device):
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)

    return model


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    coco_root = resolve_path_key(cfg, cfg["dataset"]["root_key"])
    resolution = int(cfg["dataset"]["resolution"])

    transform = build_image_transform(resolution)

    dataset = CocoCaptionsDataset(
        root=coco_root,
        split=args.split,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.num_images,
        shuffle=False,
        num_workers=int(cfg["dataset"].get("num_workers", 8)),
        pin_memory=bool(cfg["dataset"].get("pin_memory", True)),
        drop_last=False,
    )

    model = build_model(cfg)
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    batch = next(iter(loader))
    x = batch["image"].to(device)
    x = x[: args.num_images]

    if device == "cuda":
        precision = str(cfg["train"].get("precision", "bf16"))

        if precision == "bf16":
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        elif precision == "fp16":
            autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
        else:
            autocast_ctx = torch.autocast("cuda", enabled=False)
    else:
        autocast_ctx = torch.autocast("cpu", enabled=False)

    with autocast_ctx:
        x_recon = model.reconstruct(
            x,
            sample_posterior=args.sample_posterior,
        )

    images = []

    for original, recon in zip(x, x_recon):
        images.append(original)
        images.append(recon)

    grid = torch.stack(images, dim=0)
    grid = ((grid + 1.0) / 2.0).clamp(0.0, 1.0)

    if args.output is None:
        output_root = resolve_path_key(cfg, "outputs.sample_dir")
        output_root.mkdir(parents=True, exist_ok=True)
        save_path = output_root / f"vae_reconstruction_{args.split}.png"
    else:
        save_path = Path(args.output)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(
        grid,
        save_path,
        nrow=2,
    )

    print("Saved reconstruction grid:", save_path)
    print("Format: original | reconstruction")
    print("Captions used:")
    for caption in batch["caption"][: args.num_images]:
        print("-", caption)


if __name__ == "__main__":
    main()