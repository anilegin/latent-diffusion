from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        required=True,
        help="Path to VAE config, config_used.yaml.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to VAE checkpoint, e.g. best.pt or last.pt.",
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
        default=5000,
        help="Number of images to evaluate. Use 5000 for SD VAE comparison.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--sample-posterior",
        action="store_true",
        help="Use posterior sampling instead of posterior mean.",
    )

    parser.add_argument(
        "--no-fid",
        action="store_true",
        help="Disable FID/rFID computation.",
    )

    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Disable LPIPS computation.",
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


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device):
    checkpoint_path = Path(checkpoint_path)

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


def image_to_01(x: torch.Tensor) -> torch.Tensor:
    """
    Convert image from [-1, 1] to [0, 1].
    """
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def compute_psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    x, y expected in [0, 1].

    Returns per-image PSNR: [B]
    """
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    psnr = -10.0 * torch.log10(mse + eps)
    return psnr


def gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device,
    dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - window_size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    window_2d = g[:, None] * g[None, :]
    window_2d = window_2d.expand(channels, 1, window_size, window_size)

    return window_2d


def compute_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> torch.Tensor:
    """
    Simple SSIM implementation.

    x, y expected in [0, 1].

    Returns per-image SSIM: [B]
    """
    import torch.nn.functional as F

    b, c, h, w = x.shape

    window = gaussian_window(
        window_size=window_size,
        sigma=sigma,
        channels=c,
        device=x.device,
        dtype=x.dtype,
    )

    padding = window_size // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=c)
    mu_y = F.conv2d(y, window, padding=padding, groups=c)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=c) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=c) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=c) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / (denominator + 1e-8)

    return ssim_map.mean(dim=[1, 2, 3])


def get_autocast_context(device, precision: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)

    if precision == "bf16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        )

    if precision == "fp16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )

    return torch.autocast(device_type="cuda", enabled=False)


def build_lpips_model(device):
    import lpips

    model = lpips.LPIPS(net="vgg")
    model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    return model


def build_inception_model(device):
    """
    InceptionV3 feature extractor for FID-like metric.

    Uses torchvision InceptionV3 pretrained on ImageNet.
    Feature dimension: 2048.
    """
    from torchvision.models import inception_v3, Inception_V3_Weights

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(
        weights=weights,
        transform_input=False,
        aux_logits=True,
    )

    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    return model


@torch.no_grad()
def inception_features(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    x expected in [0, 1], shape [B, 3, H, W].

    Inception expects 299x299.
    """
    import torch.nn.functional as F

    x = F.interpolate(
        x,
        size=(299, 299),
        mode="bilinear",
        align_corners=False,
    )

    # ImageNet normalization.
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 3, 1, 1)

    x = (x - mean) / std

    feats = model(x)

    if isinstance(feats, tuple):
        feats = feats[0]

    return feats.float()


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    FID formula:

        ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 sigma2))
    """
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(
        sigma1.dot(sigma2),
        disp=False,
    )

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm(
            (sigma1 + offset).dot(sigma2 + offset)
        )

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = (
        diff.dot(diff)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * np.trace(covmean)
    )

    return float(fid)


def compute_fid_from_features(
    real_features: np.ndarray,
    recon_features: np.ndarray,
) -> float:
    mu_real = np.mean(real_features, axis=0)
    mu_recon = np.mean(recon_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_recon = np.cov(recon_features, rowvar=False)

    return calculate_frechet_distance(
        mu_real,
        sigma_real,
        mu_recon,
        sigma_recon,
    )


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

    if args.num_images > 0:
        dataset_size = min(args.num_images, len(dataset))
    else:
        dataset_size = len(dataset)

    subset = torch.utils.data.Subset(
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
    model = load_checkpoint(model, args.checkpoint, device)

    lpips_model = None
    if not args.no_lpips:
        lpips_model = build_lpips_model(device)

    inception_model = None
    if not args.no_fid:
        inception_model = build_inception_model(device)

    psnr_values = []
    ssim_values = []
    lpips_values = []

    real_features = []
    recon_features = []

    print("=============================================")
    print("VAE reconstruction evaluation")
    print("Config:", args.config)
    print("Checkpoint:", args.checkpoint)
    print("Split:", args.split)
    print("Images:", dataset_size)
    print("Batch size:", args.batch_size)
    print("Device:", device)
    print("Precision:", precision)
    print("FID enabled:", not args.no_fid)
    print("LPIPS enabled:", not args.no_lpips)
    print("=============================================")

    progress = tqdm(loader, desc="Evaluating")

    for batch in progress:
        x = batch["image"].to(
            device,
            non_blocking=True,
        )

        with get_autocast_context(device, precision):
            x_recon = model.reconstruct(
                x,
                sample_posterior=args.sample_posterior,
            )

        x_01 = image_to_01(x)
        x_recon_01 = image_to_01(x_recon)

        psnr = compute_psnr(x_01.float(), x_recon_01.float())
        ssim = compute_ssim(x_01.float(), x_recon_01.float())

        psnr_values.append(psnr.cpu())
        ssim_values.append(ssim.cpu())

        if lpips_model is not None:
            # LPIPS expects [-1, 1].
            with torch.cuda.amp.autocast(enabled=False):
                lp = lpips_model(
                    x_recon.float(),
                    x.float(),
                ).view(-1)

            lpips_values.append(lp.cpu())

        if inception_model is not None:
            real_feat = inception_features(
                inception_model,
                x_01.float(),
            )

            recon_feat = inception_features(
                inception_model,
                x_recon_01.float(),
            )

            real_features.append(real_feat.cpu().numpy())
            recon_features.append(recon_feat.cpu().numpy())

        current = {
            "psnr": torch.cat(psnr_values).mean().item(),
            "ssim": torch.cat(ssim_values).mean().item(),
        }

        if lpips_values:
            current["lpips"] = torch.cat(lpips_values).mean().item()

        progress.set_postfix(current)

    psnr_all = torch.cat(psnr_values)
    ssim_all = torch.cat(ssim_values)

    metrics = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_images": dataset_size,
        "sample_posterior": bool(args.sample_posterior),
        "psnr_mean": float(psnr_all.mean().item()),
        "psnr_std": float(psnr_all.std(unbiased=False).item()),
        "ssim_mean": float(ssim_all.mean().item()),
        "ssim_std": float(ssim_all.std(unbiased=False).item()),
    }

    if lpips_values:
        lpips_all = torch.cat(lpips_values)
        metrics["lpips_mean"] = float(lpips_all.mean().item())
        metrics["lpips_std"] = float(lpips_all.std(unbiased=False).item())

    if real_features and recon_features:
        real_features_np = np.concatenate(real_features, axis=0)
        recon_features_np = np.concatenate(recon_features, axis=0)

        rfid = compute_fid_from_features(
            real_features_np,
            recon_features_np,
        )

        metrics["rfid"] = float(rfid)

    print("=============================================")
    print("Results")
    print("=============================================")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("Saved:", output_path)


if __name__ == "__main__":
    main()