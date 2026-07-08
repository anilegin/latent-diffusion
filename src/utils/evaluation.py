from __future__ import annotations

import math
import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise RuntimeError(f"Empty YAML config: {path}")
    return data


def safe_torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        # Only use this fallback for checkpoints/shards you trust.
        return torch.load(path, map_location=map_location)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_sample_images(folder: Path) -> list[Path]:
    sample_paths = sorted(folder.glob("sample_*.png"))
    if sample_paths:
        return sample_paths

    paths = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.name.lower() not in {"grid.png"}:
            paths.append(p)
    return paths


def list_reference_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    if not folder.is_dir():
        raise NotADirectoryError(f"Reference path exists but is not a directory: {folder}")

    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def load_prompts(folder: Path, n: int) -> list[str] | None:
    path = folder / "prompts.json"
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    prompts: list[str] = []
    for item in payload:
        if isinstance(item, dict) and "prompt" in item:
            prompts.append(str(item["prompt"]))
        else:
            prompts.append(str(item))

    return prompts[:n]

def load_image_tensor_uint8(path: Path, resolution: int | None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if resolution is not None:
        img = img.resize((resolution, resolution), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def iter_image_batches(paths: list[Path], batch_size: int, resolution: int | None):
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start: start + batch_size]
        batch = torch.stack([load_image_tensor_uint8(p, resolution) for p in batch_paths], dim=0)
        yield batch, batch_paths


_INCEPTION_FEATURE_MODEL = None
_INCEPTION_CLASSIFIER_MODEL = None


def _prepare_inception_input_uint8(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    import torch.nn.functional as F

    x = batch.to(device=device, dtype=torch.float32) / 255.0
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _get_torchvision_inception_feature_model(device: torch.device) -> torch.nn.Module:
    global _INCEPTION_FEATURE_MODEL
    if _INCEPTION_FEATURE_MODEL is not None:
        return _INCEPTION_FEATURE_MODEL

    from torchvision.models import Inception_V3_Weights, inception_v3

    model = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
        aux_logits=True,
    )
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    _INCEPTION_FEATURE_MODEL = model
    return model


def _get_torchvision_inception_classifier_model(device: torch.device) -> torch.nn.Module:
    global _INCEPTION_CLASSIFIER_MODEL
    if _INCEPTION_CLASSIFIER_MODEL is not None:
        return _INCEPTION_CLASSIFIER_MODEL

    from torchvision.models import Inception_V3_Weights, inception_v3

    model = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
        aux_logits=True,
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    _INCEPTION_CLASSIFIER_MODEL = model
    return model


@torch.no_grad()
def _extract_inception_features_torchvision(
    paths: list[Path],
    batch_size: int,
    resolution: int | None,
    device: torch.device,
    desc: str,
) -> np.ndarray:
    model = _get_torchvision_inception_feature_model(device)
    feats = []

    for batch, _ in tqdm(
        iter_image_batches(paths, batch_size, resolution),
        total=math.ceil(len(paths) / batch_size),
        desc=desc,
        leave=False,
    ):
        x = _prepare_inception_input_uint8(batch, device)
        y = model(x)
        if isinstance(y, tuple):
            y = y[0]
        feats.append(y.detach().float().cpu().numpy())

    return np.concatenate(feats, axis=0)


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def compute_fid_from_features(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    return frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


def _compute_fid_torchvision_fallback(
    real_paths: list[Path],
    fake_paths: list[Path],
    batch_size: int,
    resolution: int | None,
    device: torch.device,
) -> float:
    real_features = _extract_inception_features_torchvision(real_paths, batch_size, resolution, device, "FID real fallback")
    fake_features = _extract_inception_features_torchvision(fake_paths, batch_size, resolution, device, "FID fake fallback")
    return compute_fid_from_features(real_features, fake_features)


def _polynomial_mmd2_unbiased(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    if n < 2 or m < 2:
        return torch.tensor(float("nan"), device=x.device)

    k_xx = ((x @ x.T) / d + 1.0) ** 3
    k_yy = ((y @ y.T) / d + 1.0) ** 3
    k_xy = ((x @ y.T) / d + 1.0) ** 3

    k_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
    k_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
    k_xy = k_xy.mean()
    return k_xx + k_yy - 2.0 * k_xy


def _compute_kid_torchvision_fallback(
    real_paths: list[Path],
    fake_paths: list[Path],
    batch_size: int,
    resolution: int | None,
    device: torch.device,
) -> tuple[float, float]:
    real_features_np = _extract_inception_features_torchvision(real_paths, batch_size, resolution, device, "KID real fallback")
    fake_features_np = _extract_inception_features_torchvision(fake_paths, batch_size, resolution, device, "KID fake fallback")

    n = min(real_features_np.shape[0], fake_features_np.shape[0])
    if n < 2:
        return float("nan"), float("nan")

    subset_size = min(100, n)
    num_subsets = min(100, max(1, n // subset_size))
    rng = np.random.default_rng(12345)

    real = torch.from_numpy(real_features_np).float()
    fake = torch.from_numpy(fake_features_np).float()
    values = []

    for _ in range(num_subsets):
        real_idx = rng.choice(real.shape[0], size=subset_size, replace=False)
        fake_idx = rng.choice(fake.shape[0], size=subset_size, replace=False)
        values.append(_polynomial_mmd2_unbiased(real[real_idx], fake[fake_idx]).item())

    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


@torch.no_grad()
def _compute_inception_score_torchvision_fallback(
    fake_paths: list[Path],
    batch_size: int,
    resolution: int | None,
    device: torch.device,
) -> tuple[float, float]:
    import torch.nn.functional as F

    model = _get_torchvision_inception_classifier_model(device)
    probs = []

    for batch, _ in tqdm(
        iter_image_batches(fake_paths, batch_size, resolution),
        total=math.ceil(len(fake_paths) / batch_size),
        desc="IS fake fallback",
        leave=False,
    ):
        x = _prepare_inception_input_uint8(batch, device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs.append(F.softmax(logits, dim=1).detach().float().cpu())

    p_yx = torch.cat(probs, dim=0)
    n = p_yx.shape[0]
    splits = min(10, max(1, n))
    split_scores = []

    for part in torch.chunk(p_yx, splits, dim=0):
        p_y = part.mean(dim=0, keepdim=True)
        kl = part * (torch.log(part.clamp_min(1e-12)) - torch.log(p_y.clamp_min(1e-12)))
        split_scores.append(torch.exp(kl.sum(dim=1).mean()).item())

    arr = np.asarray(split_scores, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def compute_fid(
    real_paths: list[Path],
    fake_paths: list[Path],
    batch_size: int,
    resolution: int | None,
    device: torch.device,
) -> float:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

        for batch, _ in tqdm(iter_image_batches(real_paths, batch_size, resolution), total=math.ceil(len(real_paths) / batch_size), desc="FID real", leave=False):
            metric.update(batch.to(device), real=True)

        for batch, _ in tqdm(iter_image_batches(fake_paths, batch_size, resolution), total=math.ceil(len(fake_paths) / batch_size), desc="FID fake", leave=False):
            metric.update(batch.to(device), real=False)

        return float(metric.compute().detach().cpu().item())
    except ModuleNotFoundError as exc:
        if "torch_fidelity" not in repr(exc) and "Torch-fidelity" not in repr(exc):
            raise
        print("torch-fidelity is not installed; using torchvision Inception fallback for FID.")
        return _compute_fid_torchvision_fallback(real_paths, fake_paths, batch_size, resolution, device)


def compute_kid(
    real_paths: list[Path],
    fake_paths: list[Path],
    batch_size: int,
    resolution: int | None,
    device: torch.device,
) -> tuple[float, float]:
    try:
        from torchmetrics.image.kid import KernelInceptionDistance

        n = min(len(real_paths), len(fake_paths))
        if n < 2:
            return float("nan"), float("nan")

        subset_size = min(100, n)
        metric = KernelInceptionDistance(subset_size=subset_size, normalize=False).to(device)

        for batch, _ in tqdm(iter_image_batches(real_paths, batch_size, resolution), total=math.ceil(len(real_paths) / batch_size), desc="KID real", leave=False):
            metric.update(batch.to(device), real=True)

        for batch, _ in tqdm(iter_image_batches(fake_paths, batch_size, resolution), total=math.ceil(len(fake_paths) / batch_size), desc="KID fake", leave=False):
            metric.update(batch.to(device), real=False)

        mean, std = metric.compute()
        return float(mean.detach().cpu().item()), float(std.detach().cpu().item())
    except ModuleNotFoundError as exc:
        if "torch_fidelity" not in repr(exc) and "Torch-fidelity" not in repr(exc):
            raise
        print("torch-fidelity is not installed; using torchvision Inception fallback for KID.")
        return _compute_kid_torchvision_fallback(real_paths, fake_paths, batch_size, resolution, device)


def compute_inception_score(
    fake_paths: list[Path],
    batch_size: int,
    resolution: int | None,
    device: torch.device,
) -> tuple[float, float]:
    try:
        from torchmetrics.image.inception import InceptionScore

        splits = min(10, max(1, len(fake_paths)))
        metric = InceptionScore(normalize=False, splits=splits).to(device)

        for batch, _ in tqdm(iter_image_batches(fake_paths, batch_size, resolution), total=math.ceil(len(fake_paths) / batch_size), desc="IS fake", leave=False):
            metric.update(batch.to(device))

        mean, std = metric.compute()
        return float(mean.detach().cpu().item()), float(std.detach().cpu().item())
    except ModuleNotFoundError as exc:
        if "torch_fidelity" not in repr(exc) and "Torch-fidelity" not in repr(exc):
            raise
        print("torch-fidelity is not installed; using torchvision Inception fallback for Inception Score.")
        return _compute_inception_score_torchvision_fallback(fake_paths, batch_size, resolution, device)


@torch.no_grad()
def compute_clip_score_local(
    fake_paths: list[Path],
    prompts: list[str],
    batch_size: int,
    device: torch.device,
    clip_model_name: str,
    local_files_only: bool,
) -> float:
    from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer

    if local_files_only:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model = CLIPModel.from_pretrained(clip_model_name, local_files_only=local_files_only).to(device)
    model.eval()

    processor = None
    tokenizer = None
    image_processor = None

    try:
        processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=local_files_only)
    except OSError:
        print("CLIPProcessor files are incomplete; falling back to manual CLIPImageProcessor + cached CLIPTokenizer.")
        tokenizer = CLIPTokenizer.from_pretrained(clip_model_name, local_files_only=local_files_only)
        image_size = int(getattr(model.config.vision_config, "image_size", 224))
        image_processor = CLIPImageProcessor(
            do_resize=True,
            size={"shortest_edge": image_size},
            do_center_crop=True,
            crop_size={"height": image_size, "width": image_size},
            do_rescale=True,
            rescale_factor=1.0 / 255.0,
            do_normalize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )

    scores: list[torch.Tensor] = []

    for start in tqdm(range(0, len(fake_paths), batch_size), desc="CLIPScore", leave=False):
        batch_paths = fake_paths[start: start + batch_size]
        batch_prompts = prompts[start: start + len(batch_paths)]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        if processor is not None:
            inputs = processor(
                text=batch_prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        else:
            image_inputs = image_processor(images=images, return_tensors="pt")
            text_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {**image_inputs, **text_inputs}

        inputs = {k: v.to(device) for k, v in inputs.items()}

        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        batch_scores = 100.0 * (image_features * text_features).sum(dim=-1)
        scores.append(batch_scores.detach().cpu())

    return float(torch.cat(scores).mean().item())
