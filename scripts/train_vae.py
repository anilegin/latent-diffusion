from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.coco_captions import CocoCaptionsDataset
from src.data.image_transforms import build_image_transform
from src.losses.vae_loss import VAELoss
from src.models.autoencoder.vae import AutoencoderKL
from src.train.trainer_vae import VAETrainer
from src.utils.config import load_config, resolve_path_key, save_yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/vae_coco_256.yaml",
        help="Path to VAE experiment config.",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional checkpoint path for full resume. Overrides config.",
    )

    parser.add_argument(
        "--finetune-from",
        type=str,
        default=None,
        help="Optional checkpoint path for model-weight-only fine-tuning. Overrides config.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: dict):
    coco_root = resolve_path_key(cfg, cfg["dataset"]["root_key"])
    resolution = int(cfg["dataset"]["resolution"])

    train_split = cfg["dataset"]["train_split"]
    val_split = cfg["dataset"]["val_split"]

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["dataset"].get("num_workers", cfg["train"].get("num_workers", 8)))
    pin_memory = bool(cfg["dataset"].get("pin_memory", True))

    transform = build_image_transform(resolution)

    train_dataset = CocoCaptionsDataset(
        root=coco_root,
        split=train_split,
        transform=transform,
    )

    val_dataset = CocoCaptionsDataset(
        root=coco_root,
        split=val_split,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


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


def build_loss(cfg: dict) -> VAELoss:
    l = cfg["loss"]

    return VAELoss(
        recon_loss_type=str(l.get("recon_loss_type", "l1")),
        recon_weight=float(l.get("recon_weight", 1.0)),
        kl_weight=float(l.get("kl_weight", 1e-6)),
        perceptual_weight=float(l.get("perceptual_weight", 0.0)),
        use_lpips=bool(l.get("use_lpips", False)),
        lpips_net=str(l.get("lpips_net", "vgg")),
    )


def build_optimizer(cfg: dict, model: torch.nn.Module):
    name = str(cfg["optimizer"].get("name", "adamw")).lower()

    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    betas = tuple(float(x) for x in cfg["train"].get("betas", [0.9, 0.999]))

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {name}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.resume_from is not None:
        cfg["train"]["resume_from"] = args.resume_from

    if args.finetune_from is not None:
        cfg["train"]["finetune_from"] = args.finetune_from

    resume_from = cfg["train"].get("resume_from")
    finetune_from = cfg["train"].get("finetune_from")

    if resume_from is not None and finetune_from is not None:
        raise ValueError("Use either resume_from or finetune_from, not both.")

    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_root = resolve_path_key(cfg, cfg["train"]["output_dir_key"])
    experiment_name = cfg["experiment"]["name"]
    output_dir = output_root / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(cfg, output_dir / "config_used.yaml")

    print("Experiment:", experiment_name)
    print("Output dir:", output_dir)
    print("Device:", device)

    train_loader, val_loader = build_dataloaders(cfg)

    model = build_model(cfg)
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)

    # Important:
    # initialize only when training from scratch.
    initialize_from_scratch = bool(cfg["train"].get("initialize_from_scratch", True))
    if resume_from is not None or finetune_from is not None:
        initialize_from_scratch = False

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))

    trainer = VAETrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        output_dir=output_dir,
        precision=str(cfg["train"].get("precision", "bf16")),
        grad_clip=cfg["train"].get("grad_clip", 1.0),
        max_epochs=int(cfg["train"].get("max_epochs", 100)),
        log_every=int(cfg["train"].get("log_every", 100)),
        validate_every=int(cfg["train"].get("validate_every", 1)),
        save_every=int(cfg["train"].get("save_every", 1)),
        sample_every=int(cfg["train"].get("sample_every", 1)),
        num_sample_images=int(cfg["train"].get("num_sample_images", 8)),
        kl_weight=float(cfg["loss"].get("kl_weight", 1e-6)),
        kl_warmup_steps=int(cfg["loss"].get("kl_warmup_steps", 0)),
        early_stopping_patience=int(early_cfg.get("patience", 15)) if early_enabled else None,
        early_stopping_min_delta=float(early_cfg.get("min_delta", 0.0)),
        monitor_metric=str(early_cfg.get("monitor_metric", "val_total_loss")),
        initialize_from_scratch=initialize_from_scratch,
    )

    if resume_from is not None:
        trainer.resume_from_checkpoint(resume_from)

    elif finetune_from is not None:
        trainer.load_model_for_finetuning(finetune_from)

    trainer.train()


if __name__ == "__main__":
    main()