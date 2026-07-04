from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.latent_dataset import LatentShardDataset, latent_collate_fn
from src.diffusion.gaussian_diffusion import GaussianDiffusion
from src.models.conditioning.clip_text import FrozenCLIPTextEncoder
from src.models.conditioning.null_conditioning import ClassifierFreeGuidanceConditioner
from src.models.diffusion.unet import (
    build_latent_diffusion_unet_from_config,
    count_parameters,
)
from src.train.trainer_ldm import LDMTrainer
from src.utils.config import load_config, resolve_path_key, save_yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/ldm_coco_256_vpred.yaml",
        help="Path to LDM experiment config.",
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
    train_dir = Path(cfg["data"]["train_latent_dir"]).expanduser()
    val_dir = Path(cfg["data"]["val_latent_dir"]).expanduser()

    if not train_dir.is_absolute():
        train_dir = resolve_path_key(cfg, "project.root") / train_dir

    if not val_dir.is_absolute():
        val_dir = resolve_path_key(cfg, "project.root") / val_dir

    train_dataset = LatentShardDataset(
        root=train_dir,
        caption_mode=str(cfg["data"].get("train_caption_mode", "random")),
        load_to_memory=bool(cfg["data"].get("load_latents_to_memory", False)),
    )

    val_dataset = LatentShardDataset(
        root=val_dir,
        caption_mode=str(cfg["data"].get("val_caption_mode", "first")),
        load_to_memory=bool(cfg["data"].get("load_latents_to_memory", False)),
    )

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 8))
    pin_memory = bool(cfg["train"].get("pin_memory", True))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=latent_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=latent_collate_fn,
    )

    return train_loader, val_loader


def build_conditioner(cfg: dict) -> ClassifierFreeGuidanceConditioner:
    text_cfg = cfg["text_encoder"]
    cond_cfg = cfg["conditioning"]

    text_encoder = FrozenCLIPTextEncoder(
        model_name=str(text_cfg.get("model_name", "openai/clip-vit-large-patch14")),
        max_length=int(text_cfg.get("max_length", 77)),
        freeze=bool(text_cfg.get("freeze", True)),
        use_last_hidden_state=bool(text_cfg.get("use_last_hidden_state", True)),
        cache_dir=text_cfg.get("cache_dir", None),
        local_files_only=bool(text_cfg.get("local_files_only", False)),
    )

    conditioner = ClassifierFreeGuidanceConditioner(
        text_encoder=text_encoder,
        cond_drop_prob=float(cond_cfg.get("cond_drop_prob", 0.1)),
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
        snr_gamma=d.get("snr_gamma", None),
        snr_weighting=str(d.get("snr_weighting", "none")),
        normalize_snr_weights=bool(d.get("normalize_snr_weights", False)),
    )

    return diffusion


def build_optimizer(cfg: dict, model: torch.nn.Module):
    name = str(cfg["optimizer"].get("name", "adamw")).lower()

    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.01))
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

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_yaml(cfg, output_dir / f"config_used_{date_str}.yaml")

    print("=============================================")
    print("LDM training")
    print("Experiment:", experiment_name)
    print("Config:", args.config)
    print("Output dir:", output_dir)
    print("Device:", device)
    print("Resume from:", resume_from)
    print("Finetune from:", finetune_from)
    print("=============================================")

    train_loader, val_loader = build_dataloaders(cfg)

    model = build_latent_diffusion_unet_from_config(cfg)
    conditioner = build_conditioner(cfg)
    diffusion = build_diffusion(cfg)
    optimizer = build_optimizer(cfg, model)

    print("U-Net trainable params:", count_parameters(model) / 1e6, "M")
    print("Text context dim:", conditioner.context_dim)
    print("Diffusion prediction type:", diffusion.prediction_type)
    print("Diffusion timesteps:", diffusion.num_timesteps)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Batch size:", cfg["train"]["batch_size"])
    print("Grad accumulation:", cfg["train"].get("gradient_accumulation_steps", 1))
    print("=============================================")

    trainer = LDMTrainer(
        model=model,
        conditioner=conditioner,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        output_dir=output_dir,
        precision=str(cfg["train"].get("precision", "bf16")),
        grad_clip=cfg["train"].get("grad_clip", 1.0),
        gradient_accumulation_steps=int(
            cfg["train"].get("gradient_accumulation_steps", 1)
        ),
        max_epochs=int(cfg["train"].get("max_epochs", 100)),
        log_every=int(cfg["train"].get("log_every", 100)),
        validate_every=int(cfg["train"].get("validate_every", 1)),
        save_every=int(cfg["train"].get("save_every", 1)),
    )

    if resume_from is not None:
        trainer.resume_from_checkpoint(resume_from)

    elif finetune_from is not None:
        trainer.load_model_for_finetuning(finetune_from)

    trainer.train()


if __name__ == "__main__":
    main()