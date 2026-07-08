from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.elastic.multiprocessing.errors import record

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.latent_dataset import LatentShardDataset, latent_collate_fn
from src.diffusion.gaussian_diffusion import GaussianDiffusion
from src.network.conditioning.clip_text import FrozenCLIPTextEncoder
from src.network.conditioning.null_conditioning import ClassifierFreeGuidanceConditioner
from src.network.diffusion.unet import (
    build_latent_diffusion_unet_from_config,
    count_parameters,
)
from src.utils.config import load_config, resolve_path_key, save_yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/ldm_coco_256_vpred_strong_ddp.yaml",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--finetune-from",
        type=str,
        default=None,
    )

    return parser.parse_args()


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_dist_available_and_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if is_dist_available_and_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed():
    """
    Expects torchrun environment variables:

        RANK
        WORLD_SIZE
        LOCAL_RANK
    """
    if "RANK" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    return rank, world_size, local_rank


def cleanup_distributed():
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0) -> None:
    seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not is_dist_available_and_initialized():
        return value

    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value = value / get_world_size()
    return value


def unwrap_model(model):
    if isinstance(model, DDP):
        return model.module
    return model


def build_datasets(cfg: dict):
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

    return train_dataset, val_dataset


def build_dataloaders(cfg: dict, train_dataset, val_dataset, rank: int, world_size: int):
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 8))
    pin_memory = bool(cfg["train"].get("pin_memory", True))

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=latent_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=latent_collate_fn,
    )

    return train_loader, val_loader, train_sampler, val_sampler


def build_conditioner(cfg: dict):
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


def build_lr_scheduler(
    cfg: dict,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
):
    scheduler_name = str(cfg["train"].get("scheduler", "constant")).lower()
    warmup_steps = int(cfg["train"].get("warmup_steps", 0))
    min_lr = float(cfg["train"].get("min_lr", 0.0))
    base_lr = float(cfg["train"]["lr"])
    max_epochs = int(cfg["train"]["max_epochs"])

    total_steps = int(cfg["train"].get("max_train_steps", 0))
    if total_steps <= 0:
        total_steps = max(1, steps_per_epoch * max_epochs)

    if scheduler_name in {"none", "constant"} and warmup_steps <= 0:
        return None

    min_factor = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        if scheduler_name == "cosine":
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine_factor

        if scheduler_name in {"none", "constant"}:
            return 1.0

        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


def autocast_context(device: torch.device, precision: str):
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

    if precision == "fp32":
        return torch.autocast(device_type="cuda", enabled=False)

    raise ValueError(f"Unknown precision={precision}")


def save_checkpoint(
    output_dir: Path,
    name: str,
    model,
    conditioner,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_val_loss: float | None,
    metrics: dict[str, Any],
):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model": unwrap_model(model).state_dict(),
        "conditioner": conditioner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "metrics": metrics,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }

    path = checkpoint_dir / name
    torch.save(checkpoint, path)
    print_main(f"Saved checkpoint: {path}")


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def load_resume_checkpoint(
    checkpoint_path: str | Path,
    model,
    conditioner,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    unwrap_model(model).load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    if "conditioner" in checkpoint:
        conditioner.load_state_dict(
            checkpoint["conditioner"],
            strict=False,
        )

    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer_to_device(optimizer, device)

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(checkpoint["global_step"])
    best_val_loss = checkpoint.get("best_val_loss")

    print_main(f"Resumed from: {checkpoint_path}")
    print_main(f"Start epoch: {start_epoch}")
    print_main(f"Global step: {global_step}")

    return start_epoch, global_step, best_val_loss


def load_finetune_checkpoint(
    checkpoint_path: str | Path,
    model,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    unwrap_model(model).load_state_dict(
        state_dict,
        strict=True,
    )

    print_main(f"Loaded model weights for fine-tuning from: {checkpoint_path}")


def write_metrics(output_dir: Path, metrics: dict[str, Any]):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")


def train_one_epoch(
    epoch: int,
    model,
    conditioner,
    diffusion,
    train_loader,
    train_sampler,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    precision: str,
    grad_clip: float | None,
    gradient_accumulation_steps: int,
    global_step: int,
    log_every: int,
):
    model.train()
    conditioner.train()

    train_sampler.set_epoch(epoch)

    optimizer.zero_grad(set_to_none=True)

    loss_sum = 0.0
    num_batches = 0
    raw_loss_sum = 0.0

    iterator = train_loader
    use_tqdm = (
        is_main_process()
        and os.environ.get("DISABLE_TQDM", "0") != "1"
    )

    if use_tqdm:
        iterator = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}",
            leave=False,
            mininterval=10.0,
            dynamic_ncols=False,
        )

    for batch_idx, batch in enumerate(iterator):
        z0 = batch["latent"].to(
            device,
            non_blocking=True,
        )

        captions = batch["caption"]

        with autocast_context(device, precision):
            cond = conditioner(
                captions,
                device=device,
                apply_dropout=True,
            )

            out = diffusion.p_losses(
                model=model,
                z_0=z0,
                context=cond["context"],
                model_kwargs={
                    "attention_mask": cond["attention_mask"],
                },
            )

            loss = out.loss
            loss_for_backward = loss / gradient_accumulation_steps

        should_step = (
            (batch_idx + 1) % gradient_accumulation_steps == 0
            or (batch_idx + 1) == len(train_loader)
        )

        sync_context = (
            model.no_sync()
            if hasattr(model, "no_sync") and not should_step
            else nullcontext()
        )

        with sync_context:
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

        if should_step:
            if scaler.is_enabled():
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        unwrap_model(model).parameters(),
                        grad_clip,
                    )

                scaler.step(optimizer)
                scaler.update()

            else:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        unwrap_model(model).parameters(),
                        grad_clip,
                    )

                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        reduced_loss = reduce_mean(loss.detach())
        loss_sum += float(reduced_loss.cpu())
        reduced_raw_loss = reduce_mean(out.simple_loss.detach())
        raw_loss_sum += float(reduced_raw_loss.cpu())
        num_batches += 1

        if use_tqdm and batch_idx % log_every == 0:
            iterator.set_postfix(
                {
                    "loss": loss_sum / max(1, num_batches),
                    "step": global_step,
                }
            )
        elif is_main_process() and batch_idx % log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                f"loss={loss_sum / max(1, num_batches):.6f} "
                f"step={global_step} lr={current_lr:.8f}",
                flush=True,
            )

    return {
        "loss": loss_sum / max(1, num_batches),
        "raw_mse": raw_loss_sum / max(1, num_batches),
    }, global_step


@torch.no_grad()
def validate(
    epoch: int,
    model,
    conditioner,
    diffusion,
    val_loader,
    device: torch.device,
    precision: str,
    max_batches: int | None = None,
):
    model.eval()
    conditioner.eval()

    loss_sum = 0.0
    num_batches = 0

    iterator = val_loader
    use_tqdm = (
        is_main_process()
        and os.environ.get("DISABLE_TQDM", "0") != "1"
    )

    if use_tqdm:
        iterator = tqdm(
            val_loader,
            desc=f"Val epoch {epoch}",
            leave=False,
            mininterval=10.0,
            dynamic_ncols=False,
        )

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break

        z0 = batch["latent"].to(
            device,
            non_blocking=True,
        )

        captions = batch["caption"]

        with autocast_context(device, precision):
            cond = conditioner(
                captions,
                device=device,
                apply_dropout=False,
            )

            out = diffusion.p_losses(
                model=model,
                z_0=z0,
                context=cond["context"],
                model_kwargs={
                    "attention_mask": cond["attention_mask"],
                },
            )

            loss = out.loss

        reduced_loss = reduce_mean(loss.detach())
        loss_sum += float(reduced_loss.cpu())
        num_batches += 1

    return {
        "loss": loss_sum / max(1, num_batches),
    }

@torch.no_grad()
def validate_timestep_buckets(
    epoch: int,
    model,
    conditioner,
    diffusion,
    val_loader,
    device: torch.device,
    precision: str,
    timestep_buckets: list[list[int]] | None = None,
    max_batches: int | None = None,
):
    if not timestep_buckets:
        return {}

    model.eval()
    conditioner.eval()

    metrics: dict[str, float] = {}

    for bucket in timestep_buckets:
        low = int(bucket[0])
        high = int(bucket[1])

        loss_sum = 0.0
        num_batches = 0

        iterator = val_loader

        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break

            z0 = batch["latent"].to(
                device,
                non_blocking=True,
            )

            captions = batch["caption"]

            t = torch.randint(
                low=low,
                high=high,
                size=(z0.shape[0],),
                device=device,
                dtype=torch.long,
            )

            with autocast_context(device, precision):
                cond = conditioner(
                    captions,
                    device=device,
                    apply_dropout=False,
                )

                out = diffusion.p_losses(
                    model=model,
                    z_0=z0,
                    context=cond["context"],
                    t=t,
                    model_kwargs={
                        "attention_mask": cond["attention_mask"],
                    },
                )

                loss = out.loss

            reduced_loss = reduce_mean(loss.detach())
            loss_sum += float(reduced_loss.cpu())
            num_batches += 1

        key = f"loss_t_{low:03d}_{high:03d}"
        metrics[key] = loss_sum / max(1, num_batches)

    return metrics


@record
def main():
    args = parse_args()

    rank, world_size, local_rank = setup_distributed()

    try:
        cfg = load_config(args.config)

        if args.resume_from is not None:
            cfg["train"]["resume_from"] = args.resume_from

        if args.finetune_from is not None:
            cfg["train"]["finetune_from"] = args.finetune_from

        resume_from = cfg["train"].get("resume_from")
        finetune_from = cfg["train"].get("finetune_from")

        if resume_from is not None and finetune_from is not None:
            raise ValueError("Use either resume_from or finetune_from, not both.")

        set_seed(
            int(cfg["train"].get("seed", 42)),
            rank=rank,
        )

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        output_root = resolve_path_key(cfg, cfg["train"]["output_dir_key"])
        experiment_name = cfg["experiment"]["name"]
        output_dir = output_root / experiment_name

        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
            save_yaml(cfg, output_dir / "config_used.yaml")

        if is_dist_available_and_initialized():
            dist.barrier()

        train_dataset, val_dataset = build_datasets(cfg)

        train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(
            cfg=cfg,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            rank=rank,
            world_size=world_size,
        )

        model = build_latent_diffusion_unet_from_config(cfg)
        conditioner = build_conditioner(cfg)
        diffusion = build_diffusion(cfg).to(device)

        model.to(device)
        conditioner.to(device)

        optimizer = build_optimizer(cfg, model)

        precision = str(cfg["train"].get("precision", "bf16"))

        scaler = torch.cuda.amp.GradScaler(
            enabled=(precision == "fp16" and device.type == "cuda")
        )

        gradient_accumulation_steps = int(cfg["train"].get("gradient_accumulation_steps", 1))
        steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
        scheduler = build_lr_scheduler(
            cfg=cfg,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
        )

        if finetune_from is not None:
            load_finetune_checkpoint(
                checkpoint_path=finetune_from,
                model=model,
            )

        # Important: wrap only the trainable U-Net in DDP.
        # Conditioner is frozen, so we do not need DDP for it.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

        start_epoch = 0
        global_step = 0
        best_val_loss = None

        if resume_from is not None:
            start_epoch, global_step, best_val_loss = load_resume_checkpoint(
                checkpoint_path=resume_from,
                model=model,
                conditioner=conditioner,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
            )

        if is_main_process():
            print("=============================================")
            print("LDM DDP training")
            print("Experiment:", experiment_name)
            print("Config:", args.config)
            print("Output dir:", output_dir)
            print("World size:", world_size)
            print("Rank:", rank)
            print("Local rank:", local_rank)
            print("Device:", device)
            print("Resume from:", resume_from)
            print("Finetune from:", finetune_from)
            print("U-Net trainable params:", count_parameters(unwrap_model(model)) / 1e6, "M")
            print("Text context dim:", conditioner.context_dim)
            print("Diffusion prediction type:", diffusion.prediction_type)
            print("Diffusion timesteps:", diffusion.num_timesteps)
            print("Train batches per process:", len(train_loader))
            print("Val batches per process:", len(val_loader))
            print("Per-GPU batch size:", cfg["train"]["batch_size"])
            print("Grad accumulation:", cfg["train"].get("gradient_accumulation_steps", 1))
            print("Effective batch:", int(cfg["train"]["batch_size"]) * int(cfg["train"].get("gradient_accumulation_steps", 1)) * world_size)
            print("Scheduler:", cfg["train"].get("scheduler", "constant"))
            print("Warmup steps:", cfg["train"].get("warmup_steps", 0))
            print("Min LR:", cfg["train"].get("min_lr", 0.0))
            print("Steps per epoch:", steps_per_epoch)
            print("=============================================")

        max_epochs = int(cfg["train"]["max_epochs"])
        validate_every = int(cfg["train"].get("validate_every", 1))
        validation_enabled = bool(cfg.get("validation", {}).get("enabled", True))
        save_every = int(cfg["train"].get("save_every", 1))
        grad_clip = cfg["train"].get("grad_clip", 1.0)
        if grad_clip is not None:
            grad_clip = float(grad_clip)

        gradient_accumulation_steps = int(cfg["train"].get("gradient_accumulation_steps", 1))
        log_every = int(cfg["train"].get("log_every", 50))

        for epoch in range(start_epoch, max_epochs):
            train_metrics, global_step = train_one_epoch(
                epoch=epoch,
                model=model,
                conditioner=conditioner,
                diffusion=diffusion,
                train_loader=train_loader,
                train_sampler=train_sampler,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                precision=precision,
                grad_clip=grad_clip,
                gradient_accumulation_steps=gradient_accumulation_steps,
                global_step=global_step,
                log_every=log_every,
            )

            metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "world_size": world_size,
                **{f"train_{k}": v for k, v in train_metrics.items()},
            }

            if validation_enabled and (epoch + 1) % validate_every == 0:
                val_metrics = validate(
                    epoch=epoch,
                    model=model,
                    conditioner=conditioner,
                    diffusion=diffusion,
                    val_loader=val_loader,
                    device=device,
                    precision=precision,
                    max_batches=cfg.get("validation", {}).get("max_batches", None),
                )

                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                bucket_metrics = validate_timestep_buckets(
                    epoch=epoch,
                    model=model,
                    conditioner=conditioner,
                    diffusion=diffusion,
                    val_loader=val_loader,
                    device=device,
                    precision=precision,
                    timestep_buckets=cfg.get("validation", {}).get("timestep_buckets", None),
                    max_batches=cfg.get("validation", {}).get("bucket_max_batches", cfg.get("validation", {}).get("max_batches", None)),
                )

                metrics.update({f"val_{k}": v for k, v in bucket_metrics.items()})

                val_loss = val_metrics["loss"]

                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss

                    if is_main_process():
                        save_checkpoint(
                            output_dir=output_dir,
                            name="best.pt",
                            model=model,
                            conditioner=conditioner,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            epoch=epoch,
                            global_step=global_step,
                            best_val_loss=best_val_loss,
                            metrics=metrics,
                        )

            if is_main_process():
                metric_str = " ".join(
                    f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metrics.items()
                )
                print(f"[epoch summary] {metric_str}", flush=True)

                write_metrics(output_dir, metrics)

                if (epoch + 1) % save_every == 0:
                    save_checkpoint(
                        output_dir=output_dir,
                        name="last.pt",
                        model=model,
                        conditioner=conditioner,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        metrics=metrics,
                    )

            if is_dist_available_and_initialized():
                dist.barrier()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()