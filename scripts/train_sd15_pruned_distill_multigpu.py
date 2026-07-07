from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from diffusers import DDPMScheduler, UNet2DConditionModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, get_cosine_schedule_with_warmup
from torch.distributed.elastic.multiprocessing.errors import record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--finetune-from", type=str, default=None)
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main_process():
    return get_rank() == 0


def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    return rank, world_size, local_rank


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_mean(x: torch.Tensor):
    if not is_dist():
        return x
    x = x.detach().clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= get_world_size()
    return x


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


class LatentShardDataset(Dataset):
    def __init__(self, root: str | Path, load_to_memory: bool = True):
        self.root = Path(root)
        self.shard_paths = sorted(self.root.glob("*.pt"))

        if len(self.shard_paths) == 0:
            raise RuntimeError(f"No .pt shards found in {self.root}")

        self.load_to_memory = load_to_memory
        self.shards = []
        self.lengths = []
        self.cumulative_sizes = []

        running = 0
        for shard_path in self.shard_paths:
            payload = torch.load(shard_path, map_location="cpu")
            n = len(payload["captions"])
            self.lengths.append(n)
            running += n
            self.cumulative_sizes.append(running)

            if self.load_to_memory:
                self.shards.append(payload)
            else:
                self.shards.append(None)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _get_shard(self, shard_idx: int):
        if self.shards[shard_idx] is None:
            return torch.load(self.shard_paths[shard_idx], map_location="cpu")
        return self.shards[shard_idx]

    def __getitem__(self, index: int):
        shard_idx = bisect.bisect_right(self.cumulative_sizes, index)
        shard_start = 0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
        local_idx = index - shard_start

        payload = self._get_shard(shard_idx)

        latent = payload["latents"][local_idx]
        captions = payload["captions"][local_idx]

        if isinstance(captions, list):
            caption = random.choice(captions)
        else:
            caption = str(captions)

        return {
            "latent": latent,
            "caption": caption,
        }

def latent_collate_fn(batch):
    latents = torch.stack([item["latent"] for item in batch], dim=0)
    captions = [item["caption"] for item in batch]
    return {
        "latent": latents,
        "caption": captions,
    }


def build_dataloaders(cfg, rank, world_size):
    train_dataset = LatentShardDataset(
        root=cfg["data"]["train_latent_dir"],
        load_to_memory=bool(cfg["data"].get("load_latents_to_memory", True)),
    )

    val_dataset = LatentShardDataset(
        root=cfg["data"]["val_latent_dir"],
        load_to_memory=bool(cfg["data"].get("load_latents_to_memory", True)),
    )

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

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 2))
    pin_memory = bool(cfg["train"].get("pin_memory", True))

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


def build_models(cfg, device):
    teacher_model_id = cfg["teacher"]["model_id"]
    student_unet_path = cfg["student"]["unet_path"]

    precision = str(cfg["train"].get("precision", "bf16"))

    if device.type == "cuda" and precision == "fp16":
        model_dtype = torch.float16
    elif device.type == "cuda" and precision == "bf16":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(
        teacher_model_id,
        subfolder="tokenizer",
        local_files_only=True,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        teacher_model_id,
        subfolder="text_encoder",
        torch_dtype=model_dtype,
        local_files_only=True,
    ).to(device)

    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    teacher_unet = UNet2DConditionModel.from_pretrained(
        teacher_model_id,
        subfolder="unet",
        torch_dtype=model_dtype,
        local_files_only=True,
    ).to(device)

    teacher_unet.eval()
    for p in teacher_unet.parameters():
        p.requires_grad = False

    student_unet = UNet2DConditionModel.from_pretrained(
        student_unet_path,
        torch_dtype=model_dtype,
        local_files_only=True,
    ).to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(
        teacher_model_id,
        subfolder="scheduler",
        local_files_only=True,
    )

    return tokenizer, text_encoder, teacher_unet, student_unet, noise_scheduler


def maybe_apply_cfg_dropout(captions, cond_drop_prob: float, empty_text: str = ""):
    if cond_drop_prob <= 0.0:
        return captions

    out = []
    for c in captions:
        if random.random() < cond_drop_prob:
            out.append(empty_text)
        else:
            out.append(c)
    return out


def encode_text(tokenizer, text_encoder, captions, device, max_length: int):
    tokenized = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    with torch.no_grad():
        hidden = text_encoder(**tokenized).last_hidden_state

    return hidden


def autocast_context(device, precision: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)

    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)

    if precision == "fp32":
        return torch.autocast(device_type="cuda", enabled=False)

    raise ValueError(f"Unknown precision {precision}")


def build_optimizer_and_scheduler(cfg, student_unet, train_loader):
    train_cfg = cfg["train"]

    optimizer = torch.optim.AdamW(
        student_unet.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    max_epochs = int(train_cfg["max_epochs"])
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_training_steps = steps_per_epoch * max_epochs
    warmup_steps = int(train_cfg.get("warmup_steps", 0))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    return optimizer, scheduler


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def save_checkpoint(
    output_dir: Path,
    name: str,
    student_unet,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    best_val_loss,
    metrics,
):
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "student_unet": unwrap_model(student_unet).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "metrics": metrics,
    }

    torch.save(checkpoint, ckpt_dir / name)

    export_dir = output_dir / f"{name.replace('.pt', '')}_student_unet"
    unwrap_model(student_unet).save_pretrained(export_dir)

    print_main(f"Saved checkpoint: {ckpt_dir / name}")
    print_main(f"Saved student UNet: {export_dir}")


def load_resume_checkpoint(path, student_unet, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location="cpu")

    unwrap_model(student_unet).load_state_dict(ckpt["student_unet"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    optimizer_to_device(optimizer, device)

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt["epoch"]) + 1
    global_step = int(ckpt["global_step"])
    best_val_loss = ckpt.get("best_val_loss", None)

    return start_epoch, global_step, best_val_loss


def load_finetune_checkpoint(path, student_unet):
    ckpt = torch.load(path, map_location="cpu")
    if "student_unet" in ckpt:
        state_dict = ckpt["student_unet"]
    else:
        state_dict = ckpt
    unwrap_model(student_unet).load_state_dict(state_dict, strict=True)


def write_metrics(output_dir: Path, metrics: dict):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")


def train_one_epoch(
    epoch,
    student_unet,
    teacher_unet,
    tokenizer,
    text_encoder,
    noise_scheduler,
    train_loader,
    train_sampler,
    optimizer,
    scheduler,
    scaler,
    device,
    cfg,
    global_step,
):
    student_unet.train()
    teacher_unet.eval()
    text_encoder.eval()

    train_sampler.set_epoch(epoch)

    grad_accum = int(cfg["train"].get("gradient_accumulation_steps", 1))
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    precision = str(cfg["train"].get("precision", "bf16"))
    log_every = int(cfg["train"].get("log_every", 50))
    max_length = int(cfg["conditioning"].get("max_length", 77))
    cond_drop_prob = float(cfg["conditioning"].get("cond_drop_prob", 0.1))
    empty_text = str(cfg["conditioning"].get("empty_text", ""))
    distill_weight = float(cfg["loss"].get("distill_weight", 1.0))
    diffusion_weight = float(cfg["loss"].get("diffusion_weight", 0.0))

    optimizer.zero_grad(set_to_none=True)

    total_loss_sum = 0.0
    distill_loss_sum = 0.0
    diffusion_loss_sum = 0.0
    num_batches = 0

    use_tqdm = is_main_process() and os.environ.get("DISABLE_TQDM", "0") != "1"
    iterator = train_loader
    if use_tqdm:
        iterator = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}",
            leave=False,
            mininterval=10.0,
            dynamic_ncols=False,
        )

    for batch_idx, batch in enumerate(iterator):
        latents = batch["latent"].to(device, non_blocking=True)
        captions = maybe_apply_cfg_dropout(
            batch["caption"],
            cond_drop_prob=cond_drop_prob,
            empty_text=empty_text,
        )

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = encode_text(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            captions=captions,
            device=device,
            max_length=max_length,
        )

        with torch.no_grad():
            with autocast_context(device, precision):
                teacher_pred = teacher_unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

        should_step = (
            (batch_idx + 1) % grad_accum == 0
            or (batch_idx + 1) == len(train_loader)
        )

        sync_context = (
            student_unet.no_sync()
            if hasattr(student_unet, "no_sync") and not should_step
            else nullcontext()
        )

        with sync_context:
            with autocast_context(device, precision):
                student_pred = student_unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                distill_loss = F.mse_loss(student_pred.float(), teacher_pred.float())
                diffusion_loss = F.mse_loss(student_pred.float(), noise.float())
                total_loss = distill_weight * distill_loss + diffusion_weight * diffusion_loss

                loss_for_backward = total_loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

        if should_step:
            if scaler.is_enabled():
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unwrap_model(student_unet).parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(unwrap_model(student_unet).parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        reduced_total = reduce_mean(total_loss.detach())
        reduced_distill = reduce_mean(distill_loss.detach())
        reduced_diff = reduce_mean(diffusion_loss.detach())

        total_loss_sum += float(reduced_total.cpu())
        distill_loss_sum += float(reduced_distill.cpu())
        diffusion_loss_sum += float(reduced_diff.cpu())
        num_batches += 1

        if use_tqdm and batch_idx % log_every == 0:
            iterator.set_postfix(
                {
                    "loss": total_loss_sum / max(1, num_batches),
                    "distill": distill_loss_sum / max(1, num_batches),
                    "eps": diffusion_loss_sum / max(1, num_batches),
                    "step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
        elif is_main_process() and batch_idx % log_every == 0:
            print(
                f"epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                f"loss={total_loss_sum / max(1, num_batches):.6f} "
                f"distill={distill_loss_sum / max(1, num_batches):.6f} "
                f"eps={diffusion_loss_sum / max(1, num_batches):.6f} "
                f"step={global_step} lr={optimizer.param_groups[0]['lr']:.8f}",
                flush=True,
            )

    return {
        "loss": total_loss_sum / max(1, num_batches),
        "distill_loss": distill_loss_sum / max(1, num_batches),
        "diffusion_loss": diffusion_loss_sum / max(1, num_batches),
    }, global_step


@torch.no_grad()
def validate(
    epoch,
    student_unet,
    teacher_unet,
    tokenizer,
    text_encoder,
    noise_scheduler,
    val_loader,
    device,
    cfg,
):
    student_unet.eval()
    teacher_unet.eval()
    text_encoder.eval()

    precision = str(cfg["train"].get("precision", "bf16"))
    max_length = int(cfg["conditioning"].get("max_length", 77))
    distill_weight = float(cfg["loss"].get("distill_weight", 1.0))
    diffusion_weight = float(cfg["loss"].get("diffusion_weight", 0.0))
    max_batches = cfg.get("validation", {}).get("max_batches", None)

    total_loss_sum = 0.0
    distill_loss_sum = 0.0
    diffusion_loss_sum = 0.0
    num_batches = 0

    use_tqdm = is_main_process() and os.environ.get("DISABLE_TQDM", "0") != "1"
    iterator = val_loader
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

        latents = batch["latent"].to(device, non_blocking=True)
        captions = batch["caption"]

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = encode_text(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            captions=captions,
            device=device,
            max_length=max_length,
        )

        with autocast_context(device, precision):
            teacher_pred = teacher_unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            student_pred = student_unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            distill_loss = F.mse_loss(student_pred.float(), teacher_pred.float())
            diffusion_loss = F.mse_loss(student_pred.float(), noise.float())
            total_loss = distill_weight * distill_loss + diffusion_weight * diffusion_loss

        reduced_total = reduce_mean(total_loss.detach())
        reduced_distill = reduce_mean(distill_loss.detach())
        reduced_diff = reduce_mean(diffusion_loss.detach())

        total_loss_sum += float(reduced_total.cpu())
        distill_loss_sum += float(reduced_distill.cpu())
        diffusion_loss_sum += float(reduced_diff.cpu())
        num_batches += 1

        if use_tqdm:
            iterator.set_postfix(
                {
                    "val_loss": total_loss_sum / max(1, num_batches),
                    "val_distill": distill_loss_sum / max(1, num_batches),
                    "val_eps": diffusion_loss_sum / max(1, num_batches),
                }
            )

    return {
        "loss": total_loss_sum / max(1, num_batches),
        "distill_loss": distill_loss_sum / max(1, num_batches),
        "diffusion_loss": diffusion_loss_sum / max(1, num_batches),
    }


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

        resume_from = cfg["train"].get("resume_from", None)
        finetune_from = cfg["train"].get("finetune_from", None)

        if resume_from is not None and finetune_from is not None:
            raise ValueError("Use either resume_from or finetune_from, not both.")

        set_seed(int(cfg["train"].get("seed", 42)), rank=rank)

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        output_dir = Path(cfg["outputs"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        if is_main_process():
            with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

        if is_dist():
            dist.barrier()

        train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(cfg, rank, world_size)
        tokenizer, text_encoder, teacher_unet, student_unet, noise_scheduler = build_models(cfg, device)

        optimizer, scheduler = build_optimizer_and_scheduler(cfg, student_unet, train_loader)

        precision = str(cfg["train"].get("precision", "bf16"))
        scaler = torch.cuda.amp.GradScaler(
            enabled=(precision == "fp16" and device.type == "cuda")
        )

        if finetune_from is not None:
            load_finetune_checkpoint(finetune_from, student_unet)

        student_unet = DDP(
            student_unet,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

        start_epoch = 0
        global_step = 0
        best_val_loss = None

        if resume_from is not None:
            start_epoch, global_step, best_val_loss = load_resume_checkpoint(
                resume_from,
                student_unet,
                optimizer,
                scheduler,
                scaler,
                device,
            )

        print_main("============================================")
        print_main("DreamShaper pruned distillation training")
        print_main("Teacher:", cfg["teacher"]["model_id"])
        print_main("Student:", cfg["student"]["unet_path"])
        print_main("Output dir:", output_dir)
        print_main("World size:", world_size)
        print_main("Device:", device)
        print_main("Resume from:", resume_from)
        print_main("Finetune from:", finetune_from)
        print_main("Train batches per process:", len(train_loader))
        print_main("Val batches per process:", len(val_loader))
        print_main("============================================")

        max_epochs = int(cfg["train"]["max_epochs"])
        validate_every = int(cfg["train"].get("validate_every", 1))
        save_every = int(cfg["train"].get("save_every", 1))

        for epoch in range(start_epoch, max_epochs):
            train_metrics, global_step = train_one_epoch(
                epoch=epoch,
                student_unet=student_unet,
                teacher_unet=teacher_unet,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                noise_scheduler=noise_scheduler,
                train_loader=train_loader,
                train_sampler=train_sampler,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                cfg=cfg,
                global_step=global_step,
            )

            metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "world_size": world_size,
                **{f"train_{k}": v for k, v in train_metrics.items()},
            }

            if (epoch + 1) % validate_every == 0:
                val_metrics = validate(
                    epoch=epoch,
                    student_unet=student_unet,
                    teacher_unet=teacher_unet,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    noise_scheduler=noise_scheduler,
                    val_loader=val_loader,
                    device=device,
                    cfg=cfg,
                )

                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                val_loss = val_metrics["loss"]

                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if is_main_process():
                        save_checkpoint(
                            output_dir=output_dir,
                            name="best.pt",
                            student_unet=student_unet,
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
                        student_unet=student_unet,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        metrics=metrics,
                    )

            if is_dist():
                dist.barrier()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()