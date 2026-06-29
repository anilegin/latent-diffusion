from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.diffusion.gaussian_diffusion import GaussianDiffusion


class LDMTrainer:
    """
    Trainer for latent diffusion.

    Expected batch:

        {
            "latent": Tensor [B, C, H, W],
            "caption": list[str],
        }

    """

    def __init__(
        self,
        model: nn.Module,
        conditioner: nn.Module,
        diffusion: GaussianDiffusion,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device,
        output_dir: str | Path,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        precision: str = "bf16",
        grad_clip: float | None = 1.0,
        gradient_accumulation_steps: int = 1,
        max_epochs: int = 100,
        log_every: int = 100,
        validate_every: int = 1,
        save_every: int = 1,
    ):
        self.model = model
        self.conditioner = conditioner
        self.diffusion = diffusion

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = torch.device(device)
        self.output_dir = Path(output_dir)

        self.precision = precision
        self.grad_clip = grad_clip
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.max_epochs = max_epochs
        self.log_every = log_every
        self.validate_every = validate_every
        self.save_every = save_every

        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss: float | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.log_dir / "history.jsonl"

        self.model.to(self.device)
        self.conditioner.to(self.device)
        self.diffusion.to(self.device)

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.precision == "fp16" and self.device.type == "cuda")
        )

    def autocast_context(self):
        if self.device.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)

        if self.precision == "bf16":
            return torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            )

        if self.precision == "fp16":
            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            )

        if self.precision == "fp32":
            return torch.autocast(device_type="cuda", enabled=False)

        raise ValueError(
            f"Unknown precision={self.precision}. "
            "Use 'bf16', 'fp16', or 'fp32'."
        )

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.max_epochs):
            train_metrics = self.train_one_epoch(epoch)

            metrics = {
                "epoch": epoch,
                "global_step": self.global_step,
                **{f"train_{k}": v for k, v in train_metrics.items()},
            }

            if self.val_loader is not None and (epoch + 1) % self.validate_every == 0:
                val_metrics = self.validate(epoch)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                val_loss = val_metrics["loss"]

                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        epoch=epoch,
                        name="best.pt",
                        metrics=metrics,
                    )

            self.write_metrics(metrics)

            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    name="last.pt",
                    metrics=metrics,
                )

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.conditioner.train()

        loss_sum = 0.0
        num_batches = 0

        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            self.train_loader,
            desc=f"Train epoch {epoch}",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress):
            z0 = batch["latent"].to(
                self.device,
                non_blocking=True,
            )

            captions = batch["caption"]

            with self.autocast_context():
                cond = self.conditioner(
                    captions,
                    device=self.device,
                    apply_dropout=True,
                )

                out = self.diffusion.p_losses(
                    model=self.model,
                    z_0=z0,
                    context=cond["context"],
                    model_kwargs={
                        "attention_mask": cond["attention_mask"],
                    },
                )

                loss = out.loss
                loss_for_backward = loss / self.gradient_accumulation_steps

            should_step = (
                (batch_idx + 1) % self.gradient_accumulation_steps == 0
                or (batch_idx + 1) == len(self.train_loader)
            )

            if self.scaler.is_enabled():
                self.scaler.scale(loss_for_backward).backward()

                if should_step:
                    if self.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip,
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

            else:
                loss_for_backward.backward()

                if should_step:
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip,
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            if should_step:
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            loss_sum += float(loss.detach().cpu())
            num_batches += 1

            if batch_idx % self.log_every == 0:
                progress.set_postfix(
                    {
                        "loss": loss_sum / max(1, num_batches),
                        "step": self.global_step,
                    }
                )

        return {
            "loss": loss_sum / max(1, num_batches),
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        self.conditioner.eval()

        loss_sum = 0.0
        num_batches = 0

        progress = tqdm(
            self.val_loader,
            desc=f"Val epoch {epoch}",
            leave=False,
        )

        for batch in progress:
            z0 = batch["latent"].to(
                self.device,
                non_blocking=True,
            )

            captions = batch["caption"]

            with self.autocast_context():
                cond = self.conditioner(
                    captions,
                    device=self.device,
                    apply_dropout=False,
                )

                out = self.diffusion.p_losses(
                    model=self.model,
                    z_0=z0,
                    context=cond["context"],
                    model_kwargs={
                        "attention_mask": cond["attention_mask"],
                    },
                )

                loss = out.loss

            loss_sum += float(loss.detach().cpu())
            num_batches += 1

        return {
            "loss": loss_sum / max(1, num_batches),
        }

    def save_checkpoint(
        self,
        epoch: int,
        name: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "conditioner": self.conditioner.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "scaler": self.scaler.state_dict()
            if self.scaler is not None
            else None,
            "best_val_loss": self.best_val_loss,
            "metrics": metrics or {},
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }

        save_path = self.checkpoint_dir / name
        torch.save(checkpoint, save_path)

    def resume_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
    ) -> None:
        import gc

        checkpoint_path = Path(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        self.model.load_state_dict(
            checkpoint["model"],
            strict=strict,
        )

        if "conditioner" in checkpoint:
            self.conditioner.load_state_dict(
                checkpoint["conditioner"],
                strict=False,
            )

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._optimizer_to_device(self.optimizer, self.device)

        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if self.scaler is not None and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_val_loss = checkpoint.get("best_val_loss")

        if checkpoint.get("random_state") is not None:
            random.setstate(checkpoint["random_state"])

        if checkpoint.get("numpy_random_state") is not None:
            np.random.set_state(checkpoint["numpy_random_state"])

        if checkpoint.get("torch_random_state") is not None:
            torch_state = checkpoint["torch_random_state"]

            if not isinstance(torch_state, torch.Tensor):
                torch_state = torch.tensor(torch_state, dtype=torch.uint8)

            torch_state = torch_state.detach().cpu().to(dtype=torch.uint8)
            torch.random.set_rng_state(torch_state)

        if (
            torch.cuda.is_available()
            and checkpoint.get("cuda_random_state") is not None
        ):
            cuda_states = checkpoint["cuda_random_state"]

            fixed_cuda_states = []

            for state in cuda_states:
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.uint8)

                state = state.detach().cpu().to(dtype=torch.uint8)
                fixed_cuda_states.append(state)

            torch.cuda.set_rng_state_all(fixed_cuda_states)

        del checkpoint
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Resumed LDM training from: {checkpoint_path}")
        print(f"Starting epoch: {self.start_epoch}")
        print(f"Global step: {self.global_step}")

    def load_model_for_finetuning(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict,
            strict=strict,
        )

        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = None

        print(f"Loaded LDM model weights for fine-tuning from: {checkpoint_path}")

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    @staticmethod
    def _optimizer_to_device(
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)