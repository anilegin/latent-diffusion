from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm


def initialize_vae_weights(model: nn.Module) -> None:
    """
    Good default initialization for a convolutional VAE

    Use only when starting training from scratch
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class EarlyStopping:

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best: float | None = None
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        """
        Returns True if training should stop
        """
        if self.best is None:
            self.best = value
            return False

        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta

        if improved:
            self.best = value
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return {
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best = state["best"]
        self.num_bad_epochs = state["num_bad_epochs"]
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]
        self.mode = state["mode"]


class VAETrainer:
    """
    Trainer for AutoencoderKL
        - bf16/fp16/fp32 training
        - checkpoint saving
        - resume full training state
        - fine-tune from model weights only
        - early stopping
        - KL warmup
        - validation reconstruction grids

    Expected batch format:
        batch["image"] -> [B, 3, H, W]
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device,
        output_dir: str | Path,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        precision: str = "bf16",
        grad_clip: float | None = 1.0,
        max_epochs: int = 100,
        log_every: int = 100,
        validate_every: int = 1,
        save_every: int = 1,
        sample_every: int = 1,
        num_sample_images: int = 8,
        kl_weight: float = 1e-6,
        kl_warmup_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        monitor_metric: str = "val_total_loss",
        initialize_from_scratch: bool = True,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = torch.device(device)
        self.output_dir = Path(output_dir)

        self.precision = precision
        self.grad_clip = grad_clip
        self.max_epochs = max_epochs
        self.log_every = log_every
        self.validate_every = validate_every
        self.save_every = save_every
        self.sample_every = sample_every
        self.num_sample_images = num_sample_images

        self.base_kl_weight = kl_weight
        self.kl_warmup_steps = kl_warmup_steps
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))

        self.monitor_metric = monitor_metric

        self.start_epoch = 0
        self.global_step = 0
        self.best_metric: float | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "reconstructions"
        self.log_dir = self.output_dir / "logs"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        if initialize_from_scratch:
            initialize_vae_weights(self.model)

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.precision == "fp16" and self.device.type == "cuda")
        )

        self.early_stopping = None
        if early_stopping_patience is not None and early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode="min",
            )

        self.history_path = self.log_dir / "history.jsonl"

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

    def current_kl_weight(self) -> float:
        """
        Linear KL warmup

        If kl_warmup_steps=0:
            immediately use base KL weight.
        """
        if self.kl_warmup_steps <= 0:
            return self.base_kl_weight

        progress = min(1.0, self.global_step / self.kl_warmup_steps)
        return self.base_kl_weight * progress

    def train(self) -> None:
        stop_training = False

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

                if (epoch + 1) % self.sample_every == 0:
                    self.save_reconstruction_grid(epoch)

                current_metric = metrics.get(self.monitor_metric)

                if current_metric is not None:
                    is_best = self.update_best_metric(current_metric)

                    if is_best:
                        self.save_checkpoint(
                            epoch=epoch,
                            name="best.pt",
                            metrics=metrics,
                        )

                    if self.early_stopping is not None:
                        stop_training = self.early_stopping.step(float(current_metric))

            self.write_metrics(metrics)

            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    name="last.pt",
                    metrics=metrics,
                )

            if stop_training:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best {self.monitor_metric}: {self.best_metric}"
                )
                break

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()

        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        perceptual_loss_sum = 0.0

        num_batches = 0

        progress = tqdm(
            self.train_loader,
            desc=f"Train epoch {epoch}",
            leave=False,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress):
            x = batch["image"].to(
                self.device,
                non_blocking=True,
            )

            kl_weight = self.current_kl_weight()

            with self.autocast_context():
                x_recon, posterior, _ = self.model(
                    x,
                    sample_posterior=True,
                )

                loss_out = self.loss_fn(
                    x_recon=x_recon,
                    x=x,
                    posterior=posterior,
                    kl_weight=kl_weight,
                )

                loss = loss_out.total_loss
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

            if should_step and self.scheduler is not None:
                self.scheduler.step()

            if should_step:
                self.global_step += 1

            num_batches += 1

            total_loss_sum += float(loss_out.total_loss.detach().cpu())
            recon_loss_sum += float(loss_out.recon_loss.cpu())
            kl_loss_sum += float(loss_out.kl_loss.cpu())
            perceptual_loss_sum += float(loss_out.perceptual_loss.cpu())

            if batch_idx % self.log_every == 0:
                progress.set_postfix(
                    {
                        "loss": total_loss_sum / num_batches,
                        "recon": recon_loss_sum / num_batches,
                        "kl": kl_loss_sum / num_batches,
                        "kl_w": kl_weight,
                        "perc": perceptual_loss_sum / num_batches,
                        "step": self.global_step,
                    }
                )

        return {
            "total_loss": total_loss_sum / max(1, num_batches),
            "recon_loss": recon_loss_sum / max(1, num_batches),
            "kl_loss": kl_loss_sum / max(1, num_batches),
            "perceptual_loss": perceptual_loss_sum / max(1, num_batches),
            "kl_weight": self.current_kl_weight(),
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()

        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        perceptual_loss_sum = 0.0

        num_batches = 0

        progress = tqdm(
            self.val_loader,
            desc=f"Val epoch {epoch}",
            leave=False,
        )

        for batch in progress:
            x = batch["image"].to(
                self.device,
                non_blocking=True,
            )

            with self.autocast_context():
                x_recon, posterior, _ = self.model(
                    x,
                    sample_posterior=False,
                )

                loss_out = self.loss_fn(
                    x_recon=x_recon,
                    x=x,
                    posterior=posterior,
                    kl_weight=self.base_kl_weight,
                )

            num_batches += 1

            total_loss_sum += float(loss_out.total_loss.detach().cpu())
            recon_loss_sum += float(loss_out.recon_loss.cpu())
            kl_loss_sum += float(loss_out.kl_loss.cpu())
            perceptual_loss_sum += float(loss_out.perceptual_loss.cpu())

        return {
            "total_loss": total_loss_sum / max(1, num_batches),
            "recon_loss": recon_loss_sum / max(1, num_batches),
            "kl_loss": kl_loss_sum / max(1, num_batches),
            "perceptual_loss": perceptual_loss_sum / max(1, num_batches),
        }

    @torch.no_grad()
    def save_reconstruction_grid(self, epoch: int) -> None:
        if self.val_loader is None:
            return

        self.model.eval()

        batch = next(iter(self.val_loader))
        x = batch["image"].to(self.device)

        x = x[: self.num_sample_images]

        with self.autocast_context():
            x_recon = self.model.reconstruct(
                x,
                sample_posterior=False,
            )

        # Stack as:
        # original_1, recon_1, original_2, recon_2, ...
        images = []

        for original, recon in zip(x, x_recon):
            images.append(original)
            images.append(recon)

        grid = torch.stack(images, dim=0)
        grid = ((grid + 1.0) / 2.0).clamp(0.0, 1.0)

        save_path = self.sample_dir / f"epoch_{epoch:04d}.png"

        save_image(
            grid,
            save_path,
            nrow=2,
        )

    def update_best_metric(self, metric: float) -> bool:
        metric = float(metric)

        if self.best_metric is None or metric < self.best_metric:
            self.best_metric = metric
            return True

        return False

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
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "scaler": self.scaler.state_dict()
            if self.scaler is not None
            else None,
            "best_metric": self.best_metric,
            "metrics": metrics or {},
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "early_stopping": self.early_stopping.state_dict()
            if self.early_stopping is not None
            else None,
        }

        save_path = self.checkpoint_dir / name
        torch.save(checkpoint, save_path)

    def resume_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
    ) -> None:
        """
        Resume full training state.

        Loads:
            - model
            - optimizer
            - scheduler
            - scaler
            - epoch/global step
            - RNG states
            - early stopping state
        """
        checkpoint_path = Path(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model"],
            strict=strict,
        )

        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if self.scaler is not None and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_metric = checkpoint.get("best_metric")

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

        if (
            self.early_stopping is not None
            and checkpoint.get("early_stopping") is not None
        ):
            self.early_stopping.load_state_dict(checkpoint["early_stopping"])

        print(f"Resumed training from: {checkpoint_path}")
        print(f"Starting at epoch: {self.start_epoch}")
        print(f"Global step: {self.global_step}")

    def load_model_for_finetuning(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
        reset_best_metric: bool = True,
    ) -> None:
        """
        Load model weights only.

        Does NOT load:
            - optimizer
            - scheduler
            - scaler
            - RNG state
            - epoch
        """
        checkpoint_path = Path(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
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

        if reset_best_metric:
            self.best_metric = None

        print(f"Loaded model weights for fine-tuning from: {checkpoint_path}")

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")