from __future__ import annotations

import json
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


class GPUMonitor:

    def __init__(
        self,
        enabled: bool = True,
        device: torch.device | str | None = None,
        sync_cuda: bool = True,
        sample_interval_s: float = 0.25,
        use_nvidia_smi: bool = True,
    ):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.device = torch.device(device or "cuda")
        self.sync_cuda = bool(sync_cuda)
        self.sample_interval_s = float(sample_interval_s)
        self.use_nvidia_smi = bool(use_nvidia_smi)
        self.records: dict[str, list[dict[str, Any]]] = {}

        if self.enabled and self.device.type == "cuda":
            self.device_index = self.device.index
            if self.device_index is None:
                self.device_index = torch.cuda.current_device()
        else:
            self.device_index = None

    def _sync(self):
        if self.enabled and self.sync_cuda:
            torch.cuda.synchronize(self.device)

    def _mem_stats_mb(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2,
            "max_reserved_mb": torch.cuda.max_memory_reserved(self.device) / 1024**2,
        }

    def _query_nvidia_smi(self) -> tuple[float | None, float | None]:
        if not (self.enabled and self.use_nvidia_smi and self.device_index is not None):
            return None, None
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.device_index}",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=1.0,
            ).strip()
            first_line = out.splitlines()[0]
            util_s, mem_s = [x.strip() for x in first_line.split(",")[:2]]
            return float(util_s), float(mem_s)
        except Exception:
            return None, None

    def _sampler(self, stop: threading.Event, samples: list[dict[str, float]]):
        while not stop.is_set():
            util, mem = self._query_nvidia_smi()
            if util is not None or mem is not None:
                samples.append({
                    "t": time.perf_counter(),
                    "gpu_util_percent": util,
                    "memory_used_mb": mem,
                })
            stop.wait(self.sample_interval_s)

    @contextmanager
    def track(self, name: str):
        if not self.enabled:
            yield
            return

        self._sync()
        torch.cuda.reset_peak_memory_stats(self.device)
        before = self._mem_stats_mb()
        start = time.perf_counter()

        samples: list[dict[str, float]] = []
        stop = threading.Event()
        thread = None
        if self.use_nvidia_smi:
            thread = threading.Thread(target=self._sampler, args=(stop, samples), daemon=True)
            thread.start()

        try:
            yield
        finally:
            self._sync()
            end = time.perf_counter()
            stop.set()
            if thread is not None:
                thread.join(timeout=1.0)
            after = self._mem_stats_mb()

            gpu_utils = [s["gpu_util_percent"] for s in samples if s.get("gpu_util_percent") is not None]
            smi_mem = [s["memory_used_mb"] for s in samples if s.get("memory_used_mb") is not None]

            record: dict[str, Any] = {
                "seconds": end - start,
                "torch_allocated_before_mb": before.get("allocated_mb"),
                "torch_allocated_after_mb": after.get("allocated_mb"),
                "torch_allocated_delta_mb": after.get("allocated_mb", 0.0) - before.get("allocated_mb", 0.0),
                "torch_reserved_before_mb": before.get("reserved_mb"),
                "torch_reserved_after_mb": after.get("reserved_mb"),
                "torch_reserved_delta_mb": after.get("reserved_mb", 0.0) - before.get("reserved_mb", 0.0),
                "torch_peak_allocated_mb": after.get("max_allocated_mb"),
                "torch_peak_reserved_mb": after.get("max_reserved_mb"),
                "nvidia_smi_num_samples": len(samples),
                "nvidia_smi_gpu_util_avg_percent": sum(gpu_utils) / len(gpu_utils) if gpu_utils else None,
                "nvidia_smi_gpu_util_max_percent": max(gpu_utils) if gpu_utils else None,
                "nvidia_smi_memory_used_avg_mb": sum(smi_mem) / len(smi_mem) if smi_mem else None,
                "nvidia_smi_memory_used_max_mb": max(smi_mem) if smi_mem else None,
            }
            self.records.setdefault(name, []).append(record)

    def summary(self) -> dict[str, Any]:
        return self.records

    def save_json(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)

    def print_summary(self, title: str = "GPU usage"):
        if not self.enabled:
            print(f"{title}: CUDA unavailable or GPU monitor disabled.")
            return
        print(f"\n{title}")
        for name, items in self.records.items():
            latest = items[-1]
            util = latest.get("nvidia_smi_gpu_util_avg_percent")
            util_s = "n/a" if util is None else f"{util:.1f}% avg"
            print(
                f"  {name}: "
                f"peak_alloc={latest['torch_peak_allocated_mb']:.1f} MB, "
                f"peak_reserved={latest['torch_peak_reserved_mb']:.1f} MB, "
                f"delta_alloc={latest['torch_allocated_delta_mb']:+.1f} MB, "
                f"gpu_util={util_s}"
            )
