from __future__ import annotations

import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TimerRecord:
    name: str
    total: float = 0.0
    count: int = 0

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


class Timer(ContextDecorator):
    """
    Usage:

        timer = Timer()

        with timer("encode"):
            z = vae.encode_to_latent(x)

        with timer("decode"):
            x_recon = vae.decode_from_latent(z)

        timer.print_summary()

    You can reuse the same operation name many times.
    It will accumulate total time and average time.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        records: Optional[dict[str, TimerRecord]] = None,
        enabled: bool = True,
        sync_cuda: bool = True,
        verbose: bool = False,
    ):
        self.name = name
        self.records = records if records is not None else {}
        self.enabled = enabled
        self.sync_cuda = sync_cuda
        self.verbose = verbose

        self._start_time: Optional[float] = None

    def __call__(self, name: str):
        """
        Allows:

            with timer("operation_name"):
                ...
        """
        return Timer(
            name=name,
            records=self.records,
            enabled=self.enabled,
            sync_cuda=self.sync_cuda,
            verbose=self.verbose,
        )

    def __enter__(self):
        if not self.enabled:
            return self

        self._sync()
        self._start_time = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False

        self._sync()
        elapsed = time.perf_counter() - self._start_time

        if self.name is None:
            self.name = "unnamed"

        if self.name not in self.records:
            self.records[self.name] = TimerRecord(name=self.name)

        record = self.records[self.name]
        record.total += elapsed
        record.count += 1

        if self.verbose:
            print(f"[timer] {self.name}: {elapsed:.4f}s")

        return False

    def reset(self) -> None:
        self.records.clear()

    def get(self, name: str) -> TimerRecord:
        return self.records[name]

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "total_sec": record.total,
                "count": record.count,
                "avg_sec": record.avg,
            }
            for name, record in self.records.items()
        }

    def print_summary(
        self,
        sort_by: str = "total",
        title: str = "Timing summary",
    ) -> None:
        """
        sort_by:
            "total" or "avg" or "count"
        """
        if not self.records:
            print(f"{title}: no records")
            return

        if sort_by == "total":
            key_fn = lambda item: item[1].total
        elif sort_by == "avg":
            key_fn = lambda item: item[1].avg
        elif sort_by == "count":
            key_fn = lambda item: item[1].count
        else:
            raise ValueError("sort_by must be 'total', 'avg', or 'count'.")

        items = sorted(
            self.records.items(),
            key=key_fn,
            reverse=True,
        )

        print("=" * 72)
        print(title)
        print("=" * 72)
        print(f"{'operation':30s} {'count':>8s} {'total(s)':>12s} {'avg(s)':>12s}")
        print("-" * 72)

        for name, record in items:
            print(
                f"{name:30s} "
                f"{record.count:8d} "
                f"{record.total:12.4f} "
                f"{record.avg:12.4f}"
            )

        print("=" * 72)

    def _sync(self) -> None:
        """
        CUDA operations are async by default.

        If sync_cuda=True, timing becomes accurate for GPU operations.
        """
        if not self.sync_cuda:
            return

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass