from __future__ import annotations

import json
import time
from contextlib import ContextDecorator
from dataclasses import dataclass
from pathlib import Path
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
    It can be used as:
        timer = Timer(sync_cuda=True)

        with timer("sample"):
            images = sample()

        timer.print_summary()
        timer.save_json("timing.json")
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
        if self._start_time is None:
            elapsed = 0.0
        else:
            elapsed = time.perf_counter() - self._start_time

        name = self.name or "unnamed"
        if name not in self.records:
            self.records[name] = TimerRecord(name=name)

        record = self.records[name]
        record.total += elapsed
        record.count += 1

        if self.verbose:
            print(f"[timer] {name}: {elapsed:.4f}s", flush=True)

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

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)

    def print_summary(self, sort_by: str = "total", title: str = "Timing summary") -> None:
        if not self.records:
            print(f"{title}: no records", flush=True)
            return

        if sort_by == "total":
            key_fn = lambda item: item[1].total
        elif sort_by == "avg":
            key_fn = lambda item: item[1].avg
        elif sort_by == "count":
            key_fn = lambda item: item[1].count
        else:
            raise ValueError("sort_by must be 'total', 'avg', or 'count'.")

        items = sorted(self.records.items(), key=key_fn, reverse=True)

        print("=" * 72, flush=True)
        print(title, flush=True)
        print("=" * 72, flush=True)
        print(f"{'operation':36s} {'count':>8s} {'total(s)':>12s} {'avg(s)':>12s}", flush=True)
        print("-" * 72, flush=True)

        for name, record in items:
            print(
                f"{name:36s} {record.count:8d} {record.total:12.4f} {record.avg:12.4f}",
                flush=True,
            )

        print("=" * 72, flush=True)

    def _sync(self) -> None:
        if not self.sync_cuda:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
