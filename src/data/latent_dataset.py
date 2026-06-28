from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class LatentShardDataset(Dataset):
    """
    Dataset for cached VAE latents.

    Expected directory structure:

        outputs/latents/coco_train2017_vae8_scaled_allcaptions/
        ├── metadata.json
        ├── config_used_{date time}.yaml
        ├── latents_train2017_00000.pt
        ├── latents_train2017_00001.pt
        └── ...

    Each shard file contains:

        {
            "latents": Tensor[N, C, H, W],
            "captions": list[list[str]],
            "image_ids": list[int],
            "file_names": list[str],
        }
    """

    def __init__(
        self,
        root: str | Path,
        caption_mode: str = "random",
        load_to_memory: bool = False,
    ):
        self.root = Path(root)
        self.caption_mode = caption_mode
        self.load_to_memory = load_to_memory

        if caption_mode not in {"random", "first", "all"}:
            raise ValueError(
                f"Unknown caption_mode={caption_mode}. "
                "Use 'random', 'first', or 'all'."
            )

        self.metadata_path = self.root / "metadata.json"

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in latent directory: {self.root}"
            )

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.shard_infos = self.metadata["shards"]

        self.shard_paths = [
            self.root / shard_info["file"]
            for shard_info in self.shard_infos
        ]

        for path in self.shard_paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing latent shard: {path}")

        self.shard_sizes = [
            int(shard_info["num_samples"])
            for shard_info in self.shard_infos
        ]

        self.cumulative_sizes = []
        total = 0

        for size in self.shard_sizes:
            total += size
            self.cumulative_sizes.append(total)

        self.total_size = total

        self.cached_shards: dict[int, dict[str, Any]] = {}

        if self.load_to_memory:
            for shard_idx, shard_path in enumerate(self.shard_paths):
                self.cached_shards[shard_idx] = self._load_shard(shard_path)

    def __len__(self) -> int:
        return self.total_size

    def _load_shard(self, path: Path) -> dict[str, Any]:
        return torch.load(
            path,
            map_location="cpu",
        )

    def _find_shard(self, idx: int) -> tuple[int, int]:
        """
        Convert global dataset index into

            shard_idx
            local_idx_inside_shard
        """
        if idx < 0:
            idx = self.total_size + idx

        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range for dataset size {self.total_size}")

        # Binary search over cumulative sizes.
        left = 0
        right = len(self.cumulative_sizes) - 1

        while left <= right:
            mid = (left + right) // 2

            if idx < self.cumulative_sizes[mid]:
                right = mid - 1
            else:
                left = mid + 1

        shard_idx = left

        previous_cumulative = (
            0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
        )

        local_idx = idx - previous_cumulative

        return shard_idx, local_idx

    def _get_shard(self, shard_idx: int) -> dict[str, Any]:
        if shard_idx in self.cached_shards:
            return self.cached_shards[shard_idx]

        shard = self._load_shard(self.shard_paths[shard_idx])

        # Keep only one shard cached at a time when not loading everything.
        # This avoids constantly growing RAM usage.
        if not self.load_to_memory:
            self.cached_shards.clear()

        self.cached_shards[shard_idx] = shard

        return shard

    def choose_caption(self, captions: list[str]) -> str | list[str]:
        if len(captions) == 0:
            return ""

        if self.caption_mode == "random":
            return random.choice(captions)

        if self.caption_mode == "first":
            return captions[0]

        if self.caption_mode == "all":
            return captions

        raise RuntimeError("Invalid caption_mode.")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_idx, local_idx = self._find_shard(idx)
        shard = self._get_shard(shard_idx)

        latent = shard["latents"][local_idx].float()

        captions = shard["captions"][local_idx]

        # Backward compatibility: if older cache stores str instead of list[str].
        if isinstance(captions, str):
            captions = [captions]

        caption = self.choose_caption(captions)

        image_id = int(shard["image_ids"][local_idx])
        file_name = str(shard["file_names"][local_idx])

        return {
            "latent": latent,
            "caption": caption,
            "captions": captions,
            "image_id": image_id,
            "file_name": file_name,
        }


def latent_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for latent diffusion training.

    Keeps captions as list[str], because the CLIP tokenizer will process them.
    """
    latents = torch.stack(
        [item["latent"] for item in batch],
        dim=0,
    )

    captions = [item["caption"] for item in batch]
    all_captions = [item["captions"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    file_names = [item["file_name"] for item in batch]

    return {
        "latent": latents,
        "caption": captions,
        "captions": all_captions,
        "image_id": image_ids,
        "file_name": file_names,
    }