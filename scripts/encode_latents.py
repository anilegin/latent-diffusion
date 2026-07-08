from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.coco_captions import CocoCaptionsDataset
from src.data.image_transforms import build_image_transform
from src.network.autoencoder.vae import AutoencoderKL
from src.utils.config import load_config, resolve_path_key, save_yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="VAE config path, usually config_used.yaml.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Trained VAE checkpoint path.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train2017",
        choices=["train2017", "val2017"],
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where latent shards will be saved.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of image latents per saved shard.",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=-1,
        help="Use -1 for full split. Useful to debug with e.g. 1024.",
    )

    parser.add_argument(
        "--sample-posterior",
        action="store_true",
        help="Use posterior.sample() instead of posterior.mode().",
    )

    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=None,
        help="Override config model.scaling_factor.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Storage dtype for saved latents.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing latent directory.",
    )

    return parser.parse_args()


def build_model(cfg: dict) -> AutoencoderKL:
    m = cfg["model"]

    return AutoencoderKL(
        in_channels=int(m["in_channels"]),
        out_channels=int(m["out_channels"]),
        latent_channels=int(m["latent_channels"]),
        base_channels=int(m["base_channels"]),
        channel_multipliers=tuple(m["channel_multipliers"]),
        num_res_blocks=int(m["num_res_blocks"]),
        dropout=float(m.get("dropout", 0.0)),
        use_attention=bool(m.get("use_attention", True)),
        attention_heads=int(m.get("attention_heads", 1)),
        scaling_factor=float(m.get("scaling_factor", 1.0)),
    )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def get_autocast_context(
    device: torch.device,
    precision: str,
):
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False)

    if precision == "bf16":
        return torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=True,
        )

    if precision == "fp16":
        return torch.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=True,
        )

    return torch.autocast("cuda", enabled=False)


def convert_latent_dtype(
    z: torch.Tensor,
    dtype: str,
) -> torch.Tensor:
    if dtype == "float32":
        return z.float()

    if dtype == "float16":
        return z.half()

    if dtype == "bfloat16":
        return z.bfloat16()

    raise ValueError(f"Unknown dtype: {dtype}")


def get_all_captions_from_dataset(
    dataset,
    sample: dict,
) -> list[str]:
    """
    Return all captions for this image
    """
    image_id = int(sample["image_id"])

    if hasattr(dataset, "get_captions"):
        captions = dataset.get_captions(image_id)
        return list(captions)

    if hasattr(dataset, "image_id_to_captions"):
        return list(dataset.image_id_to_captions[image_id])

    if hasattr(dataset, "captions_by_image_id"):
        return list(dataset.captions_by_image_id[image_id])

    # If dataset is a torch Subset, unwrap original dataset.
    if hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset

        if hasattr(base_dataset, "get_captions"):
            captions = base_dataset.get_captions(image_id)
            return list(captions)

        if hasattr(base_dataset, "image_id_to_captions"):
            return list(base_dataset.image_id_to_captions[image_id])

        if hasattr(base_dataset, "captions_by_image_id"):
            return list(base_dataset.captions_by_image_id[image_id])

    caption = sample.get("caption", "")
    return [str(caption)]


def collate_coco_batch(batch: list[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch], dim=0)

    image_ids = [int(item["image_id"]) for item in batch]
    file_names = [str(item["file_name"]) for item in batch]

    # all_captions is list[list[str]]
    all_captions = [list(item["all_captions"]) for item in batch]

    return {
        "image": images,
        "image_id": image_ids,
        "file_name": file_names,
        "all_captions": all_captions,
    }


class CocoCaptionsWithAllCaptions(torch.utils.data.Dataset):

    def __init__(self, dataset: CocoCaptionsDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        all_captions = get_all_captions_from_dataset(
            self.dataset,
            sample,
        )

        sample["all_captions"] = all_captions
        return sample


def save_shard(
    shard_path: Path,
    latents: list[torch.Tensor],
    captions: list[list[str]],
    image_ids: list[int],
    file_names: list[str],
    dtype: str,
) -> None:
    latent_tensor = torch.cat(latents, dim=0)
    latent_tensor = convert_latent_dtype(
        latent_tensor,
        dtype=dtype,
    )

    payload = {
        "latents": latent_tensor.cpu(),
        "captions": captions,
        "image_ids": image_ids,
        "file_names": file_names,
    }

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, shard_path)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(cfg.get("train", {}).get("precision", "bf16"))

    coco_root = resolve_path_key(cfg, cfg["dataset"]["root_key"])
    resolution = int(cfg["dataset"]["resolution"])

    if args.output_dir is None:
        latent_root = resolve_path_key(cfg, "outputs.latent_dir")
        output_dir = latent_root / f"coco_{args.split}_vae"
    else:
        output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}\n"
            "Use --overwrite if you really want to write into it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(cfg["dataset"].get("num_workers", 8))
    )

    scaling_factor = (
        args.scaling_factor
        if args.scaling_factor is not None
        else float(cfg["model"].get("scaling_factor", 1.0))
    )

    transform = build_image_transform(resolution)

    base_dataset = CocoCaptionsDataset(
        root=coco_root,
        split=args.split,
        transform=transform,
    )

    dataset = CocoCaptionsWithAllCaptions(base_dataset)

    if args.num_images > 0:
        indices = list(range(min(args.num_images, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg["dataset"].get("pin_memory", True)),
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=collate_coco_batch,
    )

    model = build_model(cfg)
    model = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    print("=============================================")
    print("Encoding COCO latents")
    print("Config:", args.config)
    print("Checkpoint:", args.checkpoint)
    print("Split:", args.split)
    print("Output dir:", output_dir)
    print("Images:", len(dataset))
    print("Batch size:", args.batch_size)
    print("Shard size:", args.shard_size)
    print("Sample posterior:", args.sample_posterior)
    print("Scaling factor:", scaling_factor)
    print("Storage dtype:", args.dtype)
    print("Device:", device)
    print("Precision:", precision)
    print("Captions: all captions per image")
    print("=============================================")

    metadata = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_images": len(dataset),
        "resolution": resolution,
        "sample_posterior": bool(args.sample_posterior),
        "scaling_factor": float(scaling_factor),
        "latent_channels": int(cfg["model"]["latent_channels"]),
        "storage_dtype": args.dtype,
        "shard_size": int(args.shard_size),
        "caption_mode": "all_captions_per_image",
        "shards": [],
    }

    current_latents: list[torch.Tensor] = []
    current_captions: list[list[str]] = []
    current_image_ids: list[int] = []
    current_file_names: list[str] = []

    total_saved = 0
    shard_idx = 0

    progress = tqdm(loader, desc=f"Encoding {args.split}")

    for batch in progress:
        x = batch["image"].to(
            device,
            non_blocking=True,
        )

        with get_autocast_context(device, precision):
            posterior = model.encode(x)

            if args.sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()

            z = z * scaling_factor

        z = z.detach().cpu()

        batch_size = z.shape[0]

        for i in range(batch_size):
            current_latents.append(z[i : i + 1])
            current_captions.append(list(batch["all_captions"][i]))
            current_image_ids.append(int(batch["image_id"][i]))
            current_file_names.append(str(batch["file_name"][i]))

            if len(current_latents) >= args.shard_size:
                shard_name = f"latents_{args.split}_{shard_idx:05d}.pt"
                shard_path = output_dir / shard_name

                save_shard(
                    shard_path=shard_path,
                    latents=current_latents,
                    captions=current_captions,
                    image_ids=current_image_ids,
                    file_names=current_file_names,
                    dtype=args.dtype,
                )

                metadata["shards"].append(
                    {
                        "file": shard_name,
                        "num_samples": len(current_latents),
                    }
                )

                total_saved += len(current_latents)
                shard_idx += 1

                current_latents = []
                current_captions = []
                current_image_ids = []
                current_file_names = []

                progress.set_postfix(
                    {
                        "saved": total_saved,
                        "shards": shard_idx,
                    }
                )

    if len(current_latents) > 0:
        shard_name = f"latents_{args.split}_{shard_idx:05d}.pt"
        shard_path = output_dir / shard_name

        save_shard(
            shard_path=shard_path,
            latents=current_latents,
            captions=current_captions,
            image_ids=current_image_ids,
            file_names=current_file_names,
            dtype=args.dtype,
        )

        metadata["shards"].append(
            {
                "file": shard_name,
                "num_samples": len(current_latents),
            }
        )

        total_saved += len(current_latents)
        shard_idx += 1

    metadata["num_saved"] = total_saved
    metadata["num_shards"] = shard_idx

    metadata_path = output_dir / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    save_yaml(cfg, output_dir / "config_used.yaml")

    print("=============================================")
    print("Latent encoding finished")
    print("Saved image latents:", total_saved)
    print("Saved shards:", shard_idx)
    print("Metadata:", metadata_path)
    print("Output dir:", output_dir)
    print("=============================================")


if __name__ == "__main__":
    main()