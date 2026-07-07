from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-id", type=str, default="Lykon/dreamshaper-8")
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--captions-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="Precision used for VAE encoding.",
    )

    parser.add_argument(
        "--use-posterior-sample",
        action="store_true",
        help="If set, use latent_dist.sample(); otherwise use latent_dist.mode().",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, allow writing into an existing output directory.",
    )

    return parser.parse_args()


def get_dtype(precision: str, device: torch.device):
    if device.type != "cuda":
        return torch.float32

    if precision == "fp16":
        return torch.float16

    if precision == "bf16":
        return torch.bfloat16

    return torch.float32


def load_coco_captions(captions_json_path: str):
    with open(captions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    captions_by_image_id = defaultdict(list)

    for ann in annotations:
        captions_by_image_id[ann["image_id"]].append(ann["caption"])

    image_records = []

    for img in images:
        image_id = img["id"]
        file_name = img["file_name"]
        caps = captions_by_image_id.get(image_id, [])

        if len(caps) == 0:
            continue

        image_records.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "captions": caps,
            }
        )

    return image_records


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ]
    )


def load_image(path: Path, transform):
    image = Image.open(path).convert("RGB")
    return transform(image)


def save_shard(
    shard_idx: int,
    output_dir: Path,
    latents: list[torch.Tensor],
    captions: list[list[str]],
    image_ids: list[int],
    file_names: list[str],
):
    shard_path = output_dir / f"latents_{shard_idx:05d}.pt"

    payload = {
        "latents": torch.stack(latents, dim=0).cpu(),
        "captions": captions,
        "image_ids": image_ids,
        "file_names": file_names,
    }

    torch.save(payload, shard_path)


def main():
    args = parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if output_dir.exists() and not args.overwrite:
        existing_pt_files = list(output_dir.glob("*.pt"))
        if len(existing_pt_files) > 0:
            raise FileExistsError(
                f"Output directory already contains .pt shards: {output_dir}. "
                "Use --overwrite or choose a new output directory."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vae_dtype = get_dtype(args.precision, device)

    print("============================================")
    print("Encoding COCO latents")
    print("Format: image-level latents with caption lists")
    print("Model ID:", args.model_id)
    print("Images dir:", images_dir)
    print("Captions JSON:", args.captions_json)
    print("Output dir:", output_dir)
    print("Batch size:", args.batch_size)
    print("Shard size:", args.shard_size)
    print("Device:", device)
    print("VAE dtype:", vae_dtype)
    print("Offline/local cache mode: True")
    print("============================================")

    print("Loading COCO captions...")
    records = load_coco_captions(args.captions_json)

    print("Number of source images:", len(records))

    print("Loading SD1.5 / DreamShaper VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=vae_dtype,
        local_files_only=True,
    ).to(device)

    vae.eval()

    latent_scaling_factor = float(vae.config.scaling_factor)
    print("VAE scaling factor:", latent_scaling_factor)

    transform = build_transform()

    shard_idx = 0
    shard_latents: list[torch.Tensor] = []
    shard_captions: list[list[str]] = []
    shard_image_ids: list[int] = []
    shard_file_names: list[str] = []

    num_missing_images = 0
    num_encoded_images = 0

    metadata = {
        "model_id": args.model_id,
        "images_dir": str(images_dir),
        "captions_json": args.captions_json,
        "latent_scaling_factor": latent_scaling_factor,
        "use_posterior_sample": bool(args.use_posterior_sample),
        "num_source_images": len(records),
        "image_resolution": 256,
        "vae_dtype": str(vae_dtype),
        "format": "image_level_latents_with_caption_lists",
    }

    print("Encoding latents...")

    for start in tqdm(range(0, len(records), args.batch_size), desc="Encoding"):
        batch_records = records[start : start + args.batch_size]

        images = []
        batch_caps = []
        batch_ids = []
        batch_names = []

        for rec in batch_records:
            img_path = images_dir / rec["file_name"]

            if not img_path.exists():
                print("Warning: missing image:", img_path)
                num_missing_images += 1
                continue

            img = load_image(img_path, transform)

            images.append(img)
            batch_caps.append(list(rec["captions"]))
            batch_ids.append(int(rec["image_id"]))
            batch_names.append(str(rec["file_name"]))

        if len(images) == 0:
            continue

        x = torch.stack(images, dim=0).to(
            device=device,
            dtype=vae_dtype,
        )

        with torch.no_grad():
            posterior = vae.encode(x).latent_dist

            if args.use_posterior_sample:
                z = posterior.sample()
            else:
                z = posterior.mode()

            z = z * latent_scaling_factor
            z = z.detach().cpu()

        for latent, captions, image_id, file_name in zip(
            z,
            batch_caps,
            batch_ids,
            batch_names,
        ):
            shard_latents.append(latent.clone())
            shard_captions.append(captions)
            shard_image_ids.append(image_id)
            shard_file_names.append(file_name)
            num_encoded_images += 1

            if len(shard_latents) >= args.shard_size:
                save_shard(
                    shard_idx=shard_idx,
                    output_dir=output_dir,
                    latents=shard_latents,
                    captions=shard_captions,
                    image_ids=shard_image_ids,
                    file_names=shard_file_names,
                )

                print(f"Saved shard {shard_idx:05d}")

                shard_idx += 1
                shard_latents = []
                shard_captions = []
                shard_image_ids = []
                shard_file_names = []

    if len(shard_latents) > 0:
        save_shard(
            shard_idx=shard_idx,
            output_dir=output_dir,
            latents=shard_latents,
            captions=shard_captions,
            image_ids=shard_image_ids,
            file_names=shard_file_names,
        )

        print(f"Saved shard {shard_idx:05d}")
        shard_idx += 1

    metadata["num_shards"] = shard_idx
    metadata["num_encoded_images"] = num_encoded_images
    metadata["num_missing_images"] = num_missing_images
    metadata["num_caption_pairs_available"] = sum(len(rec["captions"]) for rec in records)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")
    print("Saved to:", output_dir)
    print("Metadata:", output_dir / "metadata.json")
    print("Encoded images:", num_encoded_images)
    print("Missing images:", num_missing_images)
    print("Available caption pairs:", metadata["num_caption_pairs_available"])


if __name__ == "__main__":
    main()