# python scripts/check_coco_dataset.py --config configs/data/coco_256.yaml --split train2017

# python scripts/check_coco_dataset.py --config configs/data/coco_256.yaml --split val2017


import argparse
import sys
from pathlib import Path

from torchvision.utils import save_image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.coco_captions import CocoCaptionsDataset
from src.data.image_transforms import build_image_transform
from src.utils.config import load_config, resolve_path_key


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/coco_256.yaml",
        help="Path to data config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train2017", "val2017"],
        help="Optional split override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    coco_root = resolve_path_key(cfg, cfg["dataset"]["root_key"])
    resolution = cfg["dataset"]["resolution"]

    split = args.split or cfg["dataset"]["train_split"]

    transform = build_image_transform(resolution)

    dataset = CocoCaptionsDataset(
        root=coco_root,
        split=split,
        transform=transform,
    )

    sample = dataset[0]
    image = sample["image"]

    print("COCO root:", coco_root)
    print("Split:", split)
    print("Dataset size:", len(dataset))
    print("Image shape:", tuple(image.shape))
    print("Image min/max:", image.min().item(), image.max().item())
    print("Caption:", sample["caption"])
    print("File:", sample["file_name"])

    output_dir = resolve_path_key(cfg, "outputs.sample_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"debug_coco_{split}.png"
    save_image((image + 1.0) / 2.0, save_path)

    print("Saved:", save_path)


if __name__ == "__main__":
    main()