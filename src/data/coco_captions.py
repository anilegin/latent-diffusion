import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CocoCaptionsDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform=None,
        caption_mode: str = "first",
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.caption_mode = caption_mode

        self.image_dir = self.root / "images" / split
        self.annotation_path = self.root / "annotations" / f"captions_{split}.json"

        with open(self.annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]

        self.image_id_to_captions: dict[int, list[str]] = {}

        for ann in self.annotations:
            image_id = int(ann["image_id"])
            caption = str(ann["caption"])

            if image_id not in self.image_id_to_captions:
                self.image_id_to_captions[image_id] = []

            self.image_id_to_captions[image_id].append(caption)

        # Keep only images that have at least one caption.
        self.images = [
            img for img in self.images
            if int(img["id"]) in self.image_id_to_captions
        ]

    def __len__(self):
        return len(self.images)

    def get_captions(self, image_id: int) -> list[str]:
        return self.image_id_to_captions[int(image_id)]

    def __getitem__(self, idx: int) -> dict:
        image_info = self.images[idx]

        image_id = int(image_info["id"])
        file_name = image_info["file_name"]

        image_path = self.image_dir / file_name

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        captions = self.get_captions(image_id)

        if self.caption_mode == "first":
            caption = captions[0]
        elif self.caption_mode == "random":
            caption = random.choice(captions)
        else:
            raise ValueError(
                f"Unknown caption_mode={self.caption_mode}. "
                "Use 'first' or 'random'."
            )

        return {
            "image": image,
            "caption": caption,
            "captions": captions,
            "image_id": image_id,
            "file_name": file_name,
        }