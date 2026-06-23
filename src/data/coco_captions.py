import json
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset


class CocoCaptionsDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train2017",
        transform=None,
        return_all_captions: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_all_captions = return_all_captions

        self.image_dir = self.root / "images" / split
        self.annotation_file = self.root / "annotations" / f"captions_{split}.json"

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.id_to_filename = {
            img["id"]: img["file_name"]
            for img in data["images"]
        }

        self.id_to_captions = defaultdict(list)
        for ann in data["annotations"]:
            self.id_to_captions[ann["image_id"]].append(ann["caption"])

        self.image_ids = sorted(list(self.id_to_filename.keys()))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        filename = self.id_to_filename[image_id]
        image_path = self.image_dir / filename

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        captions = self.id_to_captions[image_id]

        if self.return_all_captions:
            caption = captions
        else:
            caption = random.choice(captions)

        return {
            "image": image,
            "caption": caption,
            "image_id": image_id,
            "file_name": filename,
        }