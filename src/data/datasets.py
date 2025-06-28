"""
BioViT3R-Beta Dataset Handling Module
Utility loaders for ACFR Orchard Fruit and MinneApple datasets.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import json
import pandas as pd
from PIL import Image

SUPPORTED_DATASETS = ["acfr", "minneapple"]

class DatasetLoader:
    """Generic dataset loader with ACFR and MinneApple specializations."""

    def __init__(self, dataset_root: str, dataset_type: str):
        if dataset_type not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        self.dataset_root = Path(dataset_root)
        self.dataset_type = dataset_type
        self.image_paths, self.annotations = self._load_dataset()

    def _load_dataset(self) -> Tuple[List[Path], List[Dict[str, Any]]]:
        if self.dataset_type == "acfr":
            return self._load_acfr()
        elif self.dataset_type == "minneapple":
            return self._load_minneapple()
        else:
            raise ValueError("Unsupported dataset type")

    def _load_acfr(self):
        images_dir = self.dataset_root / "acfr_orchard" / "images"
        ann_file = self.dataset_root / "acfr_orchard" / "annotations.json"

        with open(ann_file, "r") as f:
            annotations = json.load(f)

        image_paths = [images_dir / img["file_name"] for img in annotations["images"]]
        return image_paths, annotations["annotations"]

    def _load_minneapple(self):
        images_dir = self.dataset_root / "minneapple" / "train_images"
        ann_file = self.dataset_root / "minneapple" / "annotations.csv"
        df = pd.read_csv(ann_file)
        image_paths = [images_dir / p for p in df["filename"].unique()]
        annotations = df
        return image_paths, annotations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        ann = self.annotations[idx] if isinstance(self.annotations, list) else self.annotations[self.annotations["filename"] == self.image_paths[idx].name]
        return img, ann
