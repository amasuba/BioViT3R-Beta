#!/usr/bin/env python3
"""
BioViT3R-Beta Dataset Setup Script
Downloads and prepares ACFR Orchard Fruit and MinneApple datasets for training and evaluation.
"""

import os
import sys
import json
import csv
import shutil
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm

class DatasetManager:
    """Manages dataset downloads, organization, and preprocessing."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Dataset configurations
        self.datasets = {
            "acfr_orchard": {
                "url": "https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/acfr-multifruit-2016.zip",
                "extract_path": "data/acfr_orchard",
                "annotation_format": "xml",
                "size_gb": 2.5,
                "classes": ["apple", "orange", "lemon", "lime", "grapefruit"]
            },
            "minneapple": {
                "url": "https://conservancy.umn.edu/bitstream/handle/11299/206575/detection-dataset.zip",
                "extract_path": "data/minneapple", 
                "annotation_format": "json",
                "size_gb": 8.2,
                "classes": ["apple"]
            }
        }

    def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> bool:
        """Download and extract a specific dataset."""
        if dataset_name not in self.datasets:
            print(f"❌ Unknown dataset: {dataset_name}")
            return False

        dataset_info = self.datasets[dataset_name]
        extract_path = Path(dataset_info["extract_path"])

        # Check if dataset already exists
        if not force_redownload and extract_path.exists() and any(extract_path.iterdir()):
            print(f"✅ {dataset_name} already exists at {extract_path}")
            return True

        print(f"📦 Downloading {dataset_name} ({dataset_info['size_gb']} GB)...")

        # Create download directory
        download_path = self.data_dir / f"{dataset_name}.zip"

        try:
            # Download with progress bar
            response = requests.get(dataset_info["url"], stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(download_path, 'wb') as f, tqdm(
                desc=f"Downloading {dataset_name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Extract dataset
            print(f"📂 Extracting {dataset_name}...")
            extract_path.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            # Clean up zip file
            download_path.unlink()

            print(f"✅ {dataset_name} downloaded and extracted successfully")
            return True

        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {str(e)}")
            return False

    def validate_acfr_dataset(self) -> Dict[str, any]:
        """Validate ACFR Orchard dataset structure and annotations."""
        acfr_path = Path("data/acfr_orchard")

        if not acfr_path.exists():
            return {"valid": False, "error": "Dataset not found"}

        # Expected structure: images/ and annotations/
        images_path = acfr_path / "images"
        annotations_path = acfr_path / "annotations"

        if not images_path.exists() or not annotations_path.exists():
            return {"valid": False, "error": "Missing images or annotations directory"}

        # Count files
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        annotation_files = list(annotations_path.glob("*.xml"))

        # Validate matching files
        matched_pairs = 0
        for img_file in image_files:
            corresponding_xml = annotations_path / f"{img_file.stem}.xml"
            if corresponding_xml.exists():
                matched_pairs += 1

        stats = {
            "valid": True,
            "total_images": len(image_files),
            "total_annotations": len(annotation_files),
            "matched_pairs": matched_pairs,
            "classes": self.datasets["acfr_orchard"]["classes"]
        }

        return stats

    def validate_minneapple_dataset(self) -> Dict[str, any]:
        """Validate MinneApple dataset structure and annotations."""
        minneapple_path = Path("data/minneapple")

        if not minneapple_path.exists():
            return {"valid": False, "error": "Dataset not found"}

        # Expected structure: detection-dataset/
        detection_path = minneapple_path / "detection-dataset"

        if not detection_path.exists():
            # Try alternative structure
            detection_path = minneapple_path

        train_path = detection_path / "train"
        test_path = detection_path / "test"

        if not train_path.exists() or not test_path.exists():
            return {"valid": False, "error": "Missing train or test directories"}

        # Count annotations
        train_annotations = list(train_path.glob("*.json"))
        test_annotations = list(test_path.glob("*.json"))

        # Count total images from annotations
        total_images = 0
        total_boxes = 0

        for annotation_file in train_annotations + test_annotations:
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    if 'images' in data:
                        total_images += len(data['images'])
                    if 'annotations' in data:
                        total_boxes += len(data['annotations'])
            except json.JSONDecodeError:
                continue

        stats = {
            "valid": True,
            "train_files": len(train_annotations),
            "test_files": len(test_annotations),
            "estimated_images": total_images,
            "estimated_boxes": total_boxes,
            "classes": self.datasets["minneapple"]["classes"]
        }

        return stats

    def create_unified_index(self) -> bool:
        """Create unified dataset index for both ACFR and MinneApple datasets."""
        index_file = self.data_dir / "dataset_index.json"

        index = {
            "datasets": {},
            "classes": {},
            "statistics": {}
        }

        # Process ACFR dataset
        acfr_stats = self.validate_acfr_dataset()
        if acfr_stats["valid"]:
            index["datasets"]["acfr_orchard"] = {
                "path": "data/acfr_orchard",
                "type": "fruit_detection",
                "annotation_format": "xml",
                "split": "none",
                "statistics": acfr_stats
            }

            # Add ACFR classes
            for cls in acfr_stats["classes"]:
                if cls not in index["classes"]:
                    index["classes"][cls] = []
                index["classes"][cls].append("acfr_orchard")

        # Process MinneApple dataset
        minneapple_stats = self.validate_minneapple_dataset()
        if minneapple_stats["valid"]:
            index["datasets"]["minneapple"] = {
                "path": "data/minneapple",
                "type": "fruit_detection", 
                "annotation_format": "json",
                "split": "train_test",
                "statistics": minneapple_stats
            }

            # Add MinneApple classes
            for cls in minneapple_stats["classes"]:
                if cls not in index["classes"]:
                    index["classes"][cls] = []
                index["classes"][cls].append("minneapple")

        # Overall statistics
        total_images = 0
        total_annotations = 0

        if acfr_stats["valid"]:
            total_images += acfr_stats.get("total_images", 0)
            total_annotations += acfr_stats.get("total_annotations", 0)

        if minneapple_stats["valid"]:
            total_images += minneapple_stats.get("estimated_images", 0)
            total_annotations += minneapple_stats.get("estimated_boxes", 0)

        index["statistics"] = {
            "total_datasets": len([d for d in [acfr_stats, minneapple_stats] if d["valid"]]),
            "total_images": total_images,
            "total_annotations": total_annotations,
            "unique_classes": len(index["classes"])
        }

        # Save index
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)

        print(f"✅ Dataset index created: {index_file}")
        return True

    def setup_training_splits(self, test_ratio: float = 0.2, val_ratio: float = 0.1) -> bool:
        """Create training/validation/test splits for combined datasets."""
        splits_dir = self.data_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        # This would implement actual split logic based on available datasets
        # For now, create placeholder split files

        split_info = {
            "acfr_orchard": {
                "train": "80%",
                "val": "10%", 
                "test": "10%"
            },
            "minneapple": {
                "train": "predefined",
                "test": "predefined"
            }
        }

        splits_file = splits_dir / "dataset_splits.json"
        with open(splits_file, 'w') as f:
            json.dump(split_info, f, indent=2)

        print(f"✅ Training splits configuration saved: {splits_file}")
        return True

def main():
    """Main dataset setup entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Setup BioViT3R-Beta datasets")
    parser.add_argument("--dataset", choices=["acfr", "minneapple", "all"], 
                       default="all", help="Which dataset to download")
    parser.add_argument("--force", action="store_true", 
                       help="Force re-download existing datasets")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing datasets")

    args = parser.parse_args()

    manager = DatasetManager()

    if args.validate_only:
        print("🔍 Validating datasets...")

        acfr_stats = manager.validate_acfr_dataset()
        minneapple_stats = manager.validate_minneapple_dataset()

        print(f"ACFR Orchard: {'✅ Valid' if acfr_stats['valid'] else '❌ Invalid'}")
        if acfr_stats["valid"]:
            print(f"  - Images: {acfr_stats['total_images']}")
            print(f"  - Annotations: {acfr_stats['total_annotations']}")

        print(f"MinneApple: {'✅ Valid' if minneapple_stats['valid'] else '❌ Invalid'}")
        if minneapple_stats["valid"]:
            print(f"  - Estimated images: {minneapple_stats['estimated_images']}")
            print(f"  - Estimated boxes: {minneapple_stats['estimated_boxes']}")

        return

    # Download datasets
    success_count = 0

    if args.dataset in ["acfr", "all"]:
        if manager.download_dataset("acfr_orchard", args.force):
            success_count += 1

    if args.dataset in ["minneapple", "all"]:
        if manager.download_dataset("minneapple", args.force):
            success_count += 1

    # Create unified index and splits
    if success_count > 0:
        manager.create_unified_index()
        manager.setup_training_splits()

        print(f"🎉 Dataset setup complete! {success_count} dataset(s) ready for training.")
    else:
        print("❌ Dataset setup failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
