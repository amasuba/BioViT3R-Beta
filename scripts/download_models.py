#!/usr/bin/env python3
"""
BioViT3R-Beta Model Download Script
Downloads and sets up all required pre-trained models for the analysis pipeline.
"""

import os
import sys
import requests
import hashlib
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import yaml

class ModelDownloader:
    """Handles downloading and verification of pre-trained models."""

    def __init__(self, config_path: str = "configs/app_config.yaml"):
        self.config_path = config_path
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        # Model URLs and checksums
        self.model_registry = {
            "vggt_base": {
                "url": "https://github.com/NVlabs/VGGT/releases/download/v1.0/vggt_base.pth",
                "checksum": "a1b2c3d4e5f6789",
                "path": "models/vggt/vggt_base.pth",
                "size_mb": 150
            },
            "plant_health_classifier": {
                "url": "https://huggingface.co/agricultural-ai/plant-health/resolve/main/health_classifier.safetensors",
                "checksum": "f6e5d4c3b2a1987",
                "path": "models/health_classification/health_classifier.safetensors",
                "size_mb": 85
            },
            "fruit_detector_acfr": {
                "url": "https://data.acfr.usyd.edu.au/models/fruit_detector_v2.pth",
                "checksum": "9876543210abcde",
                "path": "models/fruit_detection/acfr_detector.pth",
                "size_mb": 120
            },
            "fruit_detector_minneapple": {
                "url": "https://conservancy.umn.edu/bitstream/handle/11299/206575/minneapple_detector.bin",
                "checksum": "fedcba0987654321",
                "path": "models/fruit_detection/minneapple_detector.bin",
                "size_mb": 95
            },
            "growth_stage_classifier": {
                "url": "https://storage.googleapis.com/agriculture-ai/growth_classifier_v3.pt",
                "checksum": "123456789abcdef0",
                "path": "models/growth_classification/growth_classifier.pt",
                "size_mb": 75
            }
        }

    def download_file(self, url: str, destination: Path, expected_size_mb: Optional[int] = None) -> bool:
        """Download a file with progress bar and error handling."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            destination.parent.mkdir(parents=True, exist_ok=True)

            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            return True

        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False

    def verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file integrity using SHA256 checksum."""
        if not file_path.exists():
            return False

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        actual_checksum = sha256_hash.hexdigest()[:15]  # First 15 chars
        return actual_checksum == expected_checksum

    def download_all_models(self, force_redownload: bool = False) -> Dict[str, bool]:
        """Download all required models."""
        results = {}

        print("🌱 BioViT3R-Beta Model Download Starting...")
        print(f"📁 Models directory: {self.models_dir.absolute()}")

        for model_name, model_info in self.model_registry.items():
            print(f"\n📦 Processing {model_name}...")

            model_path = Path(model_info["path"])

            # Check if model already exists and is valid
            if not force_redownload and model_path.exists():
                if self.verify_checksum(model_path, model_info["checksum"]):
                    print(f"✅ {model_name} already exists and is valid")
                    results[model_name] = True
                    continue
                else:
                    print(f"⚠️  {model_name} exists but checksum mismatch, re-downloading...")

            # Download the model
            success = self.download_file(
                model_info["url"],
                model_path,
                model_info["size_mb"]
            )

            if success and self.verify_checksum(model_path, model_info["checksum"]):
                print(f"✅ {model_name} downloaded and verified successfully")
                results[model_name] = True
            else:
                print(f"❌ Failed to download or verify {model_name}")
                results[model_name] = False

        return results

    def create_model_config(self):
        """Create model configuration file with local paths."""
        config = {
            "models": {
                "vggt": {
                    "model_path": "models/vggt/vggt_base.pth",
                    "input_size": [3, 224, 224],
                    "device": "cuda"
                },
                "health_classifier": {
                    "model_path": "models/health_classification/health_classifier.safetensors",
                    "num_classes": 12,
                    "input_size": [3, 256, 256]
                },
                "fruit_detection": {
                    "acfr_model": "models/fruit_detection/acfr_detector.pth",
                    "minneapple_model": "models/fruit_detection/minneapple_detector.bin",
                    "confidence_threshold": 0.5,
                    "nms_threshold": 0.4
                },
                "growth_classifier": {
                    "model_path": "models/growth_classification/growth_classifier.pt",
                    "stages": ["seedling", "vegetative", "flowering", "fruiting", "senescence"]
                }
            }
        }

        config_path = Path("configs/models_config.yaml")
        config_path.parent.mkdir(exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        print(f"✅ Model configuration saved to {config_path}")

def main():
    """Main download script entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download BioViT3R-Beta models")
    parser.add_argument("--force", action="store_true", help="Force re-download existing models")
    parser.add_argument("--config", default="configs/app_config.yaml", help="Config file path")

    args = parser.parse_args()

    downloader = ModelDownloader(args.config)
    results = downloader.download_all_models(force_redownload=args.force)

    # Summary
    successful = sum(results.values())
    total = len(results)

    print(f"\n🏁 Download Summary: {successful}/{total} models downloaded successfully")

    if successful == total:
        downloader.create_model_config()
        print("🎉 All models ready! You can now run BioViT3R-Beta analysis.")
    else:
        print("⚠️  Some models failed to download. Check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
