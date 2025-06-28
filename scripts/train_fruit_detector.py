#!/usr/bin/env python3
"""
BioViT3R-Beta Custom Fruit Detector Training Script
Trains custom fruit detection models using ACFR and MinneApple datasets.
"""

import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import wandb
from datetime import datetime
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.datasets import ACFRDataset, MinneAppleDataset
from src.data.augmentation import get_training_augmentation, get_validation_augmentation
from src.models.fruit_detector import FruitDetector
from src.utils.metrics import DetectionMetrics

class FruitDetectorTrainer:
    """Handles training of custom fruit detection models."""

    def __init__(self, config_path: str = "configs/training_config.yaml"):
        self.config_path = config_path
        self.load_config()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")

        # Initialize tracking
        if self.config.get("use_wandb", False):
            wandb.init(
                project="biovit3r-fruit-detection",
                config=self.config,
                name=f"fruit_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

    def load_config(self):
        """Load training configuration."""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                "model": {
                    "backbone": "resnet50",
                    "num_classes": 6,  # 5 fruit classes + background
                    "pretrained": True
                },
                "training": {
                    "batch_size": 8,
                    "learning_rate": 0.001,
                    "num_epochs": 50,
                    "weight_decay": 1e-4,
                    "gradient_clip": 1.0
                },
                "datasets": {
                    "acfr_weight": 0.6,
                    "minneapple_weight": 0.4,
                    "augmentation_strength": "medium"
                },
                "validation": {
                    "freq": 5,
                    "save_best": True
                },
                "use_wandb": False
            }

    def setup_datasets(self):
        """Setup training and validation datasets."""
        print("📂 Setting up datasets...")

        # Training augmentations
        train_transform = get_training_augmentation(
            size=(416, 416),
            strength=self.config["datasets"]["augmentation_strength"]
        )

        # Validation augmentations
        val_transform = get_validation_augmentation(size=(416, 416))

        # ACFR Dataset
        acfr_train_dataset = ACFRDataset(
            root_dir="data/acfr_orchard",
            split="train",
            transform=train_transform
        )

        acfr_val_dataset = ACFRDataset(
            root_dir="data/acfr_orchard", 
            split="val",
            transform=val_transform
        )

        # MinneApple Dataset
        minneapple_train_dataset = MinneAppleDataset(
            root_dir="data/minneapple/detection-dataset",
            split="train",
            transform=train_transform
        )

        # Combine datasets with weighted sampling
        from torch.utils.data import ConcatDataset, WeightedRandomSampler

        combined_train_dataset = ConcatDataset([
            acfr_train_dataset,
            minneapple_train_dataset
        ])

        # Create data loaders
        self.train_loader = DataLoader(
            combined_train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn
        )

        self.val_loader = DataLoader(
            acfr_val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn
        )

        print(f"✅ Training samples: {len(combined_train_dataset)}")
        print(f"✅ Validation samples: {len(acfr_val_dataset)}")

    def collate_fn(self, batch):
        """Custom collate function for object detection."""
        images = []
        targets = []

        for sample in batch:
            images.append(sample['image'])
            targets.append(sample['target'])

        return images, targets

    def setup_model(self):
        """Initialize the fruit detection model."""
        print("🤖 Setting up model...")

        self.model = FruitDetector(
            backbone=self.config["model"]["backbone"],
            num_classes=self.config["model"]["num_classes"],
            pretrained=self.config["model"]["pretrained"]
        )

        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )

        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        # Metrics tracker
        self.metrics = DetectionMetrics(num_classes=self.config["model"]["num_classes"])

        print("✅ Model setup complete")

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        print(f"🏃 Training Epoch {epoch+1}/{self.config['training']['num_epochs']}")

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)

            # Calculate total loss
            loss = sum(loss_dict.values())

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config["training"]["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["gradient_clip"]
                )

            self.optimizer.step()

            total_loss += loss.item()

            # Progress logging
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"📊 Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

        return avg_loss

    def validate_epoch(self, epoch: int):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0

        print(f"🔍 Validating Epoch {epoch+1}")

        with torch.no_grad():
            for images, targets in self.val_loader:
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

                total_loss += loss.item()

                # Update metrics
                predictions = self.model(images)
                self.metrics.update(predictions, targets)

        avg_loss = total_loss / len(self.val_loader)
        metrics_result = self.metrics.compute()

        print(f"📊 Epoch {epoch+1} Validation Loss: {avg_loss:.4f}")
        print(f"📊 Epoch {epoch+1} mAP@0.5: {metrics_result['map_50']:.4f}")

        return avg_loss, metrics_result

    def save_checkpoint(self, epoch: int, loss: float, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoints_dir = Path("checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = checkpoints_dir / f"fruit_detector_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = checkpoints_dir / "fruit_detector_best.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved: {best_path}")

        print(f"💾 Checkpoint saved: {checkpoint_path}")

    def train(self):
        """Main training loop."""
        print("🚀 Starting BioViT3R-Beta Fruit Detector Training")

        self.setup_datasets()
        self.setup_model()

        best_map = 0.0

        for epoch in range(self.config["training"]["num_epochs"]):
            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            if (epoch + 1) % self.config["validation"]["freq"] == 0:
                val_loss, metrics = self.validate_epoch(epoch)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Check if best model
                current_map = metrics['map_50']
                is_best = current_map > best_map
                if is_best:
                    best_map = current_map

                # Save checkpoint
                self.save_checkpoint(epoch, val_loss, metrics, is_best)

                # Wandb logging
                if self.config.get("use_wandb", False):
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "map_50": current_map,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })

        print("🎉 Training completed!")
        print(f"🏆 Best mAP@0.5: {best_map:.4f}")

def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description="Train BioViT3R-Beta fruit detector")
    parser.add_argument("--config", default="configs/training_config.yaml",
                       help="Training configuration file")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--resume", type=str,
                       help="Resume training from checkpoint")

    args = parser.parse_args()

    # Update config with command line args
    if args.wandb:
        # Enable wandb in config if not already set
        pass

    trainer = FruitDetectorTrainer(args.config)

    if args.resume:
        print(f"📂 Resuming training from {args.resume}")
        # Load checkpoint logic would go here

    trainer.train()

if __name__ == "__main__":
    main()
