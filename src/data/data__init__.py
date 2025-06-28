"""
BioViT3R-Beta Data Package

This package contains data processing utilities including:
- Dataset loading for ACFR and MinneApple
- Image and video preprocessing
- Data augmentation pipelines
- Video frame extraction utilities
"""

from .datasets import DatasetLoader, ACFRDataset, MinneAppleDataset
from .preprocessing import ImagePreprocessor, VideoPreprocessor, VGGTPreprocessor
from .augmentation import get_training_augmentation, get_validation_augmentation
from .video_utils import VideoFrameExtractor, extract_frames_from_video

__version__ = "1.0.0-beta"
__author__ = "BioViT3R Team"

__all__ = [
    "DatasetLoader",
    "ACFRDataset", 
    "MinneAppleDataset",
    "ImagePreprocessor",
    "VideoPreprocessor", 
    "VGGTPreprocessor",
    "get_training_augmentation",
    "get_validation_augmentation",
    "VideoFrameExtractor",
    "extract_frames_from_video",
]

# Dataset registry for dynamic loading
DATASET_REGISTRY = {
    "acfr": ACFRDataset,
    "minneapple": MinneAppleDataset,
    "combined": DatasetLoader,
}

def get_dataset(dataset_name: str, **kwargs):
    """
    Factory function to get dataset instances by name.
    
    Args:
        dataset_name (str): Name of the dataset to load
        **kwargs: Additional arguments passed to dataset constructor
        
    Returns:
        Dataset instance
        
    Raises:
        ValueError: If dataset_name not found in registry
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASET_REGISTRY.keys())}")
    
    return DATASET_REGISTRY[dataset_name](**kwargs)

def list_available_datasets():
    """Return list of available dataset names."""
    return list(DATASET_REGISTRY.keys())

# Common preprocessing configurations
PREPROCESSING_CONFIGS = {
    "default": {
        "resize": (224, 224),
        "normalize": True,
        "augment": False,
    },
    "training": {
        "resize": (224, 224),
        "normalize": True,
        "augment": True,
    },
    "vggt": {
        "resize": (512, 512),
        "normalize": True,
        "augment": False,
    },
}

def get_preprocessing_config(config_name: str = "default"):
    """Get preprocessing configuration by name."""
    return PREPROCESSING_CONFIGS.get(config_name, PREPROCESSING_CONFIGS["default"])