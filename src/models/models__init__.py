"""
BioViT3R-Beta Models Package

This package contains the core AI models for plant analysis including:
- VGGT 3D reconstruction
- Plant health analysis
- Fruit detection using ACFR and MinneApple datasets
- Biomass estimation
- Growth stage classification
"""

from .vggt_reconstructor import VGGTReconstructor
from .plant_analyzer import PlantAnalyzer, HealthClassifier
from .fruit_detector import FruitDetector
from .biomass_estimator import BiomassEstimator

__version__ = "1.0.0-beta"
__author__ = "BioViT3R Team"

__all__ = [
    "VGGTReconstructor",
    "PlantAnalyzer", 
    "HealthClassifier",
    "FruitDetector",
    "BiomassEstimator",
]

# Model registry for dynamic loading
MODEL_REGISTRY = {
    "vggt": VGGTReconstructor,
    "plant_analyzer": PlantAnalyzer,
    "health_classifier": HealthClassifier, 
    "fruit_detector": FruitDetector,
    "biomass_estimator": BiomassEstimator,
}

def get_model(model_name: str, **kwargs):
    """
    Factory function to get model instances by name.
    
    Args:
        model_name (str): Name of the model to instantiate
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model_name not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](**kwargs)

def list_available_models():
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())