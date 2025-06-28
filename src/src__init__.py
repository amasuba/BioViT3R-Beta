"""
BioViT3R-Beta: Advanced AI-Powered Plant Analysis Platform

A comprehensive agricultural AI platform featuring:
- VGGT 3D plant reconstruction
- Multi-dataset fruit detection (ACFR + MinneApple) 
- Plant health and growth stage analysis
- Biomass estimation
- IBM Granite AI-powered agricultural assistance

Main Components:
- models: Core AI models for plant analysis
- data: Dataset loading and preprocessing utilities
- utils: Visualization, geometry, and evaluation tools
- ai_assistant: Intelligent agricultural AI assistance
"""

from . import models
from . import data
from . import utils  
from . import ai_assistant

__version__ = "1.0.0-beta"
__author__ = "BioViT3R Team"
__license__ = "MIT"
__description__ = "Advanced AI-Powered Plant Analysis Platform"

__all__ = [
    "models",
    "data", 
    "utils",
    "ai_assistant",
]

# Main analysis pipeline convenience function
def create_analysis_pipeline(config_path: str = None, **kwargs):
    """
    Create a complete BioViT3R analysis pipeline.
    
    Args:
        config_path (str, optional): Path to configuration file
        **kwargs: Additional configuration parameters
        
    Returns:
        dict: Dictionary containing initialized components
    """
    # Initialize core models
    vggt_reconstructor = models.VGGTReconstructor(**kwargs)
    plant_analyzer = models.PlantAnalyzer(**kwargs)
    fruit_detector = models.FruitDetector(**kwargs)
    biomass_estimator = models.BiomassEstimator(**kwargs)
    
    # Initialize data processors
    image_preprocessor = data.ImagePreprocessor(**kwargs)
    video_preprocessor = data.VideoPreprocessor(**kwargs)
    
    # Initialize utilities
    visualizer = utils.PlantVisualizer(**kwargs)
    video_processor = utils.VideoProcessor(**kwargs)
    
    # Initialize AI assistant
    chat_interface, granite_client, context_manager = ai_assistant.create_assistant(**kwargs)
    
    return {
        "models": {
            "vggt_reconstructor": vggt_reconstructor,
            "plant_analyzer": plant_analyzer,
            "fruit_detector": fruit_detector,
            "biomass_estimator": biomass_estimator,
        },
        "data": {
            "image_preprocessor": image_preprocessor,
            "video_preprocessor": video_preprocessor,
        },
        "utils": {
            "visualizer": visualizer,
            "video_processor": video_processor,
        },
        "ai_assistant": {
            "chat_interface": chat_interface,
            "granite_client": granite_client,
            "context_manager": context_manager,
        }
    }

# Version information
def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__, 
        "description": __description__,
    }