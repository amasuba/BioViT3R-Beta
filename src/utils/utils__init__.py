"""
BioViT3R-Beta Utilities Package

This package contains utility functions for:
- 3D visualization and rendering
- Geometric calculations and transformations
- File I/O operations
- Evaluation metrics
- Video processing
"""

from .visualization import (
    PlantVisualizer,
    plot_3d_reconstruction,
    plot_analysis_results,
    create_health_dashboard,
    visualize_fruit_detection
)
from .geometric_utils import (
    calculate_plant_volume,
    estimate_surface_area,
    transform_point_cloud,
    filter_outliers,
    compute_bounding_box
)
from .file_utils import (
    load_image,
    save_image,
    load_point_cloud,
    save_point_cloud,
    load_mesh,
    save_mesh,
    ensure_directory,
    get_file_info
)
from .metrics import (
    calculate_iou,
    calculate_map,
    calculate_chamfer_distance,
    calculate_classification_metrics,
    evaluate_detection_results
)
from .video_processor import (
    VideoProcessor,
    extract_frames,
    create_timelapse,
    analyze_temporal_growth
)

__version__ = "1.0.0-beta"
__author__ = "BioViT3R Team"

__all__ = [
    # Visualization
    "PlantVisualizer",
    "plot_3d_reconstruction",
    "plot_analysis_results", 
    "create_health_dashboard",
    "visualize_fruit_detection",
    
    # Geometric utilities
    "calculate_plant_volume",
    "estimate_surface_area",
    "transform_point_cloud",
    "filter_outliers",
    "compute_bounding_box",
    
    # File utilities
    "load_image",
    "save_image",
    "load_point_cloud",
    "save_point_cloud",
    "load_mesh",
    "save_mesh",
    "ensure_directory",
    "get_file_info",
    
    # Metrics
    "calculate_iou",
    "calculate_map",
    "calculate_chamfer_distance",
    "calculate_classification_metrics",
    "evaluate_detection_results",
    
    # Video processing
    "VideoProcessor",
    "extract_frames",
    "create_timelapse", 
    "analyze_temporal_growth",
]

# Utility function registry
UTILITY_REGISTRY = {
    "visualizer": PlantVisualizer,
    "video_processor": VideoProcessor,
}

def get_utility(utility_name: str, **kwargs):
    """
    Factory function to get utility instances by name.
    
    Args:
        utility_name (str): Name of the utility to instantiate
        **kwargs: Additional arguments passed to utility constructor
        
    Returns:
        Utility instance
        
    Raises:
        ValueError: If utility_name not found in registry
    """
    if utility_name not in UTILITY_REGISTRY:
        raise ValueError(f"Utility '{utility_name}' not found. Available utilities: {list(UTILITY_REGISTRY.keys())}")
    
    return UTILITY_REGISTRY[utility_name](**kwargs)