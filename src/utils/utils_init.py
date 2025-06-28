# BioViT3R-Beta Utils Package
# Initialization module for utility functions

from .visualization import (
    PlantVisualization, 
    VisualizationConfig,
    quick_3d_plot,
    quick_detection_plot,
    save_figure_base64
)

from .geometric_utils import (
    GeometricUtils,
    CameraParameters,
    distance_point_to_plane,
    project_point_to_plane,
    compute_angle_between_vectors
)

from .file_utils import (
    FileUtils,
    batch_convert_images,
    find_files_by_size
)

from .metrics import (
    PlantAnalysisMetrics,
    DetectionMetrics,
    ReconstructionMetrics,
    ClassificationMetrics,
    compute_pixel_accuracy,
    compute_dice_coefficient,
    compute_jaccard_index
)

from .video_processor import (
    VideoProcessor,
    FrameExtractor,
    VideoInfo,
    FrameInfo,
    simple_overlay_function,
    batch_process_videos
)

__version__ = "1.0.0-beta"
__author__ = "BioViT3R-Beta Development Team"

# Define what's available when using "from src.utils import *"
__all__ = [
    # Visualization utilities
    'PlantVisualization',
    'VisualizationConfig', 
    'quick_3d_plot',
    'quick_detection_plot',
    'save_figure_base64',
    
    # Geometric utilities
    'GeometricUtils',
    'CameraParameters',
    'distance_point_to_plane',
    'project_point_to_plane',
    'compute_angle_between_vectors',
    
    # File utilities
    'FileUtils',
    'batch_convert_images',
    'find_files_by_size',
    
    # Metrics
    'PlantAnalysisMetrics',
    'DetectionMetrics',
    'ReconstructionMetrics',
    'ClassificationMetrics',
    'compute_pixel_accuracy',
    'compute_dice_coefficient',
    'compute_jaccard_index',
    
    # Video processing
    'VideoProcessor',
    'FrameExtractor',
    'VideoInfo',
    'FrameInfo',
    'simple_overlay_function',
    'batch_process_videos'
]

# Package-level configuration
DEFAULT_CONFIG = {
    'visualization': {
        'point_size': 2.0,
        'mesh_opacity': 0.8,
        'colormap': 'viridis',
        'figure_size': (1200, 800)
    },
    'geometric': {
        'default_fov': 60.0,
        'icp_threshold': 0.02,
        'normal_estimation_neighbors': 20
    },
    'video': {
        'default_fps': 30,
        'max_workers': 4,
        'default_frame_interval': 30
    },
    'metrics': {
        'iou_threshold': 0.5,
        'confidence_threshold': 0.5,
        'reconstruction_threshold': 0.01
    }
}

def get_default_config() -> dict:
    """Get default configuration for utils package"""
    return DEFAULT_CONFIG.copy()

def setup_logging(level: str = 'INFO'):
    """Setup logging for utils package"""
    import logging
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set specific loggers for utils modules
    for module in ['visualization', 'geometric_utils', 'file_utils', 'metrics', 'video_processor']:
        logger = logging.getLogger(f'src.utils.{module}')
        logger.setLevel(log_level)