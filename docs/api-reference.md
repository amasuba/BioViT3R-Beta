# BioViT3R-Beta API Reference

## Overview

The BioViT3R-Beta API provides comprehensive plant analysis capabilities through VGGT 3D reconstruction, fruit detection, health assessment, and IBM Granite AI assistance. This document covers all available classes, methods, and endpoints.

## Core Models

### VGGTReconstructor

3D reconstruction engine using VGGT architecture for single-image plant analysis.

#### Class: `VGGTReconstructor`

```python
from src.models.vggt_reconstructor import VGGTReconstructor

reconstructor = VGGTReconstructor(
    model_name="vggt_base",
    device="cuda",
    quality_preset="high"
)
```

**Parameters:**
- `model_name` (str): VGGT model variant ("vggt_base", "vggt_large")
- `device` (str): Computation device ("cuda", "cpu", "mps")
- `quality_preset` (str): Output quality ("low", "medium", "high", "ultra")

#### Methods

##### `reconstruct_3d(image_path, output_format="ply")`

Performs 3D reconstruction from single plant image.

**Parameters:**
- `image_path` (str): Path to input plant image
- `output_format` (str): Output format ("ply", "obj", "stl")

**Returns:**
- `dict`: Reconstruction results containing point cloud, mesh, and metadata

**Example:**
```python
result = reconstructor.reconstruct_3d(
    "plant_image.jpg",
    output_format="ply"
)
point_cloud = result["point_cloud"]
mesh = result["mesh"]
volume = result["metadata"]["volume_cm3"]
```

##### `batch_reconstruct(image_list, parallel=True)`

Processes multiple images in batch mode.

**Parameters:**
- `image_list` (List[str]): List of image paths
- `parallel` (bool): Enable parallel processing

**Returns:**
- `List[dict]`: List of reconstruction results

### PlantAnalyzer

Comprehensive plant health and growth stage analysis.

#### Class: `PlantAnalyzer`

```python
from src.models.plant_analyzer import PlantAnalyzer

analyzer = PlantAnalyzer(
    health_model="plant_health_v2",
    growth_model="growth_stages_v2",
    device="cuda"
)
```

#### Methods

##### `analyze_health(image_path, confidence_threshold=0.7)`

Performs comprehensive plant health assessment.

**Parameters:**
- `image_path` (str): Path to plant image
- `confidence_threshold` (float): Minimum confidence for disease detection

**Returns:**
- `dict`: Health analysis results

**Example:**
```python
health_result = analyzer.analyze_health("diseased_plant.jpg")
health_score = health_result["overall_health_score"]
diseases = health_result["detected_diseases"]
stress_indicators = health_result["stress_indicators"]
```

##### `classify_growth_stage(image_path)`

Determines plant growth stage and phenological phase.

**Returns:**
- `dict`: Growth stage classification with confidence scores

### FruitDetector

Multi-dataset fruit detection using ACFR and MinneApple datasets.

#### Class: `FruitDetector`

```python
from src.models.fruit_detector import FruitDetector

detector = FruitDetector(
    dataset="combined",  # "acfr", "minneapple", "combined"
    model_type="faster_rcnn",
    confidence_threshold=0.5
)
```

#### Methods

##### `detect_fruits(image_path, return_crops=False)`

Detects and localizes fruits in plant images.

**Parameters:**
- `image_path` (str): Path to input image
- `return_crops` (bool): Return individual fruit crops

**Returns:**
- `dict`: Detection results with bounding boxes and confidence scores

## Data Processing

### DatasetLoader

Universal dataset management for ACFR and MinneApple datasets.

#### Class: `DatasetLoader`

```python
from src.data.datasets import DatasetLoader

loader = DatasetLoader(
    dataset_name="acfr",
    data_dir="data/acfr_orchard",
    split="train"
)
```

#### Methods

##### `load_annotations(format="coco")`

Loads dataset annotations in specified format.

**Parameters:**
- `format` (str): Annotation format ("coco", "yolo", "pascal_voc")

**Returns:**
- `dict`: Structured annotation data

##### `get_image_paths()`

Returns list of all image paths in the dataset.

**Returns:**
- `List[str]`: Image file paths

### Preprocessing

Universal image preprocessing for all models.

#### Functions

##### `preprocess_image(image_path, target_size=(224, 224), model_type="vggt")`

Preprocesses images for specific model requirements.

**Parameters:**
- `image_path` (str): Input image path
- `target_size` (tuple): Target image dimensions
- `model_type` (str): Target model type for preprocessing

**Returns:**
- `torch.Tensor`: Preprocessed image tensor

## AI Assistant

### GraniteClient

IBM Watsonx AI client for agricultural expertise.

#### Class: `GraniteClient`

```python
from src.ai_assistant.granite_client import GraniteClient

client = GraniteClient(
    api_key="your_api_key",
    project_id="your_project_id",
    model_id="ibm/granite-13b-chat-v2"
)
```

#### Methods

##### `chat(message, context=None, max_tokens=500)`

Generates AI responses with agricultural context.

**Parameters:**
- `message` (str): User message or question
- `context` (dict): Analysis context from BioViT3R results
- `max_tokens` (int): Maximum response length

**Returns:**
- `str`: AI-generated response

### ContextManager

Analysis context management for informed AI conversations.

#### Class: `ContextManager`

```python
from src.ai_assistant.context_manager import ContextManager

context_mgr = ContextManager(max_history=10)
```

#### Methods

##### `add_analysis_result(result_data, analysis_type)`

Stores analysis results for context awareness.

**Parameters:**
- `result_data` (dict): Analysis results from any model
- `analysis_type` (str): Type of analysis ("3d", "health", "fruit", "growth")

##### `get_context_summary()`

Generates formatted context summary for AI assistant.

**Returns:**
- `str`: Formatted context summary

## Utility Functions

### Visualization

3D point cloud and analysis visualization utilities.

#### Functions

##### `plot_point_cloud(point_cloud, colors=None, title="3D Reconstruction")`

Visualizes 3D point clouds with optional coloring.

##### `create_analysis_report(results_dict, output_path)`

Generates comprehensive HTML analysis reports.

### Geometric Utils

3D geometry calculations and transformations.

#### Functions

##### `calculate_volume(point_cloud, method="convex_hull")`

Calculates plant volume from 3D point cloud.

##### `estimate_biomass(volume_cm3, plant_type="generic")`

Estimates biomass using allometric relationships.

## Configuration

### Application Configuration

Main configuration file: `configs/app_config.yaml`

```yaml
models:
  vggt:
    model_name: "vggt_base"
    quality_preset: "high"
    device: "auto"
  
  plant_health:
    confidence_threshold: 0.7
    model_version: "v2"
  
  fruit_detection:
    dataset: "combined"
    nms_threshold: 0.5

ai_assistant:
  model_id: "ibm/granite-13b-chat-v2"
  max_tokens: 500
  temperature: 0.7

interface:
  theme: "default"
  max_upload_size_mb: 50
```

## Error Handling

### Common Exceptions

#### `VGGTReconstructionError`

Raised when 3D reconstruction fails.

#### `ModelLoadError`

Raised when model loading fails.

#### `PreprocessingError`

Raised during image preprocessing failures.

### Error Response Format

```json
{
  "error": true,
  "error_type": "VGGTReconstructionError",
  "message": "Failed to reconstruct 3D model from input image",
  "details": {
    "input_image": "path/to/image.jpg",
    "error_code": "INSUFFICIENT_FEATURES"
  }
}
```

## Performance Considerations

### Hardware Requirements

- **Minimum**: 8GB RAM, GTX 1060 or equivalent
- **Recommended**: 16GB RAM, RTX 3080 or better
- **Optimal**: 32GB RAM, RTX 4090 or A100

### Optimization Tips

1. Use batch processing for multiple images
2. Enable GPU acceleration when available
3. Adjust quality presets based on requirements
4. Cache model weights for faster initialization

## Examples

### Complete Analysis Pipeline

```python
from src.models import VGGTReconstructor, PlantAnalyzer, FruitDetector
from src.ai_assistant import GraniteClient, ContextManager

# Initialize models
reconstructor = VGGTReconstructor()
analyzer = PlantAnalyzer()
detector = FruitDetector()
ai_client = GraniteClient()
context_mgr = ContextManager()

# Perform analysis
image_path = "plant_sample.jpg"

# 3D reconstruction
reconstruction = reconstructor.reconstruct_3d(image_path)
context_mgr.add_analysis_result(reconstruction, "3d")

# Health analysis
health = analyzer.analyze_health(image_path)
context_mgr.add_analysis_result(health, "health")

# Fruit detection
fruits = detector.detect_fruits(image_path)
context_mgr.add_analysis_result(fruits, "fruit")

# AI assistance
context = context_mgr.get_context_summary()
response = ai_client.chat(
    "What can you tell me about this plant's condition?",
    context=context
)
```

### Custom Training

```python
from scripts.train_fruit_detector import train_custom_detector

# Train custom fruit detector
config = {
    "dataset": "custom",
    "data_dir": "data/custom_fruits",
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 0.001
}

model_path = train_custom_detector(config)
```

## Version History

- **v1.0.0-beta**: Initial release with VGGT integration
- **v1.1.0-beta**: Added MinneApple dataset support
- **v1.2.0-beta**: Enhanced AI assistant capabilities
- **v1.3.0-beta**: Performance optimizations and bug fixes

## Support

For technical support and feature requests:
- GitHub Issues: [BioViT3R-Beta Issues](https://github.com/user/BioViT3R-Beta/issues)
- Documentation: [Full Documentation](docs/README.md)
- Email: support@biovit3r.org