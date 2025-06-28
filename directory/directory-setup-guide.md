# BioViT3R-Beta Directory Structure Setup Guide

## Understanding .gitkeep Files

`.gitkeep` files are a convention used to maintain empty directories in Git version control. Git doesn't track empty directories, so these placeholder files ensure the directory structure is preserved when cloning the repository.

## Directory Structure Overview

The BioViT3R-Beta repository uses the following data and asset organization:

### `models/` - Model Storage Directory
```
models/
├── vggt/                    # VGGT 3D reconstruction models
├── fruit_detection/         # Fruit detection models
│   ├── acfr/               # ACFR Orchard dataset models
│   └── minneapple/         # MinneApple dataset models
├── health_classification/   # Plant health assessment models
└── growth_classification/   # Growth stage classification models
```

### `data/` - Dataset Organization
```
data/
├── raw/                    # Original, unprocessed datasets
├── processed/              # Cleaned and preprocessed data
├── interim/                # Temporary processing outputs
├── external/               # Third-party datasets
├── acfr_orchard/          # ACFR Orchard Fruit Dataset
└── minneapple/            # MinneApple Dataset
```

### `assets/` - Demo and Documentation Materials
```
assets/
├── demo_images/           # Example plant images for testing
├── 3d_models/            # Sample 3D reconstructions
├── documentation/        # Screenshots and diagrams
└── visualizations/       # Analysis output examples
```

### `logs/` and `outputs/` - Runtime Directories
```
logs/                     # Application and error logs
outputs/                  # Analysis results and exports
```

## Quick Setup Commands

### Option 1: Using Command Line (Linux/Mac/Git Bash)
```bash
# Create directory structure
mkdir -p models/{vggt,fruit_detection/{acfr,minneapple},health_classification,growth_classification}
mkdir -p data/{raw,processed,interim,external,acfr_orchard,minneapple}
mkdir -p assets/{demo_images,3d_models,documentation,visualizations}
mkdir -p logs outputs

# Create .gitkeep files
find models data assets logs outputs -type d -exec touch {}/.gitkeep \;
```

### Option 2: Using Windows Command Prompt
```cmd
# Create directories
mkdir models\vggt
mkdir models\fruit_detection\acfr
mkdir models\fruit_detection\minneapple
mkdir models\health_classification
mkdir models\growth_classification
mkdir data\raw
mkdir data\processed
mkdir data\interim
mkdir data\external
mkdir data\acfr_orchard
mkdir data\minneapple
mkdir assets\demo_images
mkdir assets\3d_models
mkdir assets\documentation
mkdir assets\visualizations
mkdir logs
mkdir outputs

# Create .gitkeep files (run this in each directory)
type nul > .gitkeep
```

### Option 3: Using Python Script
```python
import os
from pathlib import Path

# Directory structure
directories = [
    'models/vggt',
    'models/fruit_detection/acfr',
    'models/fruit_detection/minneapple',
    'models/health_classification',
    'models/growth_classification',
    'data/raw',
    'data/processed',
    'data/interim',
    'data/external',
    'data/acfr_orchard',
    'data/minneapple',
    'assets/demo_images',
    'assets/3d_models',
    'assets/documentation',
    'assets/visualizations',
    'logs',
    'outputs'
]

# Create directories and .gitkeep files
for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / '.gitkeep').touch()

print("Directory structure created successfully!")
```

## Directory Usage Guidelines

### Models Directory
- **Purpose**: Store pre-trained and custom-trained model files
- **File Types**: .pth, .pt, .bin, .safetensors model files
- **Size Considerations**: Models are typically large (100MB-2GB+)
- **Git Handling**: Excluded via .gitignore, use Git LFS if needed

### Data Directory Structure

#### `data/raw/`
- Original datasets as downloaded
- Never modify files in this directory
- Serves as backup and reference

#### `data/processed/`
- Cleaned, preprocessed training data
- Normalized images, augmented datasets
- Ready for model training

#### `data/interim/`
- Temporary processing outputs
- Can be safely deleted and regenerated
- Used during data pipeline execution

#### `data/external/`
- Third-party datasets and references
- Public datasets from other sources
- Benchmark datasets for comparison

### Assets Directory Usage

#### `assets/demo_images/`
- Sample plant images for testing
- Representative examples for documentation
- Quick verification of system functionality

#### `assets/3d_models/`
- Example 3D reconstructions
- Demonstration outputs
- Reference models for comparison

#### `assets/documentation/`
- Screenshots of the interface
- Workflow diagrams
- Architecture illustrations

#### `assets/visualizations/`
- Analysis output examples
- Charts and graphs
- Result visualizations

## Git Integration Best Practices

### .gitignore Configuration
Ensure your .gitignore excludes large files:
```gitignore
# Model files
models/**/*.pth
models/**/*.pt
models/**/*.bin
models/**/*.safetensors

# Large datasets
data/raw/**/*
data/processed/**/*
!data/**/.gitkeep

# Outputs and logs
outputs/**/*
logs/**/*
!outputs/.gitkeep
!logs/.gitkeep
```

### Git LFS Setup (for large files)
```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
```

## Populating Directories

### Downloading Models
Use the provided script:
```bash
python scripts/download_models.py --all
```

### Setting Up Datasets
Use the dataset setup script:
```bash
python scripts/setup_datasets.py --dataset all
```

### Adding Demo Assets
Copy sample images to appropriate directories:
```bash
cp your_plant_images/* assets/demo_images/
```

## Maintenance and Organization

### Regular Cleanup
- Clear interim data periodically
- Archive old outputs
- Monitor disk space usage

### Directory Size Monitoring
```bash
# Check directory sizes
du -sh models/ data/ assets/ logs/ outputs/
```

### Backup Strategies
- Models: Use cloud storage or Git LFS
- Data: Backup raw datasets separately
- Outputs: Archive important results

This structure provides a professional, scalable organization for the BioViT3R-Beta repository while maintaining Git compatibility and supporting both development and production workflows.