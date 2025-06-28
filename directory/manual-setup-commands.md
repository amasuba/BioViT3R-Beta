# BioViT3R-Beta Directory Setup Commands

## Quick Setup Options

### Option 1: One-Line Command (Linux/Mac/Git Bash)
```bash
mkdir -p models/{vggt,fruit_detection/{acfr,minneapple},health_classification,growth_classification} data/{raw,processed,interim,external,acfr_orchard,minneapple} assets/{demo_images,3d_models,documentation,visualizations} logs outputs && find models data assets logs outputs -type d -exec touch {}/.gitkeep \;
```

### Option 2: Step-by-Step Commands

#### Create Directories
```bash
# Models structure
mkdir -p models/vggt
mkdir -p models/fruit_detection/acfr
mkdir -p models/fruit_detection/minneapple
mkdir -p models/health_classification
mkdir -p models/growth_classification

# Data structure
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/interim
mkdir -p data/external
mkdir -p data/acfr_orchard
mkdir -p data/minneapple

# Assets structure
mkdir -p assets/demo_images
mkdir -p assets/3d_models
mkdir -p assets/documentation
mkdir -p assets/visualizations

# Runtime directories
mkdir -p logs
mkdir -p outputs
```

#### Create .gitkeep Files
```bash
# Create all .gitkeep files at once
find models data assets logs outputs -type d -exec touch {}/.gitkeep \;
```

### Option 3: Windows Commands

#### PowerShell Commands
```powershell
# Create directories
New-Item -ItemType Directory -Force -Path "models/vggt"
New-Item -ItemType Directory -Force -Path "models/fruit_detection/acfr"
New-Item -ItemType Directory -Force -Path "models/fruit_detection/minneapple"
New-Item -ItemType Directory -Force -Path "models/health_classification"
New-Item -ItemType Directory -Force -Path "models/growth_classification"
New-Item -ItemType Directory -Force -Path "data/raw"
New-Item -ItemType Directory -Force -Path "data/processed"
New-Item -ItemType Directory -Force -Path "data/interim"
New-Item -ItemType Directory -Force -Path "data/external"
New-Item -ItemType Directory -Force -Path "data/acfr_orchard"
New-Item -ItemType Directory -Force -Path "data/minneapple"
New-Item -ItemType Directory -Force -Path "assets/demo_images"
New-Item -ItemType Directory -Force -Path "assets/3d_models"
New-Item -ItemType Directory -Force -Path "assets/documentation"
New-Item -ItemType Directory -Force -Path "assets/visualizations"
New-Item -ItemType Directory -Force -Path "logs"
New-Item -ItemType Directory -Force -Path "outputs"

# Create .gitkeep files
Get-ChildItem -Path "models","data","assets","logs","outputs" -Recurse -Directory | ForEach-Object { "# Keep this directory in Git" | Out-File -FilePath "$($_.FullName)/.gitkeep" -Encoding UTF8 }
```

#### Command Prompt (CMD)
```cmd
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

REM Create .gitkeep files manually in each directory
echo # Keep this directory in Git > models\vggt\.gitkeep
echo # Keep this directory in Git > models\fruit_detection\acfr\.gitkeep
echo # Keep this directory in Git > models\fruit_detection\minneapple\.gitkeep
REM ... (repeat for all directories)
```

## Verification Commands

### Check Directory Structure
```bash
# View the tree structure
tree models data assets logs outputs

# Or use ls if tree is not available
ls -la models/
ls -la data/
ls -la assets/
ls -la logs/
ls -la outputs/
```

### Verify .gitkeep Files
```bash
# Find all .gitkeep files
find . -name ".gitkeep" -type f

# Count .gitkeep files (should be 17)
find . -name ".gitkeep" -type f | wc -l
```

## .gitignore Content to Add

Add this content to your `.gitignore` file:

```gitignore
# BioViT3R-Beta specific gitignore additions

# Model files (exclude large model weights)
models/**/*.pth
models/**/*.pt
models/**/*.bin
models/**/*.safetensors
models/**/*.pkl
models/**/*.h5
models/**/*.onnx

# Large datasets (exclude but keep structure)
data/raw/**/*
data/processed/**/*
data/interim/**/*
data/external/**/*
data/acfr_orchard/**/*
data/minneapple/**/*
!data/**/.gitkeep

# Runtime outputs
outputs/**/*
!outputs/.gitkeep

# Logs
logs/**/*
!logs/.gitkeep

# Temporary files
*.tmp
*.temp
.DS_Store
Thumbs.db
```

## Git Commands for Repository Setup

```bash
# Initialize Git LFS for large files
git lfs install
git lfs track "*.pth" "*.pt" "*.bin" "*.safetensors"

# Add directory structure to Git
git add .

# Commit the structure
git commit -m "Add directory structure with .gitkeep files"

# Push to remote repository
git push origin main
```

## Expected Final Structure

After running the commands, you should have:

```
BioViT3R-Beta/
├── models/
│   ├── vggt/
│   │   └── .gitkeep
│   ├── fruit_detection/
│   │   ├── acfr/
│   │   │   └── .gitkeep
│   │   └── minneapple/
│   │       └── .gitkeep
│   ├── health_classification/
│   │   └── .gitkeep
│   └── growth_classification/
│       └── .gitkeep
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   ├── external/
│   │   └── .gitkeep
│   ├── acfr_orchard/
│   │   └── .gitkeep
│   └── minneapple/
│       └── .gitkeep
├── assets/
│   ├── demo_images/
│   │   └── .gitkeep
│   ├── 3d_models/
│   │   └── .gitkeep
│   ├── documentation/
│   │   └── .gitkeep
│   └── visualizations/
│       └── .gitkeep
├── logs/
│   └── .gitkeep
└── outputs/
    └── .gitkeep
```

Total: **17 directories** with **17 .gitkeep files**