#!/bin/bash

# BioViT3R-Beta Directory Structure Setup Script
# This script creates all necessary directories and .gitkeep files

echo "🚀 Setting up BioViT3R-Beta directory structure..."

# Create all directories at once
echo "📁 Creating directory structure..."

# Models directories
mkdir -p models/vggt
mkdir -p models/fruit_detection/acfr
mkdir -p models/fruit_detection/minneapple
mkdir -p models/health_classification
mkdir -p models/growth_classification

# Data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/interim
mkdir -p data/external
mkdir -p data/acfr_orchard
mkdir -p data/minneapple

# Assets directories
mkdir -p assets/demo_images
mkdir -p assets/3d_models
mkdir -p assets/documentation
mkdir -p assets/visualizations

# Runtime directories
mkdir -p logs
mkdir -p outputs

echo "✅ Directory structure created!"

# Create .gitkeep files in all directories
echo "📄 Creating .gitkeep files..."

# Find all directories and create .gitkeep files
find models data assets logs outputs -type d -exec touch {}/.gitkeep \;

echo "✅ .gitkeep files created!"

# Create .gitignore additions file
echo "📝 Creating .gitignore suggestions..."

cat > gitignore_additions.txt << 'EOF'
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

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so

# Virtual environments
venv/
env/
.venv/
.env/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# Jupyter checkpoints
.ipynb_checkpoints/
EOF

echo "✅ Created gitignore_additions.txt"

# Verify the structure
echo ""
echo "📋 Verifying directory structure:"
echo "Models directories:"
ls -la models/
echo ""
echo "Data directories:"
ls -la data/
echo ""
echo "Assets directories:"
ls -la assets/
echo ""
echo "Runtime directories:"
ls -la logs/ outputs/

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📚 Next Steps:"
echo "1. Add the content from 'gitignore_additions.txt' to your .gitignore file"
echo "2. Run 'git add .' to stage the new directories"
echo "3. Run 'git commit -m \"Add directory structure with .gitkeep files\"'"
echo "4. Use 'python scripts/download_models.py' to populate models directory"
echo "5. Use 'python scripts/setup_datasets.py' to download datasets"
echo ""
echo "✨ Happy coding with BioViT3R-Beta!"