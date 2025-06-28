@echo off
REM BioViT3R-Beta Directory Structure Setup Script for Windows
REM This script creates all necessary directories and .gitkeep files

echo 🚀 Setting up BioViT3R-Beta directory structure...

REM Create all directories
echo 📁 Creating directory structure...

REM Models directories
mkdir models\vggt 2>nul
mkdir models\fruit_detection\acfr 2>nul
mkdir models\fruit_detection\minneapple 2>nul
mkdir models\health_classification 2>nul
mkdir models\growth_classification 2>nul

REM Data directories
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\interim 2>nul
mkdir data\external 2>nul
mkdir data\acfr_orchard 2>nul
mkdir data\minneapple 2>nul

REM Assets directories
mkdir assets\demo_images 2>nul
mkdir assets\3d_models 2>nul
mkdir assets\documentation 2>nul
mkdir assets\visualizations 2>nul

REM Runtime directories
mkdir logs 2>nul
mkdir outputs 2>nul

echo ✅ Directory structure created!

REM Create .gitkeep files
echo 📄 Creating .gitkeep files...

REM Models .gitkeep files
echo # Keep this directory in Git > models\vggt\.gitkeep
echo # Keep this directory in Git > models\fruit_detection\acfr\.gitkeep
echo # Keep this directory in Git > models\fruit_detection\minneapple\.gitkeep
echo # Keep this directory in Git > models\health_classification\.gitkeep
echo # Keep this directory in Git > models\growth_classification\.gitkeep

REM Data .gitkeep files
echo # Keep this directory in Git > data\raw\.gitkeep
echo # Keep this directory in Git > data\processed\.gitkeep
echo # Keep this directory in Git > data\interim\.gitkeep
echo # Keep this directory in Git > data\external\.gitkeep
echo # Keep this directory in Git > data\acfr_orchard\.gitkeep
echo # Keep this directory in Git > data\minneapple\.gitkeep

REM Assets .gitkeep files
echo # Keep this directory in Git > assets\demo_images\.gitkeep
echo # Keep this directory in Git > assets\3d_models\.gitkeep
echo # Keep this directory in Git > assets\documentation\.gitkeep
echo # Keep this directory in Git > assets\visualizations\.gitkeep

REM Runtime .gitkeep files
echo # Keep this directory in Git > logs\.gitkeep
echo # Keep this directory in Git > outputs\.gitkeep

echo ✅ .gitkeep files created!

REM Create .gitignore additions file
echo 📝 Creating .gitignore suggestions...

(
echo # BioViT3R-Beta specific gitignore additions
echo.
echo # Model files ^(exclude large model weights^)
echo models/**/*.pth
echo models/**/*.pt
echo models/**/*.bin
echo models/**/*.safetensors
echo models/**/*.pkl
echo models/**/*.h5
echo models/**/*.onnx
echo.
echo # Large datasets ^(exclude but keep structure^)
echo data/raw/**/*
echo data/processed/**/*
echo data/interim/**/*
echo data/external/**/*
echo data/acfr_orchard/**/*
echo data/minneapple/**/*
echo !data/**/.gitkeep
echo.
echo # Runtime outputs
echo outputs/**/*
echo !outputs/.gitkeep
echo.
echo # Logs
echo logs/**/*
echo !logs/.gitkeep
echo.
echo # Temporary files
echo *.tmp
echo *.temp
echo .DS_Store
echo Thumbs.db
echo.
echo # Python cache
echo __pycache__/
echo *.pyc
echo *.pyo
echo *.pyd
echo .Python
echo *.so
echo.
echo # Virtual environments
echo venv/
echo env/
echo .venv/
echo .env/
echo.
echo # IDE files
echo .vscode/
echo .idea/
echo *.swp
echo *.swo
echo.
echo # Jupyter checkpoints
echo .ipynb_checkpoints/
) > gitignore_additions.txt

echo ✅ Created gitignore_additions.txt

REM Verify the structure
echo.
echo 📋 Verifying directory structure:
echo Models directories:
dir models /b
echo.
echo Data directories:
dir data /b
echo.
echo Assets directories:
dir assets /b
echo.
echo Runtime directories:
dir logs /b 2>nul
dir outputs /b 2>nul

echo.
echo 🎉 Setup complete!
echo.
echo 📚 Next Steps:
echo 1. Add the content from 'gitignore_additions.txt' to your .gitignore file
echo 2. Run 'git add .' to stage the new directories
echo 3. Run 'git commit -m "Add directory structure with .gitkeep files"'
echo 4. Use 'python scripts\download_models.py' to populate models directory
echo 5. Use 'python scripts\setup_datasets.py' to download datasets
echo.
echo ✨ Happy coding with BioViT3R-Beta!

pause