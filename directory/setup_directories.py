#!/usr/bin/env python3
"""
BioViT3R-Beta Directory Structure Setup Script

This script creates the complete directory structure for the BioViT3R-Beta repository
including all necessary .gitkeep files for Git version control.
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the complete BioViT3R-Beta directory structure."""
    
    # Define the directory structure
    directories = [
        # Models directory structure
        'models/vggt',
        'models/fruit_detection/acfr',
        'models/fruit_detection/minneapple',
        'models/health_classification',
        'models/growth_classification',
        
        # Data directory structure
        'data/raw',
        'data/processed',
        'data/interim',
        'data/external',
        'data/acfr_orchard',
        'data/minneapple',
        
        # Assets directory structure
        'assets/demo_images',
        'assets/3d_models',
        'assets/documentation',
        'assets/visualizations',
        
        # Runtime directories
        'logs',
        'outputs'
    ]
    
    print("🚀 Setting up BioViT3R-Beta directory structure...")
    print(f"📁 Creating {len(directories)} directories...")
    
    created_dirs = 0
    created_gitkeep = 0
    
    for directory in directories:
        # Create directory
        dir_path = Path(directory)
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs += 1
            print(f"  ✅ Created: {directory}")
            
            # Create .gitkeep file
            gitkeep_path = dir_path / '.gitkeep'
            if not gitkeep_path.exists():
                gitkeep_path.write_text("# Keep this directory in Git\n")
                created_gitkeep += 1
                print(f"     📄 Added .gitkeep")
            else:
                print(f"     ⚠️  .gitkeep already exists")
                
        except OSError as e:
            print(f"  ❌ Failed to create {directory}: {e}")
    
    print(f"\n🎉 Setup complete!")
    print(f"   📁 Directories created: {created_dirs}")
    print(f"   📄 .gitkeep files created: {created_gitkeep}")
    
    # Verify structure
    print("\n📋 Verifying directory structure:")
    for directory in directories:
        if Path(directory).exists():
            gitkeep_exists = (Path(directory) / '.gitkeep').exists()
            status = "✅" if gitkeep_exists else "⚠️"
            print(f"  {status} {directory}")
        else:
            print(f"  ❌ {directory}")

def create_gitignore_suggestions():
    """Display .gitignore suggestions for manual addition."""
    
    print("\n📝 Recommended .gitignore additions:")
    print("-" * 50)
    
    gitignore_suggestions = [
        "# BioViT3R-Beta specific gitignore additions",
        "",
        "# Model files (exclude large model weights)",
        "models/**/*.pth",
        "models/**/*.pt", 
        "models/**/*.bin",
        "models/**/*.safetensors",
        "models/**/*.pkl",
        "models/**/*.h5",
        "models/**/*.onnx",
        "",
        "# Large datasets (exclude but keep structure)",
        "data/raw/**/*",
        "data/processed/**/*",
        "data/interim/**/*",
        "data/external/**/*",
        "data/acfr_orchard/**/*",
        "data/minneapple/**/*",
        "!data/**/.gitkeep",
        "",
        "# Runtime outputs",
        "outputs/**/*",
        "!outputs/.gitkeep",
        "",
        "# Logs",
        "logs/**/*", 
        "!logs/.gitkeep",
        "",
        "# Temporary files",
        "*.tmp",
        "*.temp",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    for line in gitignore_suggestions:
        print(line)
    
    print("-" * 50)
    print("Copy the above content and add it to your .gitignore file")

def main():
    """Main function to set up the directory structure."""
    
    print("BioViT3R-Beta Directory Setup Script")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path('src').exists():
        print("⚠️  Warning: 'src' directory not found.")
        print("   Make sure you're running this script from the repository root.")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("   Exiting...")
            sys.exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Display .gitignore suggestions
    create_gitignore_suggestions()
    
    print("\n📚 Next Steps:")
    print("1. Add the suggested .gitignore content above")
    print("2. Run 'git add .' to stage the new directories")
    print("3. Run 'git commit -m \"Add directory structure with .gitkeep files\"'")
    print("4. Use 'python scripts/download_models.py' to populate models directory")
    print("5. Use 'python scripts/setup_datasets.py' to download datasets")
    print("6. Add demo images to 'assets/demo_images/' directory")
    
    print("\n✨ Happy coding with BioViT3R-Beta!")

if __name__ == "__main__":
    main()