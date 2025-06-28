# BioViT3R-Beta Setup Configuration
# PyPI packaging and installation configuration

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from package
def get_version():
    """Extract version from package __init__.py"""
    try:
        with open('src/__init__.py', 'r') as f:
            version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
            if version_match:
                return version_match.group(1)
    except FileNotFoundError:
        pass
    return "1.0.0-beta"

# Read requirements from requirements.txt
def get_requirements():
    """Parse requirements from requirements.txt"""
    requirements = []
    try:
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Handle version constraints
                    if '>=' in line or '==' in line or '~=' in line:
                        requirements.append(line)
                    else:
                        # Add basic requirement without version if not specified
                        pkg_name = line.split('#')[0].strip()
                        if pkg_name:
                            requirements.append(pkg_name)
    except FileNotFoundError:
        # Fallback to essential requirements if file not found
        requirements = [
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            'opencv-python>=4.8.0',
            'numpy>=1.24.0',
            'gradio>=3.40.0',
            'open3d>=0.17.0',
            'trimesh>=3.21.0',
            'matplotlib>=3.7.0',
            'plotly>=5.15.0',
            'pyyaml>=6.0',
            'scikit-learn>=1.3.0',
            'Pillow>=10.0.0'
        ]
    return requirements

setup(
    name="biovit3r-beta",
    version=get_version(),
    author="BioViT3R Development Team",
    author_email="contact@biovit3r.ai",
    description="AI-Powered Plant Analysis with VGGT 3D Reconstruction and Agricultural Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/BioViT3R-Beta",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/BioViT3R-Beta/issues",
        "Source": "https://github.com/yourusername/BioViT3R-Beta",
        "Documentation": "https://biovit3r-beta.readthedocs.io/",
        "Demo": "https://huggingface.co/spaces/biovit3r/demo"
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Framework :: Gradio",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0"
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",
            "torch[cuda]>=2.0.0"
        ],
        "full": [
            "jupyter>=1.0.0",
            "wandb>=0.15.0",
            "mlflow>=2.5.0",
            "ray>=2.5.0",
            "streamlit>=1.25.0"
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.23.0",
            "myst-parser>=2.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "biovit3r=main:main",
            "biovit3r-cli=main:main",
            "biovit3r-app=app:main",
        ]
    },
    include_package_data=True,
    package_data={
        "": [
            "configs/*.yaml",
            "configs/*.yml", 
            "configs/*.json",
            "assets/demo_images/*",
            "docs/*.md",
            "scripts/*.py"
        ]
    },
    zip_safe=False,
    keywords=[
        "computer-vision",
        "agriculture", 
        "3d-reconstruction",
        "plant-analysis",
        "ai",
        "machine-learning",
        "fruit-detection",
        "biomass-estimation",
        "vggt",
        "gradio",
        "agricultural-ai"
    ],
    # Platform-specific installations
    platforms=["any"],
    # Custom commands for development
    cmdclass={},
    # Data files to include
    data_files=[
        ("configs", ["configs/app_config.yaml", "configs/vggt_config.yaml"]),
        ("scripts", ["scripts/download_models.py", "scripts/setup_datasets.py"]),
        ("docs", ["README.md", "LICENSE"])
    ]
)

# Additional setup for development environment
if __name__ == "__main__":
    print("BioViT3R-Beta Setup Configuration")
    print("==================================")
    print(f"Version: {get_version()}")
    print(f"Python Requirements: {len(get_requirements())} packages")
    print("Run 'pip install -e .' for development installation")
    print("Run 'pip install .' for standard installation")
    print()
    print("Available extras:")
    print("  - dev: Development tools and testing")
    print("  - gpu: GPU acceleration with CUDA")
    print("  - full: Complete feature set with all optional dependencies")
    print("  - docs: Documentation building tools")
    print()
    print("Installation examples:")
    print("  pip install biovit3r-beta[dev]     # Development environment")
    print("  pip install biovit3r-beta[gpu]     # GPU-accelerated version")
    print("  pip install biovit3r-beta[full]    # All features")
    print("  pip install biovit3r-beta[dev,gpu] # Development + GPU")