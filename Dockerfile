# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

# Install additional GPU-specific packages
RUN pip install open3d>=0.16.0

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p logs outputs models data assets

# Download models (if download script exists)
RUN if [ -f "scripts/download_models.py" ]; then python scripts/download_models.py; fi

# Set up environment
RUN chmod +x scripts/*.py 2>/dev/null || true

# Expose port for Gradio
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default command
CMD ["python", "app.py"]