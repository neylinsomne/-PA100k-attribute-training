# PA-100k Attribute Training with PyTorch + CUDA
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.8 support (nightly for RTX 50xx)
RUN pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install additional dependencies
RUN pip3 install \
    numpy \
    pillow \
    scipy \
    tqdm \
    matplotlib \
    tensorboard

# Create app directory
WORKDIR /workspace

# Copy training script
COPY train_pytorch.py /workspace/
COPY paddle_format/ /workspace/paddle_format/

# Default command
CMD ["python3", "train_pytorch.py"]
