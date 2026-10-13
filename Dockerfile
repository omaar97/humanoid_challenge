# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# 1. System packages for Python, headless MuJoCo (EGL + OSMesa fallback), OpenCV, and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libegl1 \
    libgl1 \
    libgles2 \
    libglvnd0 \
    libosmesa6-dev \
    libglfw3 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 2. Upgrade pip and install pinned core Python dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
        "numpy==1.26.4" \
        torch torchvision torchaudio \
        "lerobot[smolvla]==0.4.4" \
        "transformers==4.57.6" \
        num2words \
        "mujoco==3.15.0" \
        "mediapipe==0.10.21" \
        opencv-contrib-python \
        scipy \
        pandas \
        pyarrow \
        huggingface_hub

# Pre-cache base SmolVLM processor/config needed by SmolVLA so offline runs work out-of-the-box
RUN python3 -c "from transformers import AutoConfig, AutoProcessor; AutoConfig.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct'); AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')" || true

# 3. Install SAM 2 globally into site-packages (skipping custom CUDA kernels via SAM2_BUILD_CUDA=0)
# Cloned to /opt/sam2 and installed directly into site-packages so it is never masked by /workspace volume mounts
RUN git clone --depth 1 https://github.com/facebookresearch/sam2.git /opt/sam2 && \
    cd /opt/sam2 && \
    SAM2_BUILD_CUDA=0 python3 -m pip install .

# 4. Copy project code, Franka FR3 MJCF assets, and calibrated camera extrinsics
COPY franka_fr3/ /workspace/franka_fr3/
COPY camera_extrinsics.npz README.md /workspace/
COPY src/ /workspace/src/

# 5. Optional build-time download of Hugging Face model, videos, and/or LeRobot dataset
ARG DOWNLOAD_MODEL=false
ARG DOWNLOAD_VIDEOS=false
ARG DOWNLOAD_DATASET=false

RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
        python3 /workspace/src/download_assets.py --model; \
    fi && \
    if [ "$DOWNLOAD_VIDEOS" = "true" ]; then \
        python3 /workspace/src/download_assets.py --videos; \
    fi && \
    if [ "$DOWNLOAD_DATASET" = "true" ]; then \
        python3 /workspace/src/download_assets.py --dataset; \
    fi

# Default command: run closed-loop MuJoCo evaluation (or override at `docker run`)
CMD ["python3", "src/mujoco_main_test_env.py"]

