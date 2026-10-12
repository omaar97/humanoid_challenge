# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl

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

# 2. Upgrade pip and install core Python dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
        torch torchvision torchaudio \
        "lerobot[smolvla]" \
        num2words \
        mujoco \
        mediapipe \
        opencv-contrib-python \
        scipy \
        pandas \
        pyarrow \
        huggingface_hub

# 3. Optional: Install SAM 2 + checkpoint for Step 2 (src/process_hand_segmentation.py)
ARG INSTALL_SAM2=false
RUN if [ "$INSTALL_SAM2" = "true" ]; then \
        git clone --depth 1 https://github.com/facebookresearch/sam2.git /workspace/sam2_repo && \
        python3 -m pip install -e /workspace/sam2_repo && \
        curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
            -o /workspace/sam2.1_hiera_tiny.pt; \
    fi

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

# Default command: run the physics verification & scripted demos (or override at `docker run`)
CMD ["python3", "src/mujoco_test_env.py"]

