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
# Note: LeRobot 0.4.4 pulls rerun-sdk which declares numpy>=2, conflicting with
# mediapipe 0.10.21 and SAM 2 which require numpy<2 (pinned to numpy==1.26.4).
# Installing all underlying dependencies first and installing rerun-sdk & lerobot
# with --no-deps resolves this conflict cleanly while keeping full functionality.
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
        "numpy==1.26.4" \
        torch torchvision torchaudio \
        "transformers==4.57.6" \
        num2words \
        "mujoco==3.15.0" \
        "mediapipe==0.10.21" \
        "opencv-contrib-python==4.11.0.86" \
        "opencv-python-headless==4.11.0.86" \
        scipy \
        pandas \
        pyarrow \
        huggingface_hub \
        datasets \
        diffusers \
        accelerate \
        einops \
        av \
        jsonlines \
        draccus \
        gymnasium \
        torchcodec \
        deepdiff \
        imageio \
        termcolor && \
    python3 -m pip install --no-deps "rerun-sdk==0.26.2" "lerobot==0.4.4"

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

# 5. Build-time automatic download of raw videos from Hugging Face for pipeline reproduction
# Stored in /workspace/media and cached to /opt/media so host workspace volume mounts do not mask them
ARG DOWNLOAD_RAW_VIDEOS=true
ARG DOWNLOAD_MODEL=false
ARG DOWNLOAD_ALL_VIDEOS=false
ARG DOWNLOAD_DATASET=false

RUN if [ "$DOWNLOAD_RAW_VIDEOS" = "true" ]; then \
        python3 /workspace/src/download_assets.py --raw-videos --media-dir /workspace/media && \
        mkdir -p /opt/media && \
        cp -r /workspace/media/* /opt/media/ 2>/dev/null || true; \
    fi && \
    if [ "$DOWNLOAD_MODEL" = "true" ]; then \
        python3 /workspace/src/download_assets.py --model; \
    fi && \
    if [ "$DOWNLOAD_ALL_VIDEOS" = "true" ]; then \
        python3 /workspace/src/download_assets.py --videos --media-dir /workspace/media && \
        mkdir -p /opt/media && \
        cp -r /workspace/media/* /opt/media/ 2>/dev/null || true; \
    fi && \
    if [ "$DOWNLOAD_DATASET" = "true" ]; then \
        python3 /workspace/src/download_assets.py --dataset; \
    fi

# Default command: run closed-loop MuJoCo evaluation (or override at `docker run`)
CMD ["python3", "src/mujoco_main_test_env.py"]

