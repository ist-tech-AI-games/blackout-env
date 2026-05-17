# Python 3.10 required by mlagents-envs 1.1.0
# Ubuntu 22.04 ships Python 3.10 as default
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
# libglib2.0-0: Unity runtime
# libgomp1: numpy / ML libs
# xvfb: virtual display (required even with --no-graphics for RenderTexture offscreen)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libglib2.0-0 \
    libgomp1 \
    xvfb \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3       1

WORKDIR /workspace

COPY . /blackout-env
# mlagents-envs declares pettingzoo==1.15.0 which conflicts with blackout-env's pettingzoo>=1.24.0.
# Install it with --no-deps, then manually install its other required dependencies.
RUN pip install --no-cache-dir "mlagents-envs==1.1.0" --no-deps \
 && pip install --no-cache-dir \
    cloudpickle \
    "grpcio>=1.11.0,<=1.48.2" \
    "Pillow>=4.2.1" \
    "protobuf>=3.6,<3.21" \
    "pyyaml>=3.1.0" \
    "gym>=0.21.0" \
    "filelock>=3.4.0" \
 && pip install --no-cache-dir /blackout-env \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Entrypoint: wrap with Xvfb so Unity RenderTexture offscreen works
# Usage: docker run ... python train.py
ENTRYPOINT ["xvfb-run", "--auto-servernum", "--server-args=-screen 0 1280x720x24"]
CMD ["python", "train.py"]
