FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TORCHINDUCTOR_DISABLE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# sm_61 kernels are available in the cu126 PyTorch wheel line used here.
RUN pip install --no-cache-dir \
        torch==2.8.0 torchvision \
        --index-url https://download.pytorch.org/whl/cu126

WORKDIR /workspace
