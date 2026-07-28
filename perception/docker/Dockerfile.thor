ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1
FROM ${BASE_IMAGE}

ARG SAM3_REF=46957e47805eaa273f4aa7bbbd25a88bca9108ce
ARG SAM2_REF=2b90b9f5ceec907a1c18123530e92e794ad901a4
ARG EFFICIENTTAM_REF=abcd061ebd3cc6e7527d152d75b890126aaa53f6
ARG ULTRALYTICS_VERSION=8.4.108

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        liblcm-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install --upgrade pip "setuptools<81" wheel \
    && python -m pip install -r /tmp/requirements-runtime.txt

# Ultralytics is installed without dependency resolution so pip cannot replace
# the CUDA 13.3 PyTorch/torchvision packages supplied by NVIDIA's Thor image.
# Its explicit runtime dependencies are in requirements-runtime.txt above.
RUN python -m pip install --no-deps "ultralytics==${ULTRALYTICS_VERSION}"

# Keep the exact source in the image so the pipeline adapter can introspect the
# installed SAM3 API instead of relying on assumed signatures.
RUN mkdir -p /opt/sam3 \
    && curl -fsSL "https://github.com/facebookresearch/sam3/archive/${SAM3_REF}.tar.gz" \
        | tar -xz -C /opt/sam3 --strip-components=1 \
    && python -m pip install -e /opt/sam3

# SAM2.1 Tiny is the generic-mask/video-memory replacement for
# EfficientViT-SAM-L0.  Skip SAM2's optional custom CUDA extension: Thor's
# CUDA 13.3 / PyTorch nightly combination is intentionally kept untouched and
# the extension only improves small-mask post-processing, not core inference.
RUN mkdir -p /opt/sam2 \
    && curl -fsSL "https://github.com/facebookresearch/sam2/archive/${SAM2_REF}.tar.gz" \
        | tar -xz -C /opt/sam2 --strip-components=1 \
    && SAM2_BUILD_CUDA=0 python -m pip install --no-deps --no-build-isolation -e /opt/sam2

# EfficientTAM follows SAM2's API and is installed separately so SAM2/SAM3
# remain selectable. Its optional CUDA post-processing extension is skipped
# for the same Thor CUDA/PyTorch compatibility reason as SAM2.
RUN mkdir -p /opt/efficienttam \
    && curl -fsSL "https://github.com/yformer/EfficientTAM/archive/${EFFICIENTTAM_REF}.tar.gz" \
        | tar -xz -C /opt/efficienttam --strip-components=1 \
    && Efficient_Track_Anything_BUILD_CUDA=0 python -m pip install --no-deps --no-build-isolation -e /opt/efficienttam

WORKDIR /workspace/perception

CMD ["python", "tools/verify_environment.py"]
