ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1
FROM ${BASE_IMAGE}

ARG SAM3_REF=46957e47805eaa273f4aa7bbbd25a88bca9108ce

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

# Keep the exact source in the image so the pipeline adapter can introspect the
# installed SAM3 API instead of relying on assumed signatures.
RUN mkdir -p /opt/sam3 \
    && curl -fsSL "https://github.com/facebookresearch/sam3/archive/${SAM3_REF}.tar.gz" \
        | tar -xz -C /opt/sam3 --strip-components=1 \
    && python -m pip install -e /opt/sam3

WORKDIR /workspace/perception

CMD ["python", "tools/verify_environment.py"]
