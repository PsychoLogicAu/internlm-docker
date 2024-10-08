# Step 1: Base image for dependencies
FROM continuumio/miniconda3:latest AS base

# Specify Python version and set environment variables
ENV PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

# Update and install essential packages
RUN apt-get update && \
    apt-get install -y bash curl git wget software-properties-common

# Create Conda environment
RUN conda create --name conda python=${PYTHON_VERSION} pip

# Upgrade pip
RUN /opt/conda/envs/conda/bin/pip install --no-cache-dir --upgrade pip

# Activate Conda environment and install necessary packages
# RUN /opt/conda/bin/conda run -n conda pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
#     --extra-index-url https://download.pytorch.org/whl/cu118
RUN /opt/conda/bin/conda run -n conda pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
RUN /opt/conda/bin/conda run -n conda pip install --no-cache-dir transformers einops "numpy<2"

# Step 2: Final image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set non-interactive frontend
ARG DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && \
    apt-get install -y bash curl git wget software-properties-common && \
    apt-get install -y libgl1 gnupg2 moreutils tk libglib2.0-0 libaio-dev && \
    apt-get install -y unzip

# Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

# Create user and set permissions
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME

# Copy Conda environment from dependencies stage
COPY --from=base /opt/conda /opt/conda

ARG PIP_EXTRA
ENV PIP_EXTRA=$PIP_EXTRA
# if PIP_EXTRA is not empty, install the extra packages
RUN test -z "$PIP_EXTRA" || /opt/conda/bin/conda run -n conda pip install --no-cache-dir ${PIP_EXTRA}

# Set environment variables
ENV PATH /opt/conda/bin:$PATH
ENV HF_HOME=/data/.cache/huggingface

# Skip downloading large files with Git LFS
ENV GIT_LFS_SKIP_SMUDGE=1

WORKDIR /app/

ARG GIT_DOMAIN
ARG GIT_ORG
ARG GIT_REPO
ARG CACHEBUST=1
RUN git clone https://${GIT_DOMAIN}/${GIT_ORG}/${GIT_REPO}.git

WORKDIR /app/${GIT_REPO}

ARG GIT_BRANCH
RUN test -z "$GIT_BRANCH" || git checkout $GIT_BRANCH

ARG EXAMPLE_FILE
COPY ${EXAMPLE_FILE} ./example.py

# Set environment variables for PyTorch CUDA
# https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
# ENV PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
ENV PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:128"
# expandable_segments:True <-- "RuntimeError: Unrecognized CachingAllocator option: expandable_segments", need newer PyTorch?
# ENV PYTORCH_CUDA_ALLOC_CONF="backend:native,max_split_size_mb:128,roundup_power2_divisions:[256:1,512:2,1024:4,>:8],garbage_collection_threshold:0.8"

# Set entrypoint to activate Conda environment and start bash shell
ENTRYPOINT ["/bin/bash", "-c", "source activate conda && /bin/bash -c 'python example.py'"]
