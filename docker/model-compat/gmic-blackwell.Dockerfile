FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    zlib1g-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/gmic

# Same GMIC source commit used by the NYU metarepository image.
RUN git clone https://github.com/nyukat/GMIC.git GMIC \
    && cd GMIC \
    && git reset --hard 3bf4ce81dfa40553f108c8bfaf03bf006e082761

# Preserve the same metarepository compatibility edit used by the legacy image.
RUN sed -i "149i \\                if len(datum[view]) == 0: continue" /home/gmic/GMIC/src/scripts/run_model.py

# PyTorch 2.7.1 + CUDA 12.8 is the first stable PyTorch generation with
# official NVIDIA Blackwell support. The model source/checkpoints are unchanged.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
       torch==2.7.1 torchvision==0.22.1 \
       --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install \
       h5py==3.8.0 \
       imageio==2.34.2 \
       numpy==1.23.5 \
       opencv-python-headless==4.10.0.84 \
       pandas==1.5.3 \
       scipy==1.10.1 \
       tqdm==4.66.5 \
       matplotlib==3.7.5

# Narrow PyTorch API compatibility patch. It does not change the network
# architecture, checkpoint, preprocessing rules or learned parameters.
RUN sed -i 's/torch.has_cudnn/torch.backends.cudnn.is_available()/g' /home/gmic/GMIC/src/scripts/run_model.py

RUN mkdir -p /home/predictions && chmod 777 /home/predictions
WORKDIR /home/gmic/GMIC
