FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
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

WORKDIR /home/bcc

# Same DMV-CNN / NYU source commit used by the metarepository image.
RUN git clone https://github.com/nyukat/breast_cancer_classifier.git breast_cancer_classifier \
    && cd breast_cancer_classifier \
    && git reset --hard de2b0855d02984df0f516008bb4513ff71460e21

# PyTorch 2.7.1 + CUDA 12.8 runtime for NVIDIA Blackwell.
# The source commit and pretrained checkpoint files remain unchanged.
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
       tqdm==4.66.5

# Preserve the metarepository compatibility patch that ensures heatmap output
# directories exist. It is relevant only when USE_HEATMAPS=True; the current
# thesis core smoke path uses the upstream image-only classifier.
RUN sed -i "179i \\\ \\\ \\\ \\\ folders = heatmap_save_path_malignant.split('/')\\n    dirs = folders[:-1]\\n    dir_path = '/'.join(dirs)\\n    os.makedirs(dir_path, exist_ok=True)\\n" /home/bcc/breast_cancer_classifier/src/heatmaps/run_producer.py \
    && sed -i "190i \\\ \\\ \\\ \\\ folders = heatmap_save_path_benign.split('/')\\n    dirs = folders[:-1]\\n    dir_path = '/'.join(dirs)\\n    os.makedirs(dir_path, exist_ok=True)\\n" /home/bcc/breast_cancer_classifier/src/heatmaps/run_producer.py

# Narrow PyTorch API compatibility patches. These change device capability
# checks only; they do not alter the architecture, learned weights or score
# aggregation semantics.
RUN sed -i 's/torch.has_cudnn/torch.backends.cudnn.is_available()/g' \
      /home/bcc/breast_cancer_classifier/src/modeling/run_model.py \
    && sed -i 's/torch.has_cudnn/torch.backends.cudnn.is_available()/g' \
      /home/bcc/breast_cancer_classifier/src/heatmaps/run_producer.py

RUN mkdir -p /home/predictions && chmod 777 /home/predictions
WORKDIR /home/bcc/breast_cancer_classifier
