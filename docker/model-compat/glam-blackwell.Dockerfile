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

WORKDIR /home/glam

# Same GLAM source commit used by the NYU metarepository image.
RUN git clone https://github.com/nyukat/GLAM.git GLAM \
    && cd GLAM \
    && git reset --hard 17a0019860441e2ea8d7b7c7e0aaeada735e871f

# PyTorch 2.7.1 + CUDA 12.8 runtime for NVIDIA Blackwell.
# The source commit, pretrained checkpoints and model architecture remain unchanged.
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

# Narrow compatibility patches required to execute the pinned GLAM implementation
# on a current headless PyTorch runtime. They preserve architecture/weights and,
# where framework semantics changed, restore the historical inference behavior.
RUN sed -i "s/matplotlib.use(\"TkAgg\")/matplotlib.use(\"Agg\")/" \
      /home/glam/GLAM/src/scripts/run_model.py \
    && sed -i "/assert torch.__version__ == '1.1.0'/c\\# Thesis runtime compatibility: validated PyTorch version is declared by the container profile." \
      /home/glam/GLAM/src/scripts/run_model.py \
    && sed -i 's/torch.has_cudnn/torch.backends.cudnn.is_available()/g' \
      /home/glam/GLAM/src/scripts/run_model.py \
      /home/glam/GLAM/src/modeling/glam.py \
    && sed -i 's/cam_combined = torch.zeros((batch_size, 2, H, W))/cam_combined = torch.zeros((batch_size, 2, H, W), device=camlocal.device)/' \
      /home/glam/GLAM/src/modeling/glam.py \
    && sed -i 's/output = torch.ones((batch_size, num_crops, crop_h, crop_w))/output = torch.ones((batch_size, num_crops, crop_h, crop_w), device=x_original_pytorch.device)/' \
      /home/glam/GLAM/src/modeling/glam.py \
    && sed -i 's/max_idx_x = max_linear_idx \/ W_map/max_idx_x = torch.div(max_linear_idx, W_map, rounding_mode="floor")/' \
      /home/glam/GLAM/src/modeling/common_functions.py \
    && sed -i 's/F.grid_sample(original_img_pytorch, grid)/F.grid_sample(original_img_pytorch, grid, align_corners=True)/' \
      /home/glam/GLAM/src/modeling/common_functions.py

RUN mkdir -p /home/predictions && chmod 777 /home/predictions
WORKDIR /home/glam/GLAM
