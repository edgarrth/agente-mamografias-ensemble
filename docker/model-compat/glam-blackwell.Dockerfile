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

# The metarepository batch contract supplies breast-level malignant labels.
# GLAM standalone copies optional benign labels into its CSV only after inference.
# Do not fabricate benign ground truth: preserve required malignant labels and emit
# NaN for benign_label metadata when those independent labels are unavailable.
RUN python - <<'PY'
from pathlib import Path
p = Path('/home/glam/GLAM/src/scripts/run_model.py')
t = p.read_text()
replacements = {
    'return cancer_label["left_benign"], cancer_label["left_malignant"]':
        'return cancer_label.get("left_benign", np.nan), cancer_label["left_malignant"]',
    'return cancer_label["right_benign"], cancer_label["right_malignant"]':
        'return cancer_label.get("right_benign", np.nan), cancer_label["right_malignant"]',
}
for old, new in replacements.items():
    if old not in t:
        raise SystemExit(f'GLAM label-contract patch anchor missing: {old}')
    t = t.replace(old, new, 1)
p.write_text(t)
PY
RUN grep -F 'cancer_label.get("left_benign", np.nan)' /home/glam/GLAM/src/scripts/run_model.py \
    && grep -F 'cancer_label.get("right_benign", np.nan)' /home/glam/GLAM/src/scripts/run_model.py

RUN mkdir -p /home/predictions && chmod 777 /home/predictions
WORKDIR /home/glam/GLAM
