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

# The mammography metarepository canonical data contract supplies only breast-level
# malignant labels. GMIC's standalone runner additionally tries to copy benign
# labels into its output CSV *after* inference. Do not invent benign ground truth:
# when those optional keys are absent, preserve the malignant labels and emit NaN
# for benign_label metadata. This does not alter the model input, forward pass,
# saliency maps, predictions, architecture or weights.
RUN python - <<'PY'
from pathlib import Path
p = Path('/home/gmic/GMIC/src/scripts/run_model.py')
t = p.read_text()
replacements = {
    'return cancer_label["left_benign"], cancer_label["left_malignant"]':
        'return cancer_label.get("left_benign", np.nan), cancer_label["left_malignant"]',
    'return cancer_label["right_benign"], cancer_label["right_malignant"]':
        'return cancer_label.get("right_benign", np.nan), cancer_label["right_malignant"]',
}
for old, new in replacements.items():
    if old not in t:
        raise SystemExit(f'GMIC label-contract patch anchor missing: {old}')
    t = t.replace(old, new, 1)
p.write_text(t)
PY
RUN grep -F 'cancer_label.get("left_benign", np.nan)' /home/gmic/GMIC/src/scripts/run_model.py \
    && grep -F 'cancer_label.get("right_benign", np.nan)' /home/gmic/GMIC/src/scripts/run_model.py

# Preserve PyTorch 1.1 integer-index semantics used by GMIC ROI proposal.
# In modern PyTorch, `/` on integer tensors performs true division and can leave
# a tiny negative remainder in max_idx_y. The historical code expected integer
# quotient/remainder coordinates. This patch changes only index arithmetic; model
# architecture, weights, saliency computation and crop-selection objective are unchanged.
RUN sed -i 's|max_idx_x = max_linear_idx / W_map|max_idx_x = torch.div(max_linear_idx, W_map, rounding_mode="floor")|' \
      /home/gmic/GMIC/src/utilities/tools.py \
    && grep -F 'max_idx_x = torch.div(max_linear_idx, W_map, rounding_mode="floor")' \
      /home/gmic/GMIC/src/utilities/tools.py

RUN mkdir -p /home/predictions && chmod 777 /home/predictions
WORKDIR /home/gmic/GMIC
