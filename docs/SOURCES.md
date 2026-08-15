# Primary technical sources

The prototype intentionally integrates official research implementations instead of reproducing model code.

- NYU Mammography Metarepository: https://github.com/nyukat/mammography_metarepository
- GMIC: https://github.com/nyukat/GMIC
- DMV-CNN / NYU Breast Cancer Classifier: https://github.com/nyukat/breast_cancer_classifier
- GLAM: https://github.com/nyukat/GLAM
- CBIS-DDSM official collection information: https://www.cancerimagingarchive.net/collection/cbis-ddsm/
- VinDr-Mammo official PhysioNet page: https://physionet.org/content/vindr-mammo/

Before definitive thesis measurements, record the exact resolved metarepository commit, Docker image IDs/digests, dataset versions and checksums in the generated workspace evidence.

Infrastructure boundary references used by v0.4:

- Docker Official Image (`docker:<version>-cli`): https://hub.docker.com/_/docker
- Docker Engine API/version compatibility: https://docs.docker.com/reference/api/engine/
- Docker Engine 29 release notes: https://docs.docker.com/engine/release-notes/29/

Legacy model Dockerfile references verified for v0.5:

- GMIC metarepository Dockerfile: https://raw.githubusercontent.com/nyukat/mammography_metarepository/master/models/nyu_gmic/Dockerfile
- DMV-CNN/NYU metarepository Dockerfile: https://raw.githubusercontent.com/nyukat/mammography_metarepository/master/models/nyu_model/Dockerfile
- GLAM metarepository Dockerfile: https://raw.githubusercontent.com/nyukat/mammography_metarepository/master/models/nyu_glam/Dockerfile
- NVIDIA CUDA/OpenGL image repository: https://hub.docker.com/r/nvidia/cudagl

## v0.6 runtime compatibility references

- NVIDIA Technical Blog, *Updating the CUDA Linux GPG Repository Key* (2022): documents NVIDIA repository signing-key rotation and the need to install/update repository keys for Debian/Ubuntu environments.
- NYU Mammography Meta-Repository `models/nyu_model/Dockerfile`: contains an upstream Ubuntu 18.04 CUDA GPG-key workaround using `3bf863cc.pub` and `7fa2af80.pub`; v0.6 reuses this pattern only when the corresponding historical GMIC/GLAM Dockerfiles lack it.

## v0.11 NYU Blackwell compatibility references

- NYU DMV-CNN source README at pinned source lineage: https://github.com/nyukat/breast_cancer_classifier — documents the original Python 3.6 / PyTorch 0.4.1 environment, four-view exam input and breast-level outputs.
- NYU metarepository Dockerfile: https://raw.githubusercontent.com/nyukat/mammography_metarepository/master/models/nyu_model/Dockerfile — pins source commit `de2b0855d02984df0f516008bb4513ff71460e21` and installs the historical PyTorch 0.4.1 CUDA 9.2 wheel.
- NYU metarepository prediction script: https://raw.githubusercontent.com/nyukat/mammography_metarepository/master/models/nyu_model/predict/predict.sh — defines the image-only smoke path used by the prototype (`USE_HEATMAPS=False`) and the same four-stage preprocessing/classification sequence.
- PyTorch 2.7 release: https://pytorch.org/blog/pytorch-2-7/ — introduces NVIDIA Blackwell support and CUDA 12.8 wheels.
- PyTorch previous versions: https://pytorch.org/get-started/previous-versions/ — records the `torch==2.7.1`, `torchvision==0.22.1`, CUDA 12.8 installation combination used by the compatibility image.

## v0.12 GLAM Blackwell compatibility references

- GLAM metarepository Dockerfile: https://raw.githubusercontent.com/nyukat/mammography_metarepository/master/models/nyu_glam/Dockerfile — pins GLAM source commit `17a0019860441e2ea8d7b7c7e0aaeada735e871f` and the historical Python 3.6 / PyTorch 1.1.0 stack.
- GLAM metarepository prediction script: https://raw.githubusercontent.com/nyukat/mammography_metarepository/master/models/nyu_glam/predict/predict.sh — defines preprocessing and image-level classifier invocation used by the prototype.
- GLAM pinned `src/scripts/run_model.py`: https://raw.githubusercontent.com/nyukat/GLAM/17a0019860441e2ea8d7b7c7e0aaeada735e871f/src/scripts/run_model.py — contains the strict PyTorch 1.1.0 assertion, cuDNN capability check, device selection and visualization path addressed by the runtime compatibility layer.
- GLAM pinned `src/modeling/glam.py`: https://raw.githubusercontent.com/nyukat/GLAM/17a0019860441e2ea8d7b7c7e0aaeada735e871f/src/modeling/glam.py — model architecture/device-placement source used to scope compatibility edits without changing layers or weights.
- GLAM pinned `src/modeling/common_functions.py`: https://raw.githubusercontent.com/nyukat/GLAM/17a0019860441e2ea8d7b7c7e0aaeada735e871f/src/modeling/common_functions.py — historical index and sampling semantics explicitly preserved in the Blackwell runtime patch.
