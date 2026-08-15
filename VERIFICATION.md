# Verification — v0.11

Package verification update on 2026-08-15.

## Static/local checks performed for this package

- Python `compileall`: **PASS**.
- Unit tests: **PASS (30/30)**, including NYU Blackwell profile and model-owned compatibility-patch audit checks.
- Docker Compose/configuration YAML parse: **PASS**.
- Host shell scripts `bash -n`: **PASS**.
- NYU Blackwell Dockerfile static validation (pinned commit, PyTorch/TorchVision/CUDA 12.8 profile, compatibility paths): **PASS**.
- Runtime configuration resolution check (`GMIC_DEVICE=gpu`, NYU/GLAM CPU, GMIC profile from YAML): **PASS**.
- Architecture remains one persistent `model-runner` plus isolated model images.
- `GPU_RUNTIME_PROFILE`, `MODEL_DEVICE` and `ALLOW_LEGACY_GPU` are no longer Docker Compose inputs.
- The Model Runner still contains no PyTorch, TensorFlow, CUDA Toolkit or cuDNN.

## Workstation evidence provided by the researcher

- Model Runner / Docker socket integration: **PASS**.
- GMIC legacy image build + real CPU smoke test: **PASS**; CSV + XAI.
- DMV-CNN/NYU legacy image build + real CPU smoke test: **PASS**.
- GLAM legacy image build + real CPU smoke test: **PASS**; CSV + XAI.
- RTX 5060 Ti visible in Fedora Remix WSL and Docker: **PASS**.
- NVIDIA Container Toolkit/CDI: **PASS** (`GPU_HOST_READY`).
- GMIC Blackwell image `mammography-model-gmic:blackwell-cu128`: **BUILD PASS**.
- GMIC Blackwell `gpu_probe`: **GPU_READY**, PyTorch `2.7.1+cu128`, CUDA 12.8, allocation and kernel execution PASS.
- GMIC Blackwell real smoke test: **PASS**; `predictions.csv`, 16 XAI artifacts, elapsed 86.9096 s, sampled average GPU util 8.25%, sampled max GPU memory 2424 MiB.
- Earlier GMIC legacy CPU smoke test elapsed 127.3208 s. These smoke timings are preliminary validation evidence, not a definitive benchmark.

## v0.11 evolution

- GPU compatibility profile is a model characteristic and is resolved only from `config/models.yaml`.
- CPU/GPU choice is a deployment setting resolved per model.
- Current intended mixed deployment: GMIC on GPU; NYU and GLAM on CPU until their own GPU profiles are implemented and validated.
- GPU inference still requires `ALLOW_GPU=true`, a configured `gpu_compatibility.image`, and a successful `gpu_probe`.
- No model code architecture, learned weight, checkpoint, ensemble formula or training behavior is modified by v0.10.

See `docs/WORKSTATION_VALIDATION.md` for the supplied execution evidence.

## v0.11 added validation surface

- DMV-CNN/NYU now has its own `blackwell-cu128` compatibility image definition.
- Static tests verify the pinned NYU commit, CUDA 12.8/PyTorch 2.7.1 runtime and model-owned compatibility patch metadata.
- Real `ensure_gpu`, `gpu_probe` and GPU smoke-test execution for NYU must be completed on the researcher workstation before this runtime is considered experimentally validated.
