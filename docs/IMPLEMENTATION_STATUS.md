# Implementation status — v0.11

## Implemented

- modular Docker Compose stack with one lightweight `model-runner` and a host-visible workspace;
- three isolated model images (`mammography-model-gmic`, `mammography-model-nyu`, `mammography-model-glam`) built from the upstream NYU mammography metarepository;
- auditable runtime compatibility generation for legacy CUDA base-image references without changing model source commits or checkpoints;
- automatic Git `safe.directory` handling for the exact host-mounted metarepository path;
- model-runner deliberately without PyTorch, TensorFlow, CUDA Toolkit or cuDNN;
- centralized routing, model image provisioning, temporary container execution, GPU serialization, logging and resource measurement in the runner;
- explicit, idempotent dataset selection; bootstrap never downloads datasets;
- dataset adapter contract and traceable source-manifest preparation path;
- DICOM/PNG preparation to a canonical four-view manifest;
- GMIC, DMV-CNN/NYU and GLAM prediction parsing and deterministic study-level aggregation;
- request for official GMIC/GLAM visualization artifacts (no fake XAI);
- weighted soft voting, threshold and discordance;
- Normal Test Flow, including optional pilot time budget;
- Experimental Test Flow with patient split, 16 × 5 = 80 combinations, selection, freeze and final evaluation;
- Sensitivity, ROC-AUC and confusion-matrix counts;
- deterministic CSV/JSON/Markdown reports;
- FastAPI, Streamlit and PostgreSQL run registry;
- tests for deterministic logic, dataset selection and Docker topology.

## Runtime boundary

`model-runner` controla orquestación, creación de contenedores temporales, lock de GPU y métricas, pero no realiza cómputo neuronal. Frameworks y runtimes CUDA permanecen dentro de cada imagen de modelo.

La elección de dispositivo es independiente por modelo:

```text
DEFAULT_MODEL_DEVICE -> fallback de despliegue
GMIC_DEVICE           -> cpu|gpu
NYU_DEVICE            -> cpu|gpu
GLAM_DEVICE           -> cpu|gpu
```

El perfil GPU no se obtiene de `.env`; se resuelve desde `config/models.yaml`. Si un modelo se solicita en GPU, el Runner solo puede usar su `gpu_compatibility.image` y exige un `gpu_probe` exitoso.

## Verificado en la workstation de investigación

1. build legacy de GMIC, DMV-CNN/NYU y GLAM: **PASS**;
2. smoke inference real CPU de los tres modelos: **PASS**;
3. XAI real de GMIC y GLAM: **PASS**;
4. Docker/NVIDIA Toolkit/CDI con RTX 5060 Ti: **PASS**;
5. GMIC Blackwell `torch 2.7.1+cu128` GPU probe: **PASS**;
6. GMIC Blackwell smoke inference GPU: **PASS**, con CSV y 16 artefactos XAI.
7. DMV-CNN/NYU Blackwell profile definition: **IMPLEMENTED in v0.11; workstation validation pending**.

## Pendiente para la tesis

1. validar en workstation el perfil GPU Blackwell de DMV-CNN/NYU incorporado en v0.11 (`ensure_gpu` → `gpu_probe` → smoke test);
2. construir y validar un perfil GPU específico para GLAM;
3. validar mapping nativo completo de los datasets autorizados hacia `source_manifest.csv`;
4. ejecutar muestras piloto y evaluación experimental/final;
5. obtener mediciones definitivas de tiempo, RAM/VRAM y métricas sobre los datasets seleccionados.

El software falla explícitamente si estos pasos fallan. No contiene fallback de inferencia simulada.
## v0.4 Docker boundary fix

The Model Runner now uses Docker's official CLI image (`docker:29-cli`) rather than the Debian `docker.io` package. This fixes the client/daemon boundary observed with current Docker Desktop/WSL2 and adds `/doctor` diagnostics. This change does not modify the selected models, checkpoints, preprocessing, ensemble method or experimental design.


## v0.5 legacy build compatibility fix

The real workstation build reached the GMIC upstream Dockerfile and failed because `nvidia/cuda:10.1-base-ubuntu18.04` could no longer be resolved. v0.5 replaces only that exact historical `FROM` reference with `nvidia/cudagl:10.1-devel-ubuntu18.04` in a generated compatibility Dockerfile. The same compatibility declaration is configured for GMIC, DMV-CNN/NYU and GLAM because their current metarepository Dockerfiles share the same historical base reference. The change is logged and hashed; model code, checkpoints and experimental methodology are unchanged.


## v0.6 NVIDIA repository key-rotation compatibility

The real GMIC build progressed past the v0.5 base-image replacement and failed during `apt-get update` because the historical CUDA Ubuntu 18.04 repository key was missing (`NO_PUBKEY A4B469963BF863CC`). v0.6 adds an auditable `auto` key-rotation repair: preserve the upstream NYU workaround if present, otherwise inject the narrow NVIDIA key refresh before package index update. Model code, checkpoints, weights and experiment logic remain unchanged.
