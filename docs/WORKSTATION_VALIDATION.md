# Workstation validation evidence

Evidencia proporcionada por el investigador durante la ejecución real del prototipo en la workstation de tesis. No sustituye la evaluación experimental final.

## Plataforma

- NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM: visible en WSL y Docker.
- NVIDIA Container Toolkit/CDI: `GPU_HOST_READY`.
- Model Runner / Docker Engine: operativo.

## Modelos legacy en CPU

- GMIC `mammography-model-gmic:research`: build PASS; smoke test real PASS; CSV y XAI generados.
- DMV-CNN/NYU `mammography-model-nyu:research`: build PASS; smoke test real PASS.
- GLAM `mammography-model-glam:research`: build PASS; smoke test real PASS; CSV y XAI generados.

## GMIC Blackwell GPU

Imagen: `mammography-model-gmic:blackwell-cu128`.

`gpu_probe` real:

```text
status = GPU_READY
torch = 2.7.1+cu128
torch_cuda = 12.8
cuda_available = true
device_count = 1
gpu_name = NVIDIA GeForce RTX 5060 Ti
allocation_ok = true
kernel_ok = true
```

Smoke test GPU real:

```text
status = READY
elapsed_seconds = 86.90960656599964
avg_gpu_util_percent = 8.25
max_gpu_memory_mib = 2424.0
XAI artifacts = 16
output = /workspace/output/smoke_tests/gmic/predictions.csv
```

El mismo smoke test había sido ejecutado previamente con el runtime legacy en CPU y reportó `elapsed_seconds = 127.32081498400021`. Estas mediciones sirven para validar funcionamiento y obtener una referencia preliminar; no deben presentarse como benchmark definitivo porque incluyen preprocesamiento y el muestreo de recursos es de baja frecuencia.

## Estado GPU por modelo en v0.11

- GMIC: perfil `blackwell-cu128` validado para GPU.
- DMV-CNN/NYU: perfil `blackwell-cu128` incorporado en v0.11; build/probe/smoke GPU aún pendientes de validación en la workstation.
- GLAM: sin perfil GPU validado todavía; CPU.

## DMV-CNN / NYU Blackwell GPU — validación posterior a v0.11

Imagen: `mammography-model-nyu:blackwell-cu128`.

`gpu_probe` real:

```text
status = GPU_READY
torch = 2.7.1+cu128
torch_cuda = 12.8
cuda_available = true
device_count = 1
gpu_name = NVIDIA GeForce RTX 5060 Ti
allocation_ok = true
kernel_ok = true
```

Smoke test GPU real:

```text
status = READY
elapsed_seconds = 14.72706528299932
avg_gpu_util_percent = 7.4
max_gpu_memory_mib = 2226.0
output = /workspace/output/smoke_tests/nyu/predictions.csv
```

El smoke test legacy CPU previo reportó `elapsed_seconds = 35.354556032999426`. Estas cifras son evidencia de funcionamiento/preliminar y no un benchmark final del experimento.

## Estado GPU por modelo en v0.13

- GMIC: `blackwell-cu128` validado con `gpu_probe` y smoke test real.
- DMV-CNN/NYU: `blackwell-cu128` validado con `gpu_probe` y smoke test real.
- GLAM: `blackwell-cu128` validado; `gpu_probe=GPU_READY`; smoke GPU `READY`, elapsed 72.9107 s, sampled max VRAM 2448 MiB, 16 artefactos XAI.


## Evidencia adicional recibida antes de v0.13

- DMV-CNN/NYU smoke GPU: `READY`, imagen `mammography-model-nyu:blackwell-cu128`, elapsed 14.7271 s, sampled GPU util 7.4%, sampled max VRAM 2226 MiB.
- GLAM smoke GPU: `READY`, imagen `mammography-model-glam:blackwell-cu128`, elapsed 72.9107 s, sampled GPU util 5.625%, sampled max VRAM 2448 MiB, 16 visualizaciones XAI.

Estos tiempos son evidencia operacional de smoke tests y no se presentan como benchmark experimental definitivo.

## Estado distribuido en v0.14

`.env.example` queda alineado con el estado realmente probado el 2026-08-15:

```env
DEFAULT_MODEL_DEVICE=cpu
GMIC_DEVICE=gpu
NYU_DEVICE=gpu
GLAM_DEVICE=gpu
ALLOW_GPU=true
GPU_NUMBER=0
```

Los tres modelos conservan perfiles Blackwell independientes en `config/models.yaml`; esta configuración de despliegue no cambia arquitectura, pesos ni checkpoints.

## CBIS-DDSM — primera inspección real completa antes de v0.15

Ejecución real del `dataset_pipeline.inspect` sobre la copia local completa:

```text
inicio = 2026-08-15 15:11:31.616
fin    = 2026-08-15 15:23:04.535
elapsed ~= 11 min 32.9 s
```

Resultado observado:

```text
metadata_rows = 3568
resolved_metadata_rows = 3568
unresolved_metadata_rows = 0
dicom_files_indexed = 10239
dicom_headers_valid = 10239
patients = 1566
complete_four_view_studies = 105
incomplete_studies = 1461
supplemental_standard_views = 0
ensemble_compatible = true
```

Este tiempo queda como baseline medido para una reconstrucción/inspección que recorrió la colección completa bajo `/mnt/d` en WSL/NTFS. v0.15 introduce reutilización explícita del índice DICOM y enriquecimiento mediante `metadata.csv` sin reabrir DICOM cuando el cache ya existe. El tiempo real de esa ruta optimizada todavía debe medirse en la workstation.

## CBIS-DDSM — validación real v0.15 y primer normal test

Evidencia de la workstation objetivo del 2026-08-15:

- `dataset_pipeline.download`: DICOM reutilizados, cuatro CSV oficiales reutilizados y SHA-256 válidos; ~0.9 s.
- `dataset_pipeline.inspect` con cache: 10,239 DICOM, 1,566 pacientes, 105 estudios de cuatro vistas, 420 imágenes seleccionadas, 72 benignos/33 malignos; ~19.5 s.
- `dataset_pipeline.prepare`: 105/105 estudios convertidos; ~8 min 20 s.
- `tests_flow.normal --samples 5`: inició GMIC y falló ~27 s después del inicio total con `AssertionError: top_k_prop_y <= 0.0` durante `GMIC._convert_crop_position`.

v0.16 identifica como causa de compatibilidad el cálculo de índice 2D en el commit GMIC fijado, donde `max_idx_x = max_linear_idx / W_map` fue escrito para semántica PyTorch 1.1. El parche sustituye únicamente esa división por `torch.div(..., rounding_mode="floor")`; la validación end-to-end de este cambio queda pendiente de la siguiente ejecución en workstation.

## 2026-08-15 — GMIC Blackwell build_revision=2 (v0.16)

Validación real reportada por la workstation objetivo después del fix de semántica de índices:

- `/app/VERSION`: `0.16.0`.
- `ensure_gpu gmic`: `READY`, `build_revision=2`, mismo commit GMIC `3bf4ce81dfa40553f108c8bfaf03bf006e082761`, arquitectura/pesos/entrenamiento sin cambios.
- `gpu_probe gmic`: `GPU_READY`, PyTorch `2.7.1+cu128`, CUDA `12.8`, RTX 5060 Ti, `allocation_ok=true`, `kernel_ok=true`.
- smoke test GMIC: `READY`, ~89.999 s, 29 muestras de métrica, 16 artefactos XAI, ~2740 MiB VRAM máxima observada.

Esta validación demuestra que el fix de cociente/remainder no rompió el sample workflow upstream. La prueba CBIS-DDSM end-to-end debe repetirse para validar específicamente el caso que originó el error.

