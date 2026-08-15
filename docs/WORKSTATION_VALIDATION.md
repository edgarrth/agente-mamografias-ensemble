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

## Estado GPU por modelo en v0.12

- GMIC: `blackwell-cu128` validado con `gpu_probe` y smoke test real.
- DMV-CNN/NYU: `blackwell-cu128` validado con `gpu_probe` y smoke test real.
- GLAM: perfil `blackwell-cu128` incorporado en v0.12; `ensure_gpu` / `gpu_probe` / smoke GPU pendientes de ejecución en la workstation.
