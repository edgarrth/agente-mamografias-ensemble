# Migración v0.12 — Runtime Blackwell de GLAM

## Motivo

v0.11 dejó GMIC y DMV-CNN/NYU con perfiles GPU independientes y validados progresivamente, pero GLAM todavía no tenía `gpu_compatibility`. v0.12 completa la misma frontera modular para GLAM sin convertir el perfil en una variable global de despliegue.

El Dockerfile upstream de `nyu_glam` fija el commit `17a0019860441e2ea8d7b7c7e0aaeada735e871f` y usa Python 3.6, PyTorch 1.1.0 y TorchVision 0.2.2. El código GLAM también incluye un `assert` estricto de PyTorch 1.1.0 y APIs/dispositivos propios de ese runtime histórico. v0.12 conserva el commit, checkpoints y arquitectura, y crea una imagen separada para Blackwell.

## Nueva imagen

```text
mammography-model-glam:blackwell-cu128
```

Runtime:

```text
Python 3.10
PyTorch 2.7.1
TorchVision 0.22.1
CUDA wheel 12.8
```

## Parches de compatibilidad declarados

Solo se aplican al runtime Blackwell de GLAM:

- se elimina el `assert` que obliga literalmente a `torch==1.1.0`; la versión queda fijada por el Dockerfile/perfil;
- `torch.has_cudnn` se sustituye por `torch.backends.cudnn.is_available()`;
- `TkAgg` se sustituye por `Agg` para producir visualizaciones en un contenedor headless;
- tensores intermedios de crops/saliency se crean en el mismo dispositivo que los tensores de entrada, evitando cruces CPU/GPU introducidos por el runtime moderno;
- se preserva la semántica histórica de división entera de índices mediante `torch.div(..., rounding_mode="floor")`;
- se fija `align_corners=True` en `grid_sample` para conservar la semántica por defecto de PyTorch 1.1 cuando esa utilidad sea utilizada.

Estas modificaciones no cambian capas, tamaños, checkpoint, pesos aprendidos, hiperparámetros de inferencia ni agregación del score. No se realiza entrenamiento ni fine-tuning.

## Pasos

Conserve `.env`, `workspace/` y las imágenes Docker existentes. Reemplace el código por v0.12:

```bash
docker compose down --remove-orphans
docker compose build --no-cache model-runner fastapi
docker compose up -d
```

Construya el runtime GPU de GLAM:

```bash
docker compose exec fastapi \
  python -m model_tools.ensure_gpu \
  --models glam
```

Valide CUDA:

```bash
docker compose exec fastapi \
  python -m model_tools.gpu_probe \
  --models glam
```

Solo si devuelve `GPU_READY`, cambie en `.env`:

```env
GLAM_DEVICE=gpu
```

El estado objetivo es:

```env
GMIC_DEVICE=gpu
NYU_DEVICE=gpu
GLAM_DEVICE=gpu
ALLOW_GPU=true
GPU_NUMBER=0
```

Recree servicios:

```bash
docker compose up -d --force-recreate model-runner fastapi
```

Verifique routing:

```bash
docker compose exec fastapi \
  python -m model_tools.status
```

Finalmente ejecute la inferencia real:

```bash
docker compose exec fastapi \
  python -m model_tools.smoke_test \
  --models glam
```

No habilite `GLAM_DEVICE=gpu` antes de `GPU_READY`. Si la inferencia completa descubre una incompatibilidad adicional, debe resolverse en una nueva versión trazable del ZIP.
