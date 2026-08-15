# Migración v0.11 — Runtime Blackwell de DMV-CNN / NYU

## Motivo

v0.10 valida GMIC en GPU, pero DMV-CNN/NYU todavía no tenía un perfil GPU propio. Por diseño, `ensure_gpu --models nyu` rechazaba la solicitud con `GPU_PROFILE_NOT_CONFIGURED` en vez de reutilizar el runtime de GMIC.

v0.11 agrega un runtime Blackwell específico para DMV-CNN/NYU. El perfil técnico permanece en `config/models.yaml`; `.env` conserva únicamente la selección de dispositivo por modelo.

## Qué cambia

Nueva imagen:

```text
mammography-model-nyu:blackwell-cu128
```

Runtime:

```text
Python 3.10
PyTorch 2.7.1
TorchVision 0.22.1
CUDA wheel 12.8
```

Se conserva el commit upstream:

```text
de2b0855d02984df0f516008bb4513ff71460e21
```

Los únicos parches de código declarados son de compatibilidad de runtime: reemplazo de `torch.has_cudnn` por `torch.backends.cudnn.is_available()` en los puntos de selección de dispositivo, más la misma creación de directorios de heatmaps ya aplicada por el Dockerfile del metarepositorio. No se cambia la arquitectura, los checkpoints ni se realiza entrenamiento.

## Pasos

Conserve `.env`, `workspace/` y las imágenes ya construidas. Después de reemplazar el código:

```bash
docker compose down --remove-orphans
docker compose build --no-cache model-runner fastapi
docker compose up -d
```

Construya el runtime GPU de NYU:

```bash
docker compose exec fastapi \
  python -m model_tools.ensure_gpu \
  --models nyu
```

Valide asignación y kernel CUDA:

```bash
docker compose exec fastapi \
  python -m model_tools.gpu_probe \
  --models nyu
```

Solo si devuelve `GPU_READY`, cambie:

```env
NYU_DEVICE=gpu
```

Mantenga:

```env
GMIC_DEVICE=gpu
GLAM_DEVICE=cpu
ALLOW_GPU=true
GPU_NUMBER=0
```

Recree el runner/aplicación:

```bash
docker compose up -d --force-recreate model-runner fastapi
```

Y ejecute la inferencia real de prueba:

```bash
docker compose exec fastapi \
  python -m model_tools.smoke_test \
  --models nyu
```

No habilite `NYU_DEVICE=gpu` si `gpu_probe` falla. Cualquier incompatibilidad adicional debe corregirse en una nueva versión, no mediante cambios manuales no trazables.
