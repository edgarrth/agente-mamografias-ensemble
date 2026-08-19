# Migración v0.10 — Perfil GPU por modelo y selección de dispositivo independiente

## Motivo

En v0.9 `GPU_RUNTIME_PROFILE=blackwell-cu128` era una variable global. Eso mezclaba dos conceptos diferentes:

- **perfil GPU**: propiedad técnica de compatibilidad de un modelo;
- **device CPU/GPU**: decisión del despliegue.

Además, solo GMIC tenía un runtime Blackwell validado, mientras NYU y GLAM seguían validados únicamente en CPU.

## Cambio

El perfil GPU deja de existir en `.env`. Se resuelve exclusivamente desde `config/models.yaml`:

```yaml
models:
  gmic:
    gpu_compatibility:
      profile: blackwell-cu128
      image: mammography-model-gmic:blackwell-cu128
```

La selección de dispositivo pasa a ser por modelo:

```env
DEFAULT_MODEL_DEVICE=cpu
GMIC_DEVICE=gpu
NYU_DEVICE=cpu
GLAM_DEVICE=cpu
ALLOW_GPU=true
GPU_NUMBER=0
```

## Variables obsoletas

Eliminar de `.env` si existen:

```env
MODEL_DEVICE=...
GPU_RUNTIME_PROFILE=...
ALLOW_LEGACY_GPU=...
```

`ALLOW_LEGACY_GPU` ya no es necesario: toda inferencia GPU utiliza únicamente la imagen declarada en `gpu_compatibility.image` y exige un `gpu_probe` exitoso.

## Actualización

Conservar `.env`, `workspace/` y las imágenes Docker ya construidas. Ajustar `.env` manualmente como se indica arriba y luego:

```bash
docker compose down --remove-orphans
docker compose build --no-cache model-runner fastapi
docker compose up -d
```

No es necesario reconstruir `mammography-model-gmic:blackwell-cu128` ni las imágenes `:research`.

## Verificación

```bash
docker compose exec fastapi python -m model_tools.status
curl http://localhost:8010/health
```

Para el estado actual esperado:

```text
gmic.device = gpu
gmic.gpu_profile = blackwell-cu128
nyu.device = cpu
glam.device = cpu
```

Un `smoke_test --models gmic` debe utilizar la imagen `mammography-model-gmic:blackwell-cu128`. NYU y GLAM siguen utilizando sus imágenes `:research` hasta que exista y se valide un perfil GPU específico para cada uno.
