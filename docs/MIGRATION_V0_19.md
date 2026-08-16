# Migración v0.19 — contrato de estado de inferencia Model Runner

## Contexto

La ejecución real de 5 estudios CBIS-DDSM en v0.18 demostró que GMIC completó inferencia, escribió `gmic.csv` y produjo 20 artefactos XAI. Sin embargo, el pipeline abortó antes de NYU porque el Model Runner devolvió `status=READY` en la respuesta de `/models/{model}/run`.

La causa era una colisión de claves al construir el payload: `ensure_gpu_image()` aporta metadata con `status=READY` (estado de disponibilidad de la imagen), y esa metadata se fusionaba **después** de `status=SUCCESS`, sobrescribiendo el estado de la operación de inferencia.

## Cambio v0.19

`/models/{model}/run` conserva toda la metadata de imagen/runtime, pero escribe el estado de operación al final del merge. Por contrato:

- `READY` describe que una imagen/runtime está disponible para ejecutar.
- `GPU_READY` describe que el probe CUDA pasó.
- `SUCCESS` describe que una inferencia real terminó y produjo su CSV.

El pipeline sigue exigiendo `SUCCESS`; no se relaja la guarda a `READY`.

## Impacto científico

No se modifica ningún modelo, checkpoint, peso, arquitectura, preprocessing, dataset, ground truth, score, XAI ni fórmula de ensemble. GMIC permanece en `build_revision=3`; NYU y GLAM permanecen en revisión 1. No es necesario reconstruir imágenes de modelos ni repetir `prepare`.

## Migración

Conservar `.env`, `workspace/` y las imágenes Docker de modelos. Reemplazar el código y ejecutar:

```bash
docker compose down --remove-orphans
docker compose build --no-cache model-runner fastapi bootstrap streamlit
docker compose up -d
```

Verificar:

```bash
docker compose exec fastapi cat /app/VERSION
docker compose exec model-runner cat /runner/VERSION
```

Ambos deben devolver `0.19.0`.

Después repetir la prueba de integración de 5 estudios. No requiere `ensure_gpu`, `gpu_probe`, `smoke_test`, `inspect` ni `prepare`, porque v0.19 solo corrige el contrato de respuesta del Model Runner.
