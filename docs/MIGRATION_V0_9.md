# Migración v0.9 — Corrección del registro de éxito de `gpu_probe`

## Problema observado

En la RTX 5060 Ti, `model_tools.gpu_probe --models gmic` devolvía HTTP 500 con:

```text
model_runner.api.log() got multiple values for keyword argument 'model'
```

El fallo ocurría **después** de que el probe ejecutara su prueba CUDA y escribiera el resultado,
porque el diccionario `result` ya incluía `model` y la llamada de logging volvía a pasar
`model=model`.

## Corrección

La llamada:

```python
log("GPU_RUNTIME_PROBE_PASSED", model=model, **result)
```

fue reemplazada por:

```python
log("GPU_RUNTIME_PROBE_PASSED", **result)
```

No cambia PyTorch, CUDA, el modelo GMIC, sus pesos, su arquitectura, el checkpoint ni el flujo de inferencia.
Es exclusivamente una corrección de trazabilidad/observabilidad del Model Runner.

## Actualización

Conservar `.env`, `workspace/` y las imágenes `mammography-model-*` ya construidas. No es necesario
reconstruir GMIC. Reconstruir solo los servicios que contienen el código corregido:

```bash
docker compose down --remove-orphans
docker compose build --no-cache model-runner fastapi
docker compose up -d
```

Luego repetir:

```bash
docker compose exec fastapi \
  python -m model_tools.gpu_probe \
  --models gmic
```

El resultado esperado es `status: GPU_READY`.
