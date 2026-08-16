# Migración v0.16 — fix GMIC con dataset real, health logs y operación trazable

## Motivo

La primera prueba end-to-end de 5 estudios CBIS-DDSM llegó a GMIC y falló en `src/modeling/gmic.py::_convert_crop_position` con `AssertionError: top_k_prop_y <= 0.0`. El error no provino de los warnings de SciPy ni de `torch.load`.

El commit GMIC fijado calcula en `src/utilities/tools.py` el índice 2D con división `/` sobre `max_linear_idx`. En el runtime histórico PyTorch 1.1 esa operación mantenía semántica de índice entero; en el runtime moderno la división verdadera puede producir un cociente float y un residuo `max_idx_y` ligeramente negativo. v0.16 preserva explícitamente la semántica histórica con:

```python
max_idx_x = torch.div(max_linear_idx, W_map, rounding_mode="floor")
max_idx_y = max_linear_idx - max_idx_x * W_map
```

No se elimina el sanity check, no se clampa una coordenada inválida y no se cambia ninguna capa, peso, checkpoint, score ni entrenamiento.

## Rebuild requerido

GMIC pasa a `gpu_compatibility.build_revision=2`. La primera ejecución de `ensure_gpu --models gmic` en v0.16 reconstruye `mammography-model-gmic:blackwell-cu128` aunque el tag ya exista, y elimina el probe GPU anterior porque corresponde a bytes de imagen previos.

NYU y GLAM mantienen revisión 1 y se reutilizan.

## Migración

Conservar `.env`, `workspace/`, DICOM/PNG preparados y las imágenes NYU/GLAM. Luego:

```bash
docker compose down --remove-orphans
docker compose build --no-cache model-runner fastapi bootstrap streamlit
docker compose up -d
```

Verificar versión:

```bash
docker compose exec fastapi cat /app/VERSION
docker compose exec model-runner cat /runner/VERSION
```

Esperado: `0.16.0` en ambos.

Reconstruir solo GMIC Blackwell y renovar evidencia GPU:

```bash
docker compose exec fastapi python -m model_tools.ensure_gpu --models gmic
docker compose exec fastapi python -m model_tools.gpu_probe --models gmic
```

Después repetir la prueba de integración de 5 estudios. `prepare` no necesita repetirse: v0.16 no cambia los 105 estudios/420 PNG ya preparados.
