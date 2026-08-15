> **Nota v0.10:** este documento conserva el procedimiento histórico de v0.8. Desde v0.10 `GPU_RUNTIME_PROFILE` ya no se configura en `.env`; el perfil vive en `config/models.yaml` y el device se selecciona por modelo (`GMIC_DEVICE`, `NYU_DEVICE`, `GLAM_DEVICE`).

# Migración v0.8 — Runtime GPU Blackwell para GMIC

El runtime legacy GMIC (PyTorch 1.1.0/CUDA 9.0) detecta la RTX 5060 Ti pero se bloquea en la primera asignación `tensor.cuda()`. Se conserva `mammography-model-gmic:research` para CPU y se agrega `mammography-model-gmic:blackwell-cu128` para la prueba GPU.

```bash
docker compose build --no-cache model-runner fastapi
docker compose up -d
docker compose exec fastapi python -m model_tools.ensure_gpu --models gmic
docker compose exec fastapi python -m model_tools.gpu_probe --models gmic
```

Solo si el probe devuelve `GPU_READY`, habilitar `MODEL_DEVICE=gpu` y `ALLOW_GPU=true`.
