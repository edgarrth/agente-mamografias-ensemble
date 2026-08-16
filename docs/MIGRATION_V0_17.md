# Migración v0.17 — validación GPU integrada

## Alcance

v0.17 no cambia ningún modelo, checkpoint, peso ni dataset. Agrega una orquestación operacional para asegurar y validar uno, varios o todos los runtimes GPU en un único comando.

## Migración

Conservar `.env`, `workspace/` y todas las imágenes de modelos existentes. Actualizar código y reconstruir únicamente servicios de aplicación:

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

Esperado: `0.17.0`.

## Validación integrada

Todos los modelos:

```bash
docker compose exec fastapi \
  python -m model_tools.validate_gpu \
  --models all
```

Subconjunto:

```bash
docker compose exec fastapi \
  python -m model_tools.validate_gpu \
  --models gmic nyu
```

La ejecución no fuerza rebuild por defecto. `ensure_gpu` reconstruye solo si falta la imagen o cambió `build_revision`. Para una reconstrucción explícita:

```bash
docker compose exec fastapi \
  python -m model_tools.validate_gpu \
  --models all \
  --force-rebuild
```

Cada corrida escribe un reporte bajo `workspace/output/model_validation/`. Los smoke tests integrados exigen que los modelos seleccionados estén configurados en GPU.

Wrapper opcional desde el host:

```bash
./scripts/validate-models.sh all
./scripts/validate-models.sh gmic nyu
./scripts/validate-models.sh --force-rebuild all
```

