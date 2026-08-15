# Migration from v0.2 to v0.3

v0.3 replaces the three persistent controller containers with a single `model-runner`.

## Before replacing files

```bash
docker compose down --remove-orphans
```

Do **not** add `-v` if you want to preserve PostgreSQL/MinIO volumes. Preserve the host `workspace/` directory as well.

## Environment variables

Remove these v0.2 variables if present:

```text
GMIC_RUNTIME_URL
NYU_RUNTIME_URL
GLAM_RUNTIME_URL
```

Use:

```env
MODEL_RUNNER_URL=http://model-runner:8010
```

The remaining variables such as `MODEL_DEVICE`, `GPU_NUMBER`, `ALLOW_LEGACY_GPU`, `MODEL_BOOTSTRAP_MODE` and `RESOURCE_SAMPLE_SECONDS` still apply, now to the single runner.

## Rebuild

```bash
docker compose up -d --build
docker compose ps
```

Expected persistent model controller:

```text
mammography-model-runner
```

The model images are not persistent Compose services. With lazy bootstrap they are built/reused on demand:

```bash
docker compose exec fastapi python -m model_tools.ensure --models gmic nyu glam
```

Expected local images after `ensure`:

```text
mammography-model-gmic:research
mammography-model-nyu:research
mammography-model-glam:research
```

Then run the real smoke tests:

```bash
docker compose exec fastapi python -m model_tools.smoke_test --models gmic nyu glam
```
