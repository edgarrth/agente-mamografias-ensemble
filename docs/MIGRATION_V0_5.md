# Migration from v0.4 to v0.5

v0.5 keeps the same architecture: one lightweight `model-runner` and three isolated model images. The change is a reproducible compatibility fix for the historical NYU model Dockerfiles.

## Why v0.5 exists

A real GMIC `ensure` reached the upstream Docker build but failed before any model package was installed because the upstream Dockerfile starts with:

```dockerfile
FROM nvidia/cuda:10.1-base-ubuntu18.04
```

That historical tag was not resolvable in the target Docker environment. A manual pull confirmed that the NVIDIA CUDA 10.1 / Ubuntu 18.04 image below is available:

```text
nvidia/cudagl:10.1-devel-ubuntu18.04
```

The official GMIC, DMV-CNN/NYU and GLAM metarepository Dockerfiles all use the same historical base tag, so the compatibility mechanism is configured consistently for all three.

## What changes

The runner now generates a Dockerfile under:

```text
/workspace/runtime/mammography_metarepository/.thesis_compat/<model>.Dockerfile
```

It copies the upstream Dockerfile verbatim and changes **only** its first `FROM` line:

```text
nvidia/cuda:10.1-base-ubuntu18.04
        ->
nvidia/cudagl:10.1-devel-ubuntu18.04
```

The runner records source and generated SHA-256 hashes plus the reason in:

```text
workspace/logs/model_runner.jsonl
workspace/models/compatibility/<model>.json
workspace/models/registry.json
```

No model source commit, checkpoint, architecture, training, fine-tuning, voting rule or prediction logic is changed by this compatibility patch.

## Git bind-mount fix

v0.5 also makes the previous manual Git workaround permanent. Before calling `git rev-parse`, the runner adds exactly:

```text
/workspace/runtime/mammography_metarepository
```

as a Git `safe.directory` when necessary and logs `GIT_SAFE_DIRECTORY_ADDED`.

## Upgrade

Do not delete volumes or the workspace:

```bash
docker compose down --remove-orphans
```

Replace project files with v0.5 while preserving `.env` and `workspace/`, then rebuild the application images:

```bash
docker compose build --no-cache model-runner fastapi
docker compose up -d
```

Check:

```bash
curl http://localhost:8010/health
docker compose exec fastapi python -m model_tools.status
```

Then retry only GMIC:

```bash
docker compose exec fastapi \
  python -m model_tools.ensure \
  --models gmic
```

If another legacy dependency fails later, do not modify the cloned metarepository manually. Capture `workspace/logs/model_runner.jsonl`; each compatibility correction should be made in a new project version so it remains reproducible.
