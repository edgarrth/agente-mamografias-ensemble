# Migration from v0.5 to v0.6

v0.6 keeps the architecture unchanged: one lightweight `model-runner` plus three isolated model images. It adds one reproducible compatibility repair discovered during the real GMIC build.

## Why v0.6 exists

After v0.5 successfully replaced the no-longer-resolvable historical base image, the GMIC build reached `apt-get update` and failed with:

```text
NO_PUBKEY A4B469963BF863CC
The repository ... ubuntu1804 ... InRelease is not signed.
```

This is a repository-signing-key issue in the historical NVIDIA CUDA/Ubuntu 18.04 environment, not a GMIC inference or checkpoint failure.

## What v0.6 changes

The generated compatibility Dockerfile still changes the historical base image exactly as v0.5 did. In addition, `nvidia_repository_key_rotation_fix: auto` is enabled for all three model definitions.

When the upstream Dockerfile already contains the NVIDIA key refresh (currently DMV-CNN/NYU), v0.6 leaves it untouched and records `upstream_present`. When it is absent (currently GMIC and GLAM), v0.6 inserts the key refresh before `apt-get update` and records `injected`.

The injected commands use the same NVIDIA Ubuntu 18.04 key endpoints already present in the NYU metarepository workaround. No model source, model weights, checkpoints, training, fine-tuning, ensemble rule or prediction logic is modified.

## Audit evidence

The Runner writes:

```text
workspace/logs/model_runner.jsonl
workspace/models/compatibility/<model>.json
```

Relevant event:

```text
NVIDIA_APT_KEY_ROTATION_COMPATIBILITY_APPLIED
```

The compatibility JSON includes whether the fix was `injected` or `upstream_present`, plus SHA-256 hashes of original/generated Dockerfiles.

## Upgrade

Preserve `.env` and `workspace/`. Do not delete Docker volumes.

```bash
docker compose down --remove-orphans
```

Replace the project files with v0.6, then rebuild the application services:

```bash
docker compose build --no-cache model-runner fastapi
docker compose up -d
```

Verify:

```bash
curl http://localhost:8010/health
docker compose ps
```

Retry only GMIC first:

```bash
docker compose exec fastapi \
  python -m model_tools.ensure \
  --models gmic
```

If another historical dependency fails, keep the full error/log. Do not patch the cloned metarepository by hand; create the next versioned compatibility fix instead.
