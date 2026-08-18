# Migration v0.30.1 → v0.30.2

## Scope

v0.30.2 is an operational-resilience release for the large RSNA formal experiment. It does **not** change the prepared datasets, diagnostic exclusions, 30/70 patient-level stratified split, model checkpoints, preprocessing semantics, orientation decision rule, canonical score semantics, 16 weight combinations, five adaptive thresholds, selection policy, freeze boundary, or Final Test isolation.

## New execution behavior

- Configuration orientation preflight is executed in deterministic chunks and checkpointed.
- GMIC → NYU → GLAM inference is executed sequentially inside deterministic chunks.
- Default formal chunk size: 25 studies.
- Each successful chunk stores input-manifest and prediction SHA-256 hashes plus a SUCCESS marker.
- Interrupted/failed chunks restart from their beginning; prior successful chunks are reused after integrity validation.
- Configuration can be resumed with `--resume-experiment <experiment-id>` without recreating the split.
- Final Test uses the same chunk/resume mechanism, but remains impossible before `experiments.freeze`.
- `chunk_progress.json` and `orientation_chunk_progress.json` expose progress.
- Successful cache identity/hash mismatches cause a hard failure instead of silent reuse or re-inference.

## Test execution

The FastAPI research image now includes the repository tests and pytest test extra so the validation command is directly available after rebuild:

```bash
docker compose exec fastapi python -m pytest -q
```

## Initial Configuration command

```bash
docker compose exec fastapi python -m experiments.run \
  --datasets rsna \
  --configuration-ratio 0.30 \
  --seed 42
```

If interrupted, reuse the same experiment ID:

```bash
docker compose exec fastapi python -m experiments.run \
  --datasets rsna \
  --configuration-ratio 0.30 \
  --seed 42 \
  --resume-experiment experiment-YYYYMMDDTHHMMSSZ
```
