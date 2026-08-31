#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${1:-$ROOT_DIR/deployment/production/.env.production}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy .env.production.example and configure it first." >&2
  exit 2
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${APP_IMAGE:?APP_IMAGE is required}"
: "${MODEL_RUNNER_IMAGE:?MODEL_RUNNER_IMAGE is required}"
: "${EDGE_IMAGE:?EDGE_IMAGE is required}"
: "${RUNTIME_ASSETS_IMAGE:?RUNTIME_ASSETS_IMAGE is required}"

HOST_WORKSPACE="${HOST_WORKSPACE:-$ROOT_DIR/workspace}"
META_DIR="$HOST_WORKSPACE/runtime/mammography_metarepository"
MODELS_DIR="$HOST_WORKSPACE/models"
if [[ ! -d "$META_DIR/.git" ]]; then
  echo "Validated metarepository not found at $META_DIR" >&2
  echo "Run the normal validated environment first so /workspace/runtime/mammography_metarepository exists." >&2
  exit 3
fi
for required in "$MODELS_DIR/compatibility/gmic-gpu.json" "$MODELS_DIR/compatibility/glam-gpu.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing validated GPU compatibility metadata: $required" >&2
    echo "This metadata is required to prevent an unnecessary Blackwell image rebuild during CPU orientation preflight." >&2
    exit 4
  fi
done

EXPECTED_VERSION="$(cat "$ROOT_DIR/VERSION")"
echo "Publishing production platform images for agent $EXPECTED_VERSION"

echo "[1/4] Building application image: $APP_IMAGE"
docker build -t "$APP_IMAGE" -f "$ROOT_DIR/docker/app.Dockerfile" "$ROOT_DIR"

echo "[2/4] Building model-runner image: $MODEL_RUNNER_IMAGE"
docker build -t "$MODEL_RUNNER_IMAGE" -f "$ROOT_DIR/docker/model-runner.Dockerfile" "$ROOT_DIR"

echo "[3/4] Building edge image: $EDGE_IMAGE"
docker build -t "$EDGE_IMAGE" -f "$ROOT_DIR/deployment/production/edge.Dockerfile" "$ROOT_DIR"

echo "[4/4] Packaging the already-resolved runtime workspace into: $RUNTIME_ASSETS_IMAGE"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
mkdir -p "$TMP_DIR/seed/runtime" "$TMP_DIR/seed/models"
# Copy only runtime/model metadata needed by Web inference; datasets, Batch outputs and study data are excluded.
cp -a "$META_DIR" "$TMP_DIR/seed/runtime/mammography_metarepository"
cp -a "$MODELS_DIR/." "$TMP_DIR/seed/models/"
tar -C "$TMP_DIR/seed" -cf "$TMP_DIR/workspace_seed.tar" runtime models
rm -rf "$TMP_DIR/seed"
cp "$ROOT_DIR/deployment/production/runtime-assets.Dockerfile" "$TMP_DIR/Dockerfile"
docker build -t "$RUNTIME_ASSETS_IMAGE" "$TMP_DIR"

for image in "$APP_IMAGE" "$MODEL_RUNNER_IMAGE" "$EDGE_IMAGE" "$RUNTIME_ASSETS_IMAGE"; do
  echo "Pushing $image"
  docker push "$image"
done

echo
echo "Published platform images. Record their registry digests before deployment:"
for image in "$APP_IMAGE" "$MODEL_RUNNER_IMAGE" "$EDGE_IMAGE" "$RUNTIME_ASSETS_IMAGE"; do
  docker image inspect "$image" --format '{{.Id}} {{json .RepoDigests}}' || true
done
