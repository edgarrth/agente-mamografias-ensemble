#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${1:-$ROOT_DIR/deployment/production/.env.production}"
OUT="${2:-$ROOT_DIR/deployment/production/production-image-lock.txt}"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
images=(
  "$APP_IMAGE" "$MODEL_RUNNER_IMAGE" "$EDGE_IMAGE" "$RUNTIME_ASSETS_IMAGE"
  "$GMIC_CPU_REMOTE_IMAGE" "$NYU_CPU_REMOTE_IMAGE" "$GLAM_CPU_REMOTE_IMAGE"
  "$GMIC_BLACKWELL_REMOTE_IMAGE" "$NYU_BLACKWELL_REMOTE_IMAGE" "$GLAM_BLACKWELL_REMOTE_IMAGE"
  "${POSTGRES_IMAGE:-postgres:16.15-alpine3.24}" "${REDIS_IMAGE:-redis:7-alpine}" "${MINIO_IMAGE:-minio/minio:latest}"
)
: > "$OUT"
for image in "${images[@]}"; do
  docker pull "$image" >/dev/null
  printf '%s\t' "$image" >> "$OUT"
  docker image inspect "$image" --format 'ID={{.Id}} RepoDigests={{json .RepoDigests}}' >> "$OUT"
done
echo "Wrote $OUT"
