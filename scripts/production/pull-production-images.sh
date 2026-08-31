#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE="$ROOT_DIR/deployment/production/docker-compose.prod.yml"
ENV_FILE="${1:-$ROOT_DIR/deployment/production/.env.production}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE" >&2
  exit 2
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

required=(
  APP_IMAGE MODEL_RUNNER_IMAGE EDGE_IMAGE RUNTIME_ASSETS_IMAGE
  GMIC_CPU_REMOTE_IMAGE NYU_CPU_REMOTE_IMAGE GLAM_CPU_REMOTE_IMAGE
  GMIC_BLACKWELL_REMOTE_IMAGE NYU_BLACKWELL_REMOTE_IMAGE GLAM_BLACKWELL_REMOTE_IMAGE
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" || "${!name}" == CHANGE_ME* ]]; then
    echo "$name is not configured" >&2
    exit 3
  fi
done

# Pull the platform and infrastructure images declared by Compose.
docker compose --env-file "$ENV_FILE" -f "$COMPOSE" pull

pull_and_alias() {
  local remote="$1" local_ref="$2"
  echo "Pulling $remote"
  docker pull "$remote"
  echo "Aliasing $remote -> $local_ref"
  docker tag "$remote" "$local_ref"
}

# Preserve the exact image names expected by the validated v0.35.3 model configuration.
pull_and_alias "$GMIC_CPU_REMOTE_IMAGE" "mammography-model-gmic:research"
pull_and_alias "$NYU_CPU_REMOTE_IMAGE" "mammography-model-nyu:research"
pull_and_alias "$GLAM_CPU_REMOTE_IMAGE" "mammography-model-glam:research"
pull_and_alias "$GMIC_BLACKWELL_REMOTE_IMAGE" "mammography-model-gmic:blackwell-cu128"
pull_and_alias "$NYU_BLACKWELL_REMOTE_IMAGE" "mammography-model-nyu:blackwell-cu128"
pull_and_alias "$GLAM_BLACKWELL_REMOTE_IMAGE" "mammography-model-glam:blackwell-cu128"

echo "All production images are present. No model build is required on this host."
