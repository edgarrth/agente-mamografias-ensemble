#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${1:-$ROOT_DIR/deployment/production/.env.production}"
[[ -f "$ENV_FILE" ]] || { echo "Missing $ENV_FILE" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

publish_exact() {
  local local_ref="$1" remote_ref="$2"
  docker image inspect "$local_ref" >/dev/null 2>&1 || { echo "Missing local validated image: $local_ref" >&2; return 10; }
  echo "Tagging existing image only (no build): $local_ref -> $remote_ref"
  docker tag "$local_ref" "$remote_ref"
  docker push "$remote_ref"
  docker image inspect "$remote_ref" --format '{{.Id}} {{json .RepoDigests}}'
}

case "${2:-all}" in
  blackwell)
    publish_exact mammography-model-gmic:blackwell-cu128 "$GMIC_BLACKWELL_REMOTE_IMAGE"
    publish_exact mammography-model-nyu:blackwell-cu128 "$NYU_BLACKWELL_REMOTE_IMAGE"
    publish_exact mammography-model-glam:blackwell-cu128 "$GLAM_BLACKWELL_REMOTE_IMAGE"
    ;;
  cpu)
    publish_exact mammography-model-gmic:research "$GMIC_CPU_REMOTE_IMAGE"
    publish_exact mammography-model-nyu:research "$NYU_CPU_REMOTE_IMAGE"
    publish_exact mammography-model-glam:research "$GLAM_CPU_REMOTE_IMAGE"
    ;;
  all)
    "$0" "$ENV_FILE" blackwell
    "$0" "$ENV_FILE" cpu
    ;;
  *)
    echo "Usage: $0 [env-file] [blackwell|cpu|all]" >&2; exit 2;;
esac
