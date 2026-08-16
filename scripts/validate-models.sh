#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/validate-models.sh all
  ./scripts/validate-models.sh gmic nyu glam
  ./scripts/validate-models.sh --force-rebuild gmic nyu

Runs the current integrated GPU validation inside the FastAPI container:
  ensure current GPU image revision -> CUDA probe -> upstream smoke test.
EOF
}

force_rebuild=false
models=()

while (($#)); do
  case "$1" in
    --force-rebuild)
      force_rebuild=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      models+=("$1")
      ;;
  esac
  shift
done

if ((${#models[@]} == 0)); then
  usage >&2
  exit 2
fi

cmd=(docker compose exec fastapi python -m model_tools.validate_gpu --models "${models[@]}")
if [[ "$force_rebuild" == "true" ]]; then
  cmd+=(--force-rebuild)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
