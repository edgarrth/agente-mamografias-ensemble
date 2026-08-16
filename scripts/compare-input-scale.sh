#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /workspace/output/normal_tests/<run-dir> [--skip-nyu-crop]" >&2
  exit 2
fi
RUN_DIR="$1"
EXTRA=()
if [[ ${2:-} == "--skip-nyu-crop" ]]; then EXTRA+=("--skip-nyu-crop"); fi
exec docker compose exec fastapi python -m experiments.input_scale_comparison --run-dir "$RUN_DIR" "${EXTRA[@]}"
