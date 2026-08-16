#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /workspace/output/normal_tests/<run-id> [--write-images]" >&2
  exit 2
fi

RUN_DIR="$1"
shift

docker compose exec fastapi \
  python -m experiments.dicom_presentation_counterfactual \
  --run-dir "$RUN_DIR" "$@"
