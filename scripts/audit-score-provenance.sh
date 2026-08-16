#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <normal_test_run_dir> [output_dir]" >&2
  exit 2
fi
RUN_DIR="$1"
if [[ $# -eq 2 ]]; then
  exec docker compose exec fastapi python -m experiments.score_provenance --run-dir "$RUN_DIR" --output "$2"
else
  exec docker compose exec fastapi python -m experiments.score_provenance --run-dir "$RUN_DIR"
fi
