#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <raw_model_predictions.csv> [output_dir]" >&2
  exit 2
fi
INPUT="$1"
if [[ $# -eq 2 ]]; then
  exec docker compose exec fastapi python -m experiments.score_analysis --input "$INPUT" --output "$2"
else
  exec docker compose exec fastapi python -m experiments.score_analysis --input "$INPUT"
fi
