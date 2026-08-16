#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${1:?usage: $0 /workspace/output/normal_tests/<run>}"
docker compose exec fastapi python -m experiments.orientation_counterfactual --run-dir "$RUN_DIR"
