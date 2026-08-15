#!/usr/bin/env bash
set -euo pipefail
TAIL_LINES="${TAIL_LINES:-100}"
echo "Following FastAPI + Model Runner container logs (last ${TAIL_LINES} lines)."
echo "CLI/pipeline audit is also persisted under workspace/logs/audit.jsonl and model_runner.jsonl."
exec docker compose logs -f --tail="${TAIL_LINES}" fastapi model-runner
