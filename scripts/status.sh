#!/usr/bin/env bash
set -euo pipefail

echo "Workspace: $(cd "${HOST_WORKSPACE:-./workspace}" 2>/dev/null && pwd || echo "${HOST_WORKSPACE:-./workspace}")"
echo
echo "Docker services (expected model controller: mammography-model-runner):"
docker compose ps -a

echo
echo "Model Runner doctor:"
if command -v curl >/dev/null 2>&1; then
  curl -sS http://localhost:8010/doctor || true
  echo
else
  echo "curl unavailable on host; run: docker compose exec -T model-runner docker version"
fi

echo
echo "Dataset status (available only after FastAPI starts):"
docker compose exec -T fastapi python -m dataset_pipeline.status || true

echo
echo "Model status through single runner (available only after FastAPI starts):"
docker compose exec -T fastapi python -m model_tools.status || true
