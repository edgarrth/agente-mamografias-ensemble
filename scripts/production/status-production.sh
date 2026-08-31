#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE="$ROOT_DIR/deployment/production/docker-compose.prod.yml"
ENV_FILE="${1:-$ROOT_DIR/deployment/production/.env.production}"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE" ps
