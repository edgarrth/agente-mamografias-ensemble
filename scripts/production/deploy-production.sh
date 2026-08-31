#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE="$ROOT_DIR/deployment/production/docker-compose.prod.yml"
ENV_FILE="${1:-$ROOT_DIR/deployment/production/.env.production}"

"$ROOT_DIR/scripts/production/validate-production-config.sh" "$ENV_FILE"
"$ROOT_DIR/scripts/production/pull-production-images.sh" "$ENV_FILE"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE" up -d --remove-orphans

echo
docker compose --env-file "$ENV_FILE" -f "$COMPOSE" ps
