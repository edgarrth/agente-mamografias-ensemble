#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE="$ROOT_DIR/deployment/production/docker-compose.prod.yml"
ENV_FILE="${1:-$ROOT_DIR/deployment/production/.env.production}"
[[ -f "$ENV_FILE" ]] || { echo "Missing $ENV_FILE" >&2; exit 2; }
if grep -q 'CHANGE_ME' "$ENV_FILE"; then
  echo "Refusing deployment: CHANGE_ME placeholders remain in $ENV_FILE" >&2
  grep -n 'CHANGE_ME' "$ENV_FILE" >&2 || true
  exit 3
fi

docker compose --env-file "$ENV_FILE" -f "$COMPOSE" config >/dev/null

echo "Compose configuration is valid."
echo "Published host ports:"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE" config | awk '/published:/ {print "  " $0}' || true

echo "Expected: only 80/tcp, 443/tcp and 443/udp are public."
