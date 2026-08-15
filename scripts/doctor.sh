#!/usr/bin/env sh
set -eu

echo '=== COMPOSE STATUS ==='
docker compose ps -a || true

echo
echo '=== MODEL RUNNER LOGS ==='
docker compose logs --tail=80 model-runner || true

echo
echo '=== MODEL RUNNER /doctor ==='
if command -v curl >/dev/null 2>&1; then
  curl -sS http://localhost:8010/doctor || true
else
  echo 'curl is not installed on the host; skipping HTTP doctor call.'
fi

echo
echo '=== DOCKER FROM INSIDE RUNNER ==='
docker compose exec -T model-runner sh -lc '
  echo "socket:"; ls -l /var/run/docker.sock || true
  echo "direct ping:"; curl --silent --show-error --unix-socket /var/run/docker.sock http://localhost/_ping || true
  echo; echo "docker version:"; docker version || true
  echo; echo "docker info:"; docker info || true
' || true
