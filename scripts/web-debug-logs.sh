#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Uso: $0 <run_id> [archivo_salida]" >&2
  echo "Ejemplo: $0 web-20260819T031746Z-a85aa6cd run-debug.log" >&2
  exit 2
fi

run_id="$1"
out="${2:-}"

collect() {
  docker compose logs --no-color fastapi model-runner 2>&1 | grep -F -- "$run_id" || true
}

if [[ -n "$out" ]]; then
  collect > "$out"
  echo "Log Web exportado: $out"
else
  collect
fi
