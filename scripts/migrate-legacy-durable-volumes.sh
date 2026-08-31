#!/usr/bin/env bash
set -euo pipefail

POSTGRES_TARGET="${POSTGRES_VOLUME_NAME:-mammography-postgres-data}"
MINIO_TARGET="${MINIO_VOLUME_NAME:-mammography-minio-data}"
ASSUME_YES=false
[[ "${1:-}" == "--yes" ]] && ASSUME_YES=true

require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: falta el comando '$1'." >&2; exit 1; }; }
require_cmd docker

latest_legacy_volume() {
  local role="$1" target="$2"
  local candidates=()
  while IFS= read -r vol; do
    [[ -z "$vol" || "$vol" == "$target" ]] && continue
    candidates+=("$vol")
  done < <(docker volume ls --filter "label=com.docker.compose.volume=${role}" --format '{{.Name}}')
  if ((${#candidates[@]} == 0)); then
    return 1
  fi
  for vol in "${candidates[@]}"; do
    printf '%s\t%s\n' "$(docker volume inspect -f '{{.CreatedAt}}' "$vol")" "$vol"
  done | sort | tail -n 1 | cut -f2-
}

volume_nonempty() {
  local vol="$1"
  docker run --rm -v "${vol}:/data:ro" alpine:3.20 sh -c 'test -n "$(ls -A /data 2>/dev/null)"'
}

volume_in_use() {
  local vol="$1"
  [[ -n "$(docker ps -q --filter "volume=${vol}")" ]]
}

copy_volume() {
  local source="$1" target="$2" label="$3"
  if volume_in_use "$source"; then
    echo "ERROR: el volumen origen '$source' está siendo usado por un contenedor activo." >&2
    echo "Detenga primero PostgreSQL/MinIO de la versión anterior y vuelva a ejecutar este script." >&2
    exit 2
  fi
  docker volume create "$target" >/dev/null
  if volume_nonempty "$target"; then
    echo "ERROR: el volumen destino '$target' ya contiene datos. No se sobrescribirá." >&2
    exit 3
  fi
  echo "Migrando ${label}: ${source} -> ${target}"
  docker run --rm -v "${source}:/from:ro" -v "${target}:/to" alpine:3.20 \
    sh -c 'cp -a /from/. /to/'
}

PG_SOURCE="${LEGACY_POSTGRES_VOLUME_NAME:-$(latest_legacy_volume postgres_data "$POSTGRES_TARGET" || true)}"
MINIO_SOURCE="${LEGACY_MINIO_VOLUME_NAME:-$(latest_legacy_volume minio_data "$MINIO_TARGET" || true)}"

if [[ -z "$PG_SOURCE" && -z "$MINIO_SOURCE" ]]; then
  echo "No se encontraron volúmenes legacy de PostgreSQL o MinIO para migrar."
  exit 0
fi

echo "Se detectaron los siguientes volúmenes legacy más recientes:"
[[ -n "$PG_SOURCE" ]] && echo "  PostgreSQL: $PG_SOURCE -> $POSTGRES_TARGET"
[[ -n "$MINIO_SOURCE" ]] && echo "  MinIO:      $MINIO_SOURCE -> $MINIO_TARGET"

if [[ "$ASSUME_YES" != true ]]; then
  read -r -p "¿Copiar estos datos a los volúmenes estables? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || { echo "Operación cancelada."; exit 0; }
fi

[[ -n "$PG_SOURCE" ]] && copy_volume "$PG_SOURCE" "$POSTGRES_TARGET" "PostgreSQL"
[[ -n "$MINIO_SOURCE" ]] && copy_volume "$MINIO_SOURCE" "$MINIO_TARGET" "MinIO"

echo "Migración completada. Ya puede levantar v0.35.3 con docker compose up -d --build."
