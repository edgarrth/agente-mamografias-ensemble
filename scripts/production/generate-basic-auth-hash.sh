#!/usr/bin/env bash
set -euo pipefail
read -r -s -p "Password for the production Web: " PASSWORD
echo
if [[ -z "$PASSWORD" ]]; then
  echo "Password cannot be empty" >&2
  exit 2
fi
HASH="$(docker run --rm caddy:2.11.4-alpine caddy hash-password --plaintext "$PASSWORD")"
printf "APP_BASIC_AUTH_HASH='%s'\n" "$HASH"
echo "Copy the line above exactly into .env.production. Single quotes protect the '$' characters from Compose interpolation."
