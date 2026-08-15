#!/usr/bin/env bash
set -euo pipefail

printf '=== Mammography AI Agent GPU host doctor ===\n'
printf 'This script only diagnoses the HOST/WSL Docker GPU path; it does not change the system.\n\n'

fail=0

printf '=== WSL / kernel ===\n'
uname -a || true
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
  echo 'WSL2 kernel detected: yes'
else
  echo 'WSL2 kernel detected: no/unknown'
fi

printf '\n=== Host NVIDIA ===\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo 'ERROR: nvidia-smi is not available in this WSL distribution.'
  fail=1
fi

printf '\n=== NVIDIA Container Toolkit ===\n'
if command -v nvidia-ctk >/dev/null 2>&1; then
  nvidia-ctk --version || true
  echo
  echo 'CDI devices:'
  if ! nvidia-ctk cdi list; then
    echo 'ERROR: NVIDIA Container Toolkit is installed, but CDI devices could not be listed.'
    fail=1
  fi
else
  echo 'ERROR: nvidia-ctk not found. Install NVIDIA Container Toolkit in the WSL Linux distribution that runs dockerd.'
  fail=1
fi

printf '\n=== CDI specifications ===\n'
for p in /var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml; do
  if [ -f "$p" ]; then
    echo "FOUND: $p"
  else
    echo "MISSING: $p"
  fi
done

printf '\n=== Docker GPU discovery ===\n'
if command -v docker >/dev/null 2>&1; then
  docker info 2>/dev/null | sed -n '/CDI spec directories/,+12p' || true
  if docker info 2>/dev/null | grep -q 'nvidia.com/gpu'; then
    echo 'Docker discovered NVIDIA CDI devices: yes'
  else
    echo 'Docker discovered NVIDIA CDI devices: no'
    fail=1
  fi
else
  echo 'ERROR: docker CLI not found.'
  fail=1
fi

printf '\n=== Result ===\n'
if [ "$fail" -eq 0 ]; then
  echo 'GPU_HOST_READY'
  exit 0
fi

echo 'GPU_HOST_NOT_READY'
echo 'See docs/MIGRATION_V0_7.md for the Fedora Remix / WSL2 setup procedure.'
exit 1
