#!/usr/bin/env bash
set -euo pipefail

cat <<'BANNER'
=== NVIDIA Container Toolkit setup for Fedora Remix on WSL2 ===
This is a HOST/WSL setup helper. It is never run by Docker Compose or bootstrap.
It installs/configures NVIDIA Container Toolkit only; it does NOT install a Linux NVIDIA display driver.
BANNER

if ! grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
  echo 'WARNING: WSL2 kernel was not detected. These commands were prepared for Fedora Remix on WSL2.' >&2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'ERROR: nvidia-smi is not visible in WSL. Fix the Windows NVIDIA driver/WSL GPU path first.' >&2
  exit 1
fi

sudo dnf install -y curl
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
sudo dnf install -y nvidia-container-toolkit

# Configure the Docker daemon with NVIDIA Container Runtime integration.
sudo nvidia-ctk runtime configure --runtime=docker

# Restart the native Docker Engine running in this Fedora WSL distribution.
sudo systemctl restart docker

# NVIDIA Container Toolkit 1.18+ normally refreshes CDI automatically. Trigger it
# when the systemd units exist, then generate once explicitly so Docker 29 can
# discover nvidia.com/gpu devices immediately and reproducibly.
if systemctl list-unit-files 2>/dev/null | grep -q '^nvidia-cdi-refresh\.path'; then
  sudo systemctl enable --now nvidia-cdi-refresh.path || true
fi
if systemctl list-unit-files 2>/dev/null | grep -q '^nvidia-cdi-refresh\.service'; then
  sudo systemctl restart nvidia-cdi-refresh.service || true
fi
sudo mkdir -p /var/run/cdi
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml

echo
echo '=== CDI devices ==='
nvidia-ctk cdi list

echo
echo '=== Docker discovery ==='
docker info | sed -n '/CDI spec directories/,+12p' || true

echo
echo 'Setup finished. Run: ./scripts/gpu-doctor.sh'
