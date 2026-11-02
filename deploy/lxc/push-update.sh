#!/usr/bin/env bash
# Run this on the Proxmox HOST to update an existing container with the code from this folder
# (after e.g. `rsync -a ./ root@proxmox:/root/camera-service/`). Config, clips and model stay.
#
#   CTID=123 bash /root/camera-service/deploy/lxc/push-update.sh
set -euo pipefail

CTID="${CTID:?Zet CTID=<nummer van de camera-service-container>}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
command -v pct >/dev/null || { echo "This script must run on a Proxmox host." >&2; exit 1; }

TARBALL="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "$TARBALL"' EXIT
tar -C "$SRC" --exclude=.venv --exclude=.git --exclude=models --exclude=__pycache__ \
  --exclude=.pytest_cache -czf "$TARBALL" .
pct push "$CTID" "$TARBALL" /root/camera-service.tar.gz
pct exec "$CTID" -- bash -c 'rm -rf /root/camera-service-src && mkdir -p /root/camera-service-src && tar -C /root/camera-service-src -xzf /root/camera-service.tar.gz'
pct exec "$CTID" -- env SRC=/root/camera-service-src bash /root/camera-service-src/deploy/lxc/install.sh
