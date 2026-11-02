#!/usr/bin/env bash
# Draai dit op de Proxmox-HOST om een bestaande container bij te werken met de code uit deze map
# (na bijv. `scp -r camera-service root@proxmox:/root/`). Config, clips en model blijven staan.
#
#   CTID=123 bash /root/camera-service/deploy/lxc/push-update.sh
set -euo pipefail

CTID="${CTID:?Zet CTID=<nummer van de camera-service-container>}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
command -v pct >/dev/null || { echo "Dit script moet op een Proxmox-host draaien." >&2; exit 1; }

TARBALL="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "$TARBALL"' EXIT
tar -C "$SRC" --exclude=.venv --exclude=.git --exclude=models --exclude=__pycache__ \
  --exclude=.pytest_cache -czf "$TARBALL" .
pct push "$CTID" "$TARBALL" /root/camera-service.tar.gz
pct exec "$CTID" -- bash -c 'rm -rf /root/camera-service-src && mkdir -p /root/camera-service-src && tar -C /root/camera-service-src -xzf /root/camera-service.tar.gz'
pct exec "$CTID" -- env SRC=/root/camera-service-src bash /root/camera-service-src/deploy/lxc/install.sh
