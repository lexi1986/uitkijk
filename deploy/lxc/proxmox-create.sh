#!/usr/bin/env bash
# Run this on the Proxmox HOST (the node's shell): creates a Debian LXC and installs Uitkijk in
# it, with a clip disk (outside the backup) and iGPU passthrough.
#
# By default the code from this folder is used (copy the whole folder to the host):
#   rsync -a ./ root@proxmox:/root/camera-service/ && ssh root@proxmox
#   bash /root/camera-service/deploy/lxc/proxmox-create.sh
# Or from a GitHub repo, with automatic updates:
#   REPO=lexi1986/uitkijk bash proxmox-create.sh
#
# Optional variables: CTID, CT_NAME, STORAGE (rootfs, default local-lvm), CLIPS_STORAGE
# (default = STORAGE), CLIPS_GB (32), TEMPLATE_STORAGE (local), BRIDGE (vmbr0),
# IP (dhcp or 192.168.1.50/24), GW, VLAN, CORES (2), MEMORY (4096), DISK (8),
# GPU (1 = pass /dev/dri/renderD128 through, 0 = not), EXPORT_MODEL (1 = build YOLO11n right away),
# CORAL (1 = pass a Google Coral USB through and build the model for it).
set -euo pipefail

REPO="${REPO:-}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH="${BRANCH:-main}"
TOKEN="${GITHUB_TOKEN:-}"
CTID="${CTID:-$(pvesh get /cluster/nextid)}"
CT_NAME="${CT_NAME:-camera-service}"
STORAGE="${STORAGE:-local-lvm}"
CLIPS_STORAGE="${CLIPS_STORAGE:-$STORAGE}"
CLIPS_GB="${CLIPS_GB:-32}"
TEMPLATE_STORAGE="${TEMPLATE_STORAGE:-local}"
BRIDGE="${BRIDGE:-vmbr0}"
IP="${IP:-dhcp}"
CORES="${CORES:-2}"
MEMORY="${MEMORY:-4096}"
DISK="${DISK:-8}"
GPU="${GPU:-1}"
CORAL="${CORAL:-0}"
EXPORT_MODEL="${EXPORT_MODEL:-1}"
RENDER_NODE=/dev/dri/renderD128

command -v pct >/dev/null || { echo "This script must run on a Proxmox host." >&2; exit 1; }

NET="name=eth0,bridge=${BRIDGE},ip=${IP}"
[ "$IP" != dhcp ] && [ -n "${GW:-}" ] && NET="${NET},gw=${GW}"
[ -n "${VLAN:-}" ] && NET="${NET},tag=${VLAN}"

echo "Looking for the template and downloading it if needed..."
pveam update >/dev/null
# Debian 12: its Intel OpenCL driver still supports the HD 530 (6th generation).
TEMPLATE="$(pveam available --section system | awk '/debian-12-standard/ {print $2}' | sort -V | tail -1)"
[ -n "$TEMPLATE" ] || { echo "No Debian 12 template found." >&2; exit 1; }
pveam list "$TEMPLATE_STORAGE" | grep -q "$TEMPLATE" || pveam download "$TEMPLATE_STORAGE" "$TEMPLATE"

echo "Creating container ${CTID} (${CT_NAME})..."
pct create "$CTID" "${TEMPLATE_STORAGE}:vztmpl/${TEMPLATE}" \
  --hostname "$CT_NAME" --unprivileged 1 --onboot 1 \
  --cores "$CORES" --memory "$MEMORY" --swap 256 \
  --rootfs "${STORAGE}:${DISK}" --net0 "$NET" \
  --mp0 "${CLIPS_STORAGE}:${CLIPS_GB},mp=/clips,backup=0" \
  --features nesting=1 --ostype debian
pct start "$CTID"

echo "Waiting for the network in the container..."
for _ in $(seq 1 30); do
  pct exec "$CTID" -- getent hosts github.com >/dev/null 2>&1 && break
  sleep 2
done

if [ "$GPU" = 1 ]; then
  if [ -e "$RENDER_NODE" ]; then
    RENDER_GID="$(pct exec "$CTID" -- getent group render | cut -d: -f3)"
    if [ -n "$RENDER_GID" ]; then
      echo "Passing the iGPU through (${RENDER_NODE}, group render = ${RENDER_GID})..."
      pct set "$CTID" --dev0 "${RENDER_NODE},gid=${RENDER_GID}"
      pct reboot "$CTID"
      for _ in $(seq 1 30); do
        pct exec "$CTID" -- getent hosts github.com >/dev/null 2>&1 && break
        sleep 2
      done
    else
      echo "Note: group 'render' not found in the container; iGPU not passed through." >&2
    fi
  else
    echo "Note: ${RENDER_NODE} does not exist on this host; iGPU not passed through." >&2
  fi
fi

if [ "$CORAL" = 1 ]; then
  echo "Passing the Google Coral (USB) through..."
  # After loading its firmware the Coral gets another USB ID (1a6e:089a → 18d1:9302) and device
  # number; so pass the whole USB bus through and make both IDs readable.
  cat > /etc/udev/rules.d/99-coral-edgetpu.rules <<'UDEV'
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1a6e", ATTRS{idProduct}=="089a", MODE="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="9302", MODE="0666"
UDEV
  udevadm control --reload-rules && udevadm trigger
  CONF_FILE="/etc/pve/lxc/${CTID}.conf"
  grep -q 'dev/bus/usb' "$CONF_FILE" || cat >> "$CONF_FILE" <<'LXC'
lxc.cgroup2.devices.allow: c 189:* rwm
lxc.mount.entry: /dev/bus/usb dev/bus/usb none bind,optional,create=dir
LXC
  pct reboot "$CTID"
  for _ in $(seq 1 30); do
    pct exec "$CTID" -- getent hosts github.com >/dev/null 2>&1 && break
    sleep 2
  done
fi

if [ -n "$REPO" ]; then
  echo "Fetching the install script from ${REPO} and running it..."
  AUTH=()
  [ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer ${TOKEN}")
  curl -fsSL "${AUTH[@]}" "https://raw.githubusercontent.com/${REPO}/${BRANCH}/deploy/lxc/install.sh" -o /tmp/camsvc-install.sh
  pct push "$CTID" /tmp/camsvc-install.sh /root/install.sh --perms 755
  rm -f /tmp/camsvc-install.sh
  pct exec "$CTID" -- env GITHUB_TOKEN="$TOKEN" REPO="$REPO" BRANCH="$BRANCH" EXPORT_MODEL="$EXPORT_MODEL" CORAL="$CORAL" bash /root/install.sh
else
  [ -f "$SRC/camsvc/__init__.py" ] || { echo "No code found next to this script ($SRC)." >&2; exit 1; }
  echo "Copying the code from ${SRC} to the container and installing..."
  TARBALL="$(mktemp --suffix=.tar.gz)"
  tar -C "$SRC" --exclude=.venv --exclude=.git --exclude=models --exclude=__pycache__ \
    --exclude=.pytest_cache -czf "$TARBALL" .
  pct push "$CTID" "$TARBALL" /root/camera-service.tar.gz
  rm -f "$TARBALL"
  pct exec "$CTID" -- bash -c 'rm -rf /root/camera-service-src && mkdir -p /root/camera-service-src && tar -C /root/camera-service-src -xzf /root/camera-service.tar.gz'
  pct exec "$CTID" -- env EXPORT_MODEL="$EXPORT_MODEL" CORAL="$CORAL" SRC=/root/camera-service-src bash /root/camera-service-src/deploy/lxc/install.sh
fi

echo
echo "Container ${CTID} is ready. Next step: open the web page on port 8080 and run the setup wizard."
echo "  Setup code: pct exec ${CTID} -- journalctl -u camera-service | grep 'setup code'"
