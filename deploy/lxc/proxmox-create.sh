#!/usr/bin/env bash
# Draai dit op de Proxmox-HOST (shell van de node): maakt een Debian-LXC en installeert daarin
# camera-service, met clipschijf (buiten de backup) en iGPU-doorgifte.
#
# Standaard wordt de code uit deze map gebruikt (kopieer de hele map naar de host):
#   scp -r camera-service root@proxmox:/root/ && ssh root@proxmox
#   bash /root/camera-service/deploy/lxc/proxmox-create.sh
# Of vanuit je eigen GitHub-repo, met automatische updates:
#   REPO=gebruiker/camera-service GITHUB_TOKEN=github_pat_xxx bash proxmox-create.sh
#
# Optionele variabelen: CTID, CT_NAME, STORAGE (rootfs, standaard local-lvm), CLIPS_STORAGE
# (standaard = STORAGE), CLIPS_GB (32), TEMPLATE_STORAGE (local), BRIDGE (vmbr0),
# IP (dhcp of 192.168.1.50/24), GW, VLAN, CORES (2), MEMORY (1024), DISK (8),
# GPU (1 = /dev/dri/renderD128 doorgeven, 0 = niet), EXPORT_MODEL (1 = YOLO11n meteen maken),
# CORAL (1 = Google Coral USB doorgeven en het model ervoor maken).
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

command -v pct >/dev/null || { echo "Dit script moet op een Proxmox-host draaien." >&2; exit 1; }

NET="name=eth0,bridge=${BRIDGE},ip=${IP}"
[ "$IP" != dhcp ] && [ -n "${GW:-}" ] && NET="${NET},gw=${GW}"
[ -n "${VLAN:-}" ] && NET="${NET},tag=${VLAN}"

echo "Template zoeken en zo nodig downloaden..."
pveam update >/dev/null
# Debian 12: de Intel OpenCL-driver daarin ondersteunt de HD 530 (6e generatie) nog.
TEMPLATE="$(pveam available --section system | awk '/debian-12-standard/ {print $2}' | sort -V | tail -1)"
[ -n "$TEMPLATE" ] || { echo "Geen Debian 12-template gevonden." >&2; exit 1; }
pveam list "$TEMPLATE_STORAGE" | grep -q "$TEMPLATE" || pveam download "$TEMPLATE_STORAGE" "$TEMPLATE"

echo "Container ${CTID} (${CT_NAME}) maken..."
pct create "$CTID" "${TEMPLATE_STORAGE}:vztmpl/${TEMPLATE}" \
  --hostname "$CT_NAME" --unprivileged 1 --onboot 1 \
  --cores "$CORES" --memory "$MEMORY" --swap 256 \
  --rootfs "${STORAGE}:${DISK}" --net0 "$NET" \
  --mp0 "${CLIPS_STORAGE}:${CLIPS_GB},mp=/clips,backup=0" \
  --features nesting=1 --ostype debian
pct start "$CTID"

echo "Wachten op netwerk in de container..."
for _ in $(seq 1 30); do
  pct exec "$CTID" -- getent hosts github.com >/dev/null 2>&1 && break
  sleep 2
done

if [ "$GPU" = 1 ]; then
  if [ -e "$RENDER_NODE" ]; then
    RENDER_GID="$(pct exec "$CTID" -- getent group render | cut -d: -f3)"
    if [ -n "$RENDER_GID" ]; then
      echo "iGPU doorgeven (${RENDER_NODE}, groep render = ${RENDER_GID})..."
      pct set "$CTID" --dev0 "${RENDER_NODE},gid=${RENDER_GID}"
      pct reboot "$CTID"
      for _ in $(seq 1 30); do
        pct exec "$CTID" -- getent hosts github.com >/dev/null 2>&1 && break
        sleep 2
      done
    else
      echo "Let op: groep 'render' niet gevonden in de container; iGPU niet doorgegeven." >&2
    fi
  else
    echo "Let op: ${RENDER_NODE} bestaat niet op deze host; iGPU niet doorgegeven." >&2
  fi
fi

if [ "$CORAL" = 1 ]; then
  echo "Google Coral (USB) doorgeven..."
  # De Coral krijgt na het laden van de firmware een ander USB-ID (1a6e:089a → 18d1:9302) en
  # apparaatnummer; daarom de hele USB-bus doorgeven en beide ID's leesbaar maken.
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
  echo "Installatiescript ophalen van ${REPO} en draaien..."
  AUTH=()
  [ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer ${TOKEN}")
  curl -fsSL "${AUTH[@]}" "https://raw.githubusercontent.com/${REPO}/${BRANCH}/deploy/lxc/install.sh" -o /tmp/camsvc-install.sh
  pct push "$CTID" /tmp/camsvc-install.sh /root/install.sh --perms 755
  rm -f /tmp/camsvc-install.sh
  pct exec "$CTID" -- env GITHUB_TOKEN="$TOKEN" REPO="$REPO" BRANCH="$BRANCH" EXPORT_MODEL="$EXPORT_MODEL" CORAL="$CORAL" bash /root/install.sh
else
  [ -f "$SRC/camsvc/__init__.py" ] || { echo "Geen code naast dit script gevonden ($SRC)." >&2; exit 1; }
  echo "Code uit ${SRC} naar de container kopiëren en installeren..."
  TARBALL="$(mktemp --suffix=.tar.gz)"
  tar -C "$SRC" --exclude=.venv --exclude=.git --exclude=models --exclude=__pycache__ \
    --exclude=.pytest_cache -czf "$TARBALL" .
  pct push "$CTID" "$TARBALL" /root/camera-service.tar.gz
  rm -f "$TARBALL"
  pct exec "$CTID" -- bash -c 'rm -rf /root/camera-service-src && mkdir -p /root/camera-service-src && tar -C /root/camera-service-src -xzf /root/camera-service.tar.gz'
  pct exec "$CTID" -- env EXPORT_MODEL="$EXPORT_MODEL" CORAL="$CORAL" SRC=/root/camera-service-src bash /root/camera-service-src/deploy/lxc/install.sh
fi

echo
echo "Container ${CTID} is klaar. Volgende stap: de config invullen:"
echo "  pct exec ${CTID} -- nano /etc/camera-service/config.yaml"
echo "  pct exec ${CTID} -- systemctl restart camera-service"
