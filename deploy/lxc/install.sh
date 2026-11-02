#!/usr/bin/env bash
# Installs Uitkijk inside a Debian LXC: packages, the code in /opt/camera-service, a Python venv
# and the systemd service.
#
# Two ways to get the code:
#   - Local (default): copies the folder this script is in (or SRC=/path) to /opt.
#     Updating: from the Proxmox host `CTID=<id> bash deploy/lxc/push-update.sh`.
#   - A GitHub repo: REPO=lexi1986/uitkijk [GITHUB_TOKEN=...] bash install.sh
#     Then a timer also fetches new commits every 5 minutes.
# Google Coral (USB): CORAL=1 installs the driver (libedgetpu) and the TFLite runtime.
# By default (CORAL=auto) that happens as soon as a Coral is attached to the container.
# Running it again is safe (repairs or updates).
set -euo pipefail

REPO="${REPO:-}"
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BRANCH="${BRANCH:-main}"
DIR="${DIR:-/opt/camera-service}"
INTERVAL="${INTERVAL:-5min}"
TOKEN="${GITHUB_TOKEN:-}"
EXPORT_MODEL="${EXPORT_MODEL:-0}"
CORAL="${CORAL:-auto}"
CONF=/etc/camera-service
STATE=/var/lib/camera-service
CLIPS=/clips

[ "$(id -u)" = 0 ] || { echo "Run this as root." >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
# the Debian template only has C.UTF-8; avoids locale warnings
export LANG=C.UTF-8 LC_ALL=C.UTF-8

echo "==> Packages"
apt-get update -qq
apt-get install -y -qq --no-install-recommends git ca-certificates python3 python3-venv \
  ffmpeg libglib2.0-0 ocl-icd-libopencl1 clinfo >/dev/null
# Intel drivers (iGPU for the object model); may be missing, then everything runs on the CPU.
apt-get install -y -qq --no-install-recommends intel-opencl-icd intel-media-va-driver vainfo >/dev/null \
  || echo "Note: Intel drivers not installed; the object model runs on the CPU."

echo "==> Fetching code"
if [ -n "$REPO" ]; then
  if [ -n "$TOKEN" ]; then URL="https://x-access-token:${TOKEN}@github.com/${REPO}.git"; else URL="https://github.com/${REPO}.git"; fi
  if [ -d "$DIR/.git" ]; then
    # without a token, keep the existing remote (with token)
    [ -n "$TOKEN" ] && git -C "$DIR" remote set-url origin "$URL"
    git -C "$DIR" fetch -q origin "$BRANCH"
    git -C "$DIR" reset -q --hard "origin/$BRANCH"
  else
    rm -rf "$DIR"
    git clone -q --branch "$BRANCH" --single-branch "$URL" "$DIR"
  fi
  # The token is in .git/config: only root may read it.
  chmod 700 "$DIR/.git"
else
  [ -f "$SRC/camsvc/__init__.py" ] || { echo "No code found in $SRC (set SRC=/path or REPO=...)." >&2; exit 1; }
  if [ "$(cd "$SRC" && pwd)" != "$DIR" ]; then
    mkdir -p "$DIR"
    # replace everything except venv, models and git
    find "$DIR" -mindepth 1 -maxdepth 1 ! -name .venv ! -name models -exec rm -rf {} +
    tar -C "$SRC" --exclude=.venv --exclude=.git --exclude=models --exclude=__pycache__ \
      --exclude=.pytest_cache -cf - . | tar -C "$DIR" -xf -
  fi
fi

if [ "$CORAL" = auto ]; then
  # Coral USB: 1a6e:089a before and 18d1:9302 after loading the firmware
  CORAL=0
  [ -f "$DIR/models/yolo11n_edgetpu.tflite" ] && CORAL=1
  grep -qsE '^(1a6e|18d1)$' /sys/bus/usb/devices/*/idVendor && CORAL=1
fi

echo "==> User and folders"
id camsvc >/dev/null 2>&1 || useradd --system --home-dir "$DIR" --shell /usr/sbin/nologin camsvc
for g in render video; do getent group "$g" >/dev/null && usermod -aG "$g" camsvc; done
mkdir -p "$CONF" "$STATE" "$CLIPS" "$DIR/models"
# /clips is a separate disk with lost+found in it: only the folder itself, not recursively
chown camsvc:camsvc "$CLIPS"
chown -R camsvc:camsvc "$STATE" "$DIR/models"

echo "==> Python environment"
[ -d "$DIR/.venv" ] || python3 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install -q --upgrade pip
"$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"

# Face recognition models; if this fails, everything else runs normally.
DEST="$DIR/models" bash "$DIR/deploy/fetch-face-models.sh" \
  || echo "Note: face models not downloaded; face recognition is off."

if [ "$CORAL" = 1 ]; then
  echo "==> Google Coral (Edge TPU)"
  source "$DIR/deploy/coral-repo.sh"
  if coral_repo && apt-get install -y -qq --no-install-recommends libedgetpu1-std >/dev/null; then
    "$DIR/.venv/bin/pip" install -q ai-edge-litert
  else
    echo "Note: Coral driver not installed; the object model runs on the CPU."
  fi
fi

# Without a config the service starts the setup wizard, which writes the config itself
[ -f "$CONF/config.yaml" ] || NEW_CONFIG=1
chown root:camsvc "$CONF"
chmod 770 "$CONF"
if [ -f "$CONF/config.yaml" ]; then
  chown root:camsvc "$CONF/config.yaml"
  chmod 660 "$CONF/config.yaml"
fi

echo "==> systemd"
cp "$DIR/deploy/camera-service.service" /etc/systemd/system/camera-service.service

if [ -n "$REPO" ]; then
cat > /usr/local/bin/camera-service-update <<UPD
#!/usr/bin/env bash
# Fetches the latest code; only on a new version: update dependencies and restart.
set -euo pipefail
cd "$DIR"
git fetch -q origin "$BRANCH"
if [ "\$(git rev-parse HEAD)" != "\$(git rev-parse origin/$BRANCH)" ]; then
  OLD_REQ="\$(sha1sum requirements.txt)"
  git reset -q --hard "origin/$BRANCH"
  if [ "\$OLD_REQ" != "\$(sha1sum requirements.txt)" ]; then
    "$DIR/.venv/bin/pip" install -q -r requirements.txt
  fi
  cp deploy/camera-service.service /etc/systemd/system/camera-service.service
  systemctl daemon-reload
  systemctl restart camera-service
  echo "Updated to \$(git rev-parse --short HEAD)"
fi
UPD
chmod 755 /usr/local/bin/camera-service-update

cat > /etc/systemd/system/camera-service-update.service <<SVC
[Unit]
Description=Update Uitkijk from GitHub
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/camera-service-update
SVC
cat > /etc/systemd/system/camera-service-update.timer <<TMR
[Unit]
Description=Check for a new version of Uitkijk every ${INTERVAL}

[Timer]
OnBootSec=2min
OnUnitActiveSec=${INTERVAL}
Persistent=true

[Install]
WantedBy=timers.target
TMR
else
  # local installation: no automatic updates (any more)
  systemctl disable --now camera-service-update.timer >/dev/null 2>&1 || true
  rm -f /etc/systemd/system/camera-service-update.{service,timer} /usr/local/bin/camera-service-update
fi
systemctl daemon-reload
systemctl enable camera-service >/dev/null
[ -n "$REPO" ] && systemctl enable --now camera-service-update.timer >/dev/null

if [ ! -f "$DIR/models/yolo11n.onnx" ] && [ "$EXPORT_MODEL" = 1 ]; then
  echo "==> Building the object model (once, takes a few minutes)"
  bash "$DIR/deploy/export-model.sh"
fi
if [ "$CORAL" = 1 ] && [ ! -f "$DIR/models/yolo11n_edgetpu.tflite" ] && [ "$EXPORT_MODEL" = 1 ]; then
  echo "==> Building the model for the Coral (once, takes longer: TensorFlow is installed temporarily)"
  FORMAT=edgetpu bash "$DIR/deploy/export-model.sh" || echo "Note: building the Coral model failed; the CPU is used."
fi
if [ "$CORAL" = 1 ] && [ -f "$DIR/models/yolo11n_edgetpu.tflite" ] && ! grep -qE '^\s*coral_path:' "$CONF/config.yaml"; then
  # set coral_path under model: (only when it is not there yet)
  sed -i "s|^model:|model:\n  coral_path: $DIR/models/yolo11n_edgetpu.tflite|" "$CONF/config.yaml"
  echo "Coral model set in $CONF/config.yaml (model.coral_path)"
fi

echo
[ -e /dev/dri/renderD128 ] && echo "iGPU: /dev/dri/renderD128 present" || echo "iGPU: not passed through (everything on the CPU)"
[ -f "$DIR/models/yolo11n.onnx" ] || echo "Object model missing: bash $DIR/deploy/export-model.sh (motion detection only until then)"
if [ "${NEW_CONFIG:-0}" = 1 ]; then
  systemctl restart camera-service
  echo "Setup wizard started. Setup code: journalctl -u camera-service | grep 'setup code'"
else
  systemctl restart camera-service
  echo "Service restarted. Logs: journalctl -u camera-service -f"
fi
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "Web page: http://${IP:-<ip-of-the-container>}:8080/"
[ -n "$REPO" ] && echo "Automatic updates every ${INTERVAL} from ${REPO} (log: journalctl -u camera-service-update)"
true
