#!/usr/bin/env bash
# Installeert camera-service binnen een Debian-LXC: pakketten, de code in /opt/camera-service,
# een Python-venv en de systemd-service.
#
# Twee manieren om de code te krijgen:
#   - Lokaal (standaard): kopieert de map waar dit script in staat (of SRC=/pad) naar /opt.
#     Bijwerken: vanaf de Proxmox-host `CTID=<id> bash deploy/lxc/push-update.sh`.
#   - Eigen GitHub-repo: REPO=gebruiker/camera-service [GITHUB_TOKEN=...] bash install.sh
#     Dan komt er ook een timer die elke 5 minuten nieuwe commits ophaalt.
# Google Coral (USB): CORAL=1 installeert de driver (libedgetpu) en de TFLite-runtime.
# Standaard (CORAL=auto) gebeurt dat zodra er een Coral aan de container hangt.
# Opnieuw draaien is veilig (herstelt of werkt bij).
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

[ "$(id -u)" = 0 ] || { echo "Draai dit als root." >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
# Debian-template heeft alleen C.UTF-8; voorkomt locale-waarschuwingen
export LANG=C.UTF-8 LC_ALL=C.UTF-8

echo "==> Pakketten"
apt-get update -qq
apt-get install -y -qq --no-install-recommends git ca-certificates python3 python3-venv \
  ffmpeg libglib2.0-0 ocl-icd-libopencl1 clinfo >/dev/null
# Intel-drivers (iGPU voor het objectmodel); mogen ontbreken, dan draait alles op de CPU.
apt-get install -y -qq --no-install-recommends intel-opencl-icd intel-media-va-driver vainfo >/dev/null \
  || echo "Let op: Intel-drivers niet geïnstalleerd; objectmodel draait op de CPU."

echo "==> Code ophalen"
if [ -n "$REPO" ]; then
  if [ -n "$TOKEN" ]; then URL="https://x-access-token:${TOKEN}@github.com/${REPO}.git"; else URL="https://github.com/${REPO}.git"; fi
  if [ -d "$DIR/.git" ]; then
    # zonder token de bestaande remote (met token) laten staan
    [ -n "$TOKEN" ] && git -C "$DIR" remote set-url origin "$URL"
    git -C "$DIR" fetch -q origin "$BRANCH"
    git -C "$DIR" reset -q --hard "origin/$BRANCH"
  else
    rm -rf "$DIR"
    git clone -q --branch "$BRANCH" --single-branch "$URL" "$DIR"
  fi
  # Het token staat in .git/config: alleen root mag dat lezen.
  chmod 700 "$DIR/.git"
else
  [ -f "$SRC/camsvc/__init__.py" ] || { echo "Geen code gevonden in $SRC (zet SRC=/pad of REPO=...)." >&2; exit 1; }
  if [ "$(cd "$SRC" && pwd)" != "$DIR" ]; then
    mkdir -p "$DIR"
    # alles behalve venv, modellen en git vervangen
    find "$DIR" -mindepth 1 -maxdepth 1 ! -name .venv ! -name models -exec rm -rf {} +
    tar -C "$SRC" --exclude=.venv --exclude=.git --exclude=models --exclude=__pycache__ \
      --exclude=.pytest_cache -cf - . | tar -C "$DIR" -xf -
  fi
fi

if [ "$CORAL" = auto ]; then
  # Coral USB: 1a6e:089a vóór en 18d1:9302 na het laden van de firmware
  CORAL=0
  [ -f "$DIR/models/yolo11n_edgetpu.tflite" ] && CORAL=1
  grep -qsE '^(1a6e|18d1)$' /sys/bus/usb/devices/*/idVendor && CORAL=1
fi

echo "==> Gebruiker en mappen"
id camsvc >/dev/null 2>&1 || useradd --system --home-dir "$DIR" --shell /usr/sbin/nologin camsvc
for g in render video; do getent group "$g" >/dev/null && usermod -aG "$g" camsvc; done
mkdir -p "$CONF" "$STATE" "$CLIPS" "$DIR/models"
# /clips is een eigen schijf met lost+found erin: alleen de map zelf, niet recursief
chown camsvc:camsvc "$CLIPS"
chown -R camsvc:camsvc "$STATE" "$DIR/models"

echo "==> Python-omgeving"
[ -d "$DIR/.venv" ] || python3 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install -q --upgrade pip
"$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"

# Gezichtsherkenning (klein, ± 40 MB); mislukt het, dan draait de rest gewoon door.
DEST="$DIR/models" bash "$DIR/deploy/fetch-face-models.sh" \
  || echo "Let op: gezichtsmodellen niet opgehaald; gezichtsherkenning staat uit."

if [ "$CORAL" = 1 ]; then
  echo "==> Google Coral (Edge TPU)"
  source "$DIR/deploy/coral-repo.sh"
  if coral_repo && apt-get install -y -qq --no-install-recommends libedgetpu1-std >/dev/null; then
    "$DIR/.venv/bin/pip" install -q ai-edge-litert
  else
    echo "Let op: Coral-driver niet geïnstalleerd; objectmodel draait op de CPU."
  fi
fi

if [ ! -f "$CONF/config.yaml" ]; then
  cp "$DIR/config.example.yaml" "$CONF/config.yaml"
  NEW_CONFIG=1
fi
chown root:camsvc "$CONF/config.yaml"
chmod 640 "$CONF/config.yaml"

echo "==> systemd"
cp "$DIR/deploy/camera-service.service" /etc/systemd/system/camera-service.service

if [ -n "$REPO" ]; then
cat > /usr/local/bin/camera-service-update <<UPD
#!/usr/bin/env bash
# Haalt de laatste code op; alleen bij een nieuwe versie: dependencies bijwerken en herstarten.
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
  echo "Bijgewerkt naar \$(git rev-parse --short HEAD)"
fi
UPD
chmod 755 /usr/local/bin/camera-service-update

cat > /etc/systemd/system/camera-service-update.service <<SVC
[Unit]
Description=camera-service bijwerken vanaf GitHub
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/camera-service-update
SVC
cat > /etc/systemd/system/camera-service-update.timer <<TMR
[Unit]
Description=Controleer elke ${INTERVAL} op een nieuwe versie van camera-service

[Timer]
OnBootSec=2min
OnUnitActiveSec=${INTERVAL}
Persistent=true

[Install]
WantedBy=timers.target
TMR
else
  # lokale installatie: geen automatische updates (meer)
  systemctl disable --now camera-service-update.timer >/dev/null 2>&1 || true
  rm -f /etc/systemd/system/camera-service-update.{service,timer} /usr/local/bin/camera-service-update
fi
systemctl daemon-reload
systemctl enable camera-service >/dev/null
[ -n "$REPO" ] && systemctl enable --now camera-service-update.timer >/dev/null

if [ ! -f "$DIR/models/yolo11n.onnx" ] && [ "$EXPORT_MODEL" = 1 ]; then
  echo "==> Objectmodel maken (eenmalig, duurt een paar minuten)"
  bash "$DIR/deploy/export-model.sh"
fi
if [ "$CORAL" = 1 ] && [ ! -f "$DIR/models/yolo11n_edgetpu.tflite" ] && [ "$EXPORT_MODEL" = 1 ]; then
  echo "==> Model voor de Coral maken (eenmalig, duurt langer: TensorFlow wordt tijdelijk geïnstalleerd)"
  FORMAT=edgetpu bash "$DIR/deploy/export-model.sh" || echo "Let op: Coral-model maken mislukt; de CPU wordt gebruikt."
fi
if [ "$CORAL" = 1 ] && [ -f "$DIR/models/yolo11n_edgetpu.tflite" ] && ! grep -qE '^\s*coral_path:' "$CONF/config.yaml"; then
  # coral_path onder model: zetten (alleen als het er nog niet staat)
  sed -i "s|^model:|model:\n  coral_path: $DIR/models/yolo11n_edgetpu.tflite|" "$CONF/config.yaml"
  echo "Coral-model ingesteld in $CONF/config.yaml (model.coral_path)"
fi

echo
[ -e /dev/dri/renderD128 ] && echo "iGPU: /dev/dri/renderD128 aanwezig" || echo "iGPU: niet doorgegeven (alles op de CPU)"
[ -f "$DIR/models/yolo11n.onnx" ] || echo "Objectmodel ontbreekt: bash $DIR/deploy/export-model.sh (tot dan alleen bewegingsdetectie)"
if [ "${NEW_CONFIG:-0}" = 1 ]; then
  echo "Vul eerst de config in: nano $CONF/config.yaml  — en start dan: systemctl start camera-service"
else
  systemctl restart camera-service
  echo "camera-service herstart. Logs: journalctl -u camera-service -f"
fi
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "Status: http://${IP:-<ip-van-de-container>}:8080/api/status"
[ -n "$REPO" ] && echo "Automatische updates elke ${INTERVAL} vanaf ${REPO} (log: journalctl -u camera-service-update)"
true
