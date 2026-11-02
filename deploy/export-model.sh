#!/usr/bin/env bash
# Zet YOLO11n om naar ONNX (standaard), Coral (Edge TPU) of OpenVINO-formaat (eenmalig).
# Gebruikt een tijdelijke venv met PyTorch (CPU) en ruimt die daarna op.
# Draai als root in de LXC:
#   bash /opt/camera-service/deploy/export-model.sh                 # yolo11n.onnx (ONNX Runtime, CPU)
#   FORMAT=edgetpu bash /opt/camera-service/deploy/export-model.sh  # yolo11n_edgetpu.tflite (Coral)
#   FORMAT=openvino bash /opt/camera-service/deploy/export-model.sh # yolo11n.xml (OpenVINO, iGPU)
set -euo pipefail

MODEL="${MODEL:-yolo11n}"
SIZE="${SIZE:-320}"
FORMAT="${FORMAT:-onnx}"
DEST=/opt/camera-service/models
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export LANG=C.UTF-8 LC_ALL=C.UTF-8

echo "==> Tijdelijke omgeving (± 1 GB, wordt na afloop verwijderd)"
python3 -m venv "$TMP/venv"
"$TMP/venv/bin/pip" install -q --upgrade pip
"$TMP/venv/bin/pip" install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision
if [ "$FORMAT" = openvino ]; then
  "$TMP/venv/bin/pip" install -q ultralytics "openvino==2026.4.1"
elif [ "$FORMAT" = edgetpu ]; then
  # De Edge TPU-compiler komt uit Google's pakketbron (alleen x86_64). TensorFlow en onnx2tf
  # installeert Ultralytics zelf; de omzetting gebruikt een paar voorbeeldbeelden (coco8).
  source "$(dirname "${BASH_SOURCE[0]}")/coral-repo.sh"
  coral_repo
  apt-get install -y -qq --no-install-recommends edgetpu-compiler >/dev/null
  "$TMP/venv/bin/pip" install -q ultralytics onnx
else
  "$TMP/venv/bin/pip" install -q ultralytics onnx onnxslim
fi

echo "==> $MODEL exporteren naar $FORMAT (invoer ${SIZE}px)"
cd "$TMP"
mkdir -p "$DEST"
if [ "$FORMAT" = openvino ]; then
  "$TMP/venv/bin/yolo" export model="$MODEL.pt" format=openvino imgsz="$SIZE" half=False
  cp "$TMP/${MODEL}_openvino_model/${MODEL}.xml" "$TMP/${MODEL}_openvino_model/${MODEL}.bin" "$DEST/"
  OUT="$DEST/${MODEL}.xml"
elif [ "$FORMAT" = edgetpu ]; then
  "$TMP/venv/bin/yolo" export model="$MODEL.pt" format=edgetpu imgsz="$SIZE"
  SRC_TFLITE="$(find "$TMP" -name '*_edgetpu.tflite' | head -1)"
  [ -n "$SRC_TFLITE" ] || { echo "Omzetten voor de Coral mislukt" >&2; exit 1; }
  cp "$SRC_TFLITE" "$DEST/${MODEL}_edgetpu.tflite"
  OUT="$DEST/${MODEL}_edgetpu.tflite"
else
  "$TMP/venv/bin/yolo" export model="$MODEL.pt" format=onnx imgsz="$SIZE" opset=17 simplify=True
  cp "$TMP/${MODEL}.onnx" "$DEST/"
  OUT="$DEST/${MODEL}.onnx"
fi
id camsvc >/dev/null 2>&1 && chown -R camsvc:camsvc "$DEST"
echo "==> Klaar: $OUT"
if [ "$FORMAT" = edgetpu ]; then
  echo "Zet model.coral_path in /etc/camera-service/config.yaml op $OUT en herstart: systemctl restart camera-service"
else
  echo "Zet model.path in /etc/camera-service/config.yaml op $OUT en herstart: systemctl restart camera-service"
fi
