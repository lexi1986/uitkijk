#!/usr/bin/env bash
# Haalt de modellen voor gezichtsherkenning op en controleert de checksums:
#  - OpenCV Zoo: YuNet (MIT) en SFace (Apache-2.0), ± 40 MB
#  - InsightFace buffalo_l: alleen w600k_r50.onnx (ArcFace, 174 MB; download is een zip van 289 MB).
#    Licentie: alleen niet-commercieel gebruik, prima voor thuis. Zonder dit model valt de
#    service terug op SFace.
# Opnieuw draaien slaat aanwezige modellen over.
#   bash /opt/camera-service/deploy/fetch-face-models.sh
set -euo pipefail

DEST="${DEST:-/opt/camera-service/models}"
BASE=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models
mkdir -p "$DEST"

fetch() {  # pad-in-zoo  sha256
  local file="$DEST/$(basename "$1")"
  if [ -f "$file" ] && echo "$2  $file" | sha256sum -c --status; then return 0; fi
  echo "==> $(basename "$1") ophalen"
  curl -fsSL -o "$file.tmp" "$BASE/$1"
  echo "$2  $file.tmp" | sha256sum -c --status || { rm -f "$file.tmp"; echo "Checksum klopt niet: $1" >&2; exit 1; }
  mv "$file.tmp" "$file"
}

fetch face_detection_yunet/face_detection_yunet_2023mar.onnx 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
fetch face_recognition_sface/face_recognition_sface_2021dec.onnx 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
# Herkennen aan kleding/postuur (YouTu-ReID, OpenCV Zoo, Apache-2.0, ± 107 MB)
fetch person_reid_youtureid/person_reid_youtu_2021nov.onnx 0579683334d4b9440221606dcb461656dd0dc64143b18f48faedaced9b4f580d

ARC_SHA=4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43
if ! { [ -f "$DEST/w600k_r50.onnx" ] && echo "$ARC_SHA  $DEST/w600k_r50.onnx" | sha256sum -c --status; }; then
  echo "==> w600k_r50.onnx (ArcFace) ophalen"
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  curl -fsSL -o "$tmp/buffalo_l.zip" https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
  python3 -c 'import sys, zipfile; open(sys.argv[2], "wb").write(zipfile.ZipFile(sys.argv[1]).read("w600k_r50.onnx"))' \
    "$tmp/buffalo_l.zip" "$tmp/w600k_r50.onnx"
  echo "$ARC_SHA  $tmp/w600k_r50.onnx" | sha256sum -c --status || { echo "Checksum klopt niet: w600k_r50.onnx" >&2; exit 1; }
  mv "$tmp/w600k_r50.onnx" "$DEST/w600k_r50.onnx"
fi
id camsvc >/dev/null 2>&1 && chown camsvc:camsvc "$DEST"/face_*.onnx "$DEST"/w600k_r50.onnx "$DEST"/person_reid_*.onnx
echo "==> Gezichtsmodellen staan in $DEST"
