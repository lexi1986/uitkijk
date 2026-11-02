# Uitkijk in Docker: recording, detection and the web page in one container.
# The object model (YOLO11n) is converted to ONNX at build time; face models are only downloaded
# at first start when face recognition is enabled (see camsvc/models.py and MODELS.md).

# --- stage 1: YOLO11n to ONNX (PyTorch is only needed here, not in the final image) ----------
FROM python:3.11-slim-bookworm AS model
WORKDIR /m
# Ultralytics pulls in the regular OpenCV, which wants X11/GL libraries (only in this stage)
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install --no-cache-dir ultralytics onnx onnxslim \
 && yolo export model=yolo11n.pt format=onnx imgsz=320 opset=17 simplify=True \
 && mkdir -p /out && cp yolo11n.onnx /out/

# --- stage 2: the service ----------------------------------------------------------------------
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CAMSVC_CONFIG=/data/config.yaml CAMSVC_STATE=/data/state CAMSVC_CLIPS=/clips \
    CAMSVC_MODEL=/data/models/yolo11n.onnx CAMSVC_BUNDLED_MODELS=/opt/camsvc-models HOME=/data
# libglib for OpenCV; VA-API drivers for hardware decoding on Intel (optional, via /dev/dri)
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends libglib2.0-0 ca-certificates intel-media-va-driver vainfo \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/camera-service
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=model /out/yolo11n.onnx /opt/camsvc-models/yolo11n.onnx
COPY camsvc ./camsvc
COPY config.example.yaml ./
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN useradd --system --uid 1000 --home-dir /data camsvc && mkdir -p /data /clips && chown camsvc:camsvc /data /clips
VOLUME ["/data", "/clips"]
EXPOSE 8080
# after saving the wizard the process exits; Docker starts it again (restart policy)
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "camsvc"]
