# Uitkijk (camera-service) in Docker: opnemen, herkennen en de webpagina in één container.
# Het objectmodel (YOLO11n) wordt bij het bouwen omgezet naar ONNX; gezichtsmodellen worden pas bij
# de eerste start gedownload als je gezichtsherkenning aanzet (zie camsvc/models.py en MODELLEN.md).

# --- stap 1: YOLO11n naar ONNX (PyTorch is alleen hier nodig, niet in het eindimage) ---------
FROM python:3.11-slim-bookworm AS model
WORKDIR /m
# Ultralytics haalt de gewone OpenCV binnen, die X11/GL-bibliotheken wil (alleen in deze stap)
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install --no-cache-dir ultralytics onnx onnxslim \
 && yolo export model=yolo11n.pt format=onnx imgsz=320 opset=17 simplify=True \
 && mkdir -p /out && cp yolo11n.onnx /out/

# --- stap 2: de service ------------------------------------------------------------------------
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CAMSVC_CONFIG=/data/config.yaml CAMSVC_STATE=/data/state CAMSVC_CLIPS=/clips \
    CAMSVC_MODEL=/data/models/yolo11n.onnx CAMSVC_BUNDLED_MODELS=/opt/camsvc-models HOME=/data
# libglib voor OpenCV; VA-API-drivers voor hardware-decoding op Intel (optioneel, via /dev/dri)
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
# na het opslaan van de wizard stopt het proces; Docker start het dan opnieuw (restart-policy)
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "camsvc"]
