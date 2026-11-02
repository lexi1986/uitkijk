# Models and their licences

Uitkijk does not ship any model files in this repository. They are downloaded from the official
source while building the Docker image or at first start, with a checksum check.

| Model | Used for | Source | Licence | When |
|---|---|---|---|---|
| YOLO11n | people, vehicles, animals | Ultralytics | AGPL-3.0 | always; converted to ONNX at build time |
| YuNet (2023mar) | finding faces | OpenCV Zoo | MIT | only with face recognition |
| SFace (2021dec) | recognising faces (fallback) | OpenCV Zoo | Apache-2.0 | only with face recognition |
| ArcFace w600k_r50 (buffalo_l) | recognising faces | InsightFace | **non-commercial only** | only with face recognition |
| YouTu-ReID (2021nov) | recognising clothing/build | OpenCV Zoo | Apache-2.0 | only with face recognition |
| Licence plate models | finding and reading plates | fast-alpr / open-image-models / fast-plate-ocr | MIT | only with plates; via pip |

## What does that mean?

- **YOLO11 (AGPL-3.0)** is why Uitkijk itself is AGPL-3.0 too. Using, changing and sharing it for
  free is fine; putting it in a closed or paid product requires an Enterprise licence from Ultralytics.
- **InsightFace models** may not be used commercially. Home use is fine. If you do not want that,
  keep face recognition off or remove `w600k_r50.onnx`: the service then falls back to SFace
  (Apache-2.0), which recognises a little less well.
- The code of Uitkijk itself: AGPL-3.0, see [`LICENSE`](LICENSE).
