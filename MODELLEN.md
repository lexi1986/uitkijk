# Modellen en hun licenties

camera-service levert zelf geen modelbestanden mee in deze repository. Ze worden bij het bouwen
van het Docker-image of bij de eerste start gedownload van de officiële bron, met een
controle van de checksum.

| Model | Waarvoor | Bron | Licentie | Wanneer |
|---|---|---|---|---|
| YOLO11n | personen, voertuigen, dieren | Ultralytics | AGPL-3.0 | altijd; omgezet naar ONNX bij het bouwen |
| YuNet (2023mar) | gezichten vinden | OpenCV Zoo | MIT | alleen met gezichtsherkenning |
| SFace (2021dec) | gezichten herkennen (reserve) | OpenCV Zoo | Apache-2.0 | alleen met gezichtsherkenning |
| ArcFace w600k_r50 (buffalo_l) | gezichten herkennen | InsightFace | **alleen niet-commercieel** | alleen met gezichtsherkenning |
| YouTu-ReID (2021nov) | herkennen aan kleding/postuur | OpenCV Zoo | Apache-2.0 | alleen met gezichtsherkenning |
| Kentekenmodellen | kentekens vinden en lezen | fast-alpr / open-image-models / fast-plate-ocr | MIT | alleen met kentekens; via pip |

## Wat betekent dat?

- **YOLO11 (AGPL-3.0)** is de reden dat camera-service zelf ook onder AGPL-3.0 valt. Gebruiken,
  aanpassen en gratis delen mag; wie het in een gesloten of betaald product wil stoppen, heeft een
  Enterprise-licentie van Ultralytics nodig.
- **InsightFace-modellen** mogen niet commercieel gebruikt worden. Thuisgebruik is prima. Wil je
  dat niet, laat gezichtsherkenning dan uit of verwijder `w600k_r50.onnx`: de service valt dan
  terug op SFace (Apache-2.0), dat iets minder goed herkent.
- De code van camera-service zelf: AGPL-3.0, zie [`LICENSE`](LICENSE).
