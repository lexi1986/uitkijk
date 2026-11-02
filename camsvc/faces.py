"""Gezichtsherkenning, volledig lokaal.

YuNet (OpenCV Zoo) zoekt gezichten en lijnt ze uit naar 112x112. Daarvan maakt ArcFace
(InsightFace w600k_r50, via onnxruntime) een kenmerkvector; ontbreekt dat model, dan
valt het terug op SFace (OpenCV Zoo). Beide gebruiken dezelfde uitlijning, dus de
opgeslagen uitsneden werken voor allebei.

Bekende personen staan als uitgelijnde 112x112-uitsneden in
<state_dir>/faces/<naam>/<id>.jpg; zelf bijgeleerde foto's in .../<naam>/auto/<id>.jpg.
Kenmerken worden per model in faces/.emb-<model>.json bewaard, zodat opstarten snel is.
Gezichten die bij niemand horen verzamelt UnknownFaces in <state_dir>/faces_unknown,
gegroepeerd per persoon, zodat je ze later in één keer een naam kunt geven.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"
ARCFACE_FILE = "w600k_r50.onnx"
MIN_FACE_PX = 36          # kleinere gezichten zijn te onbetrouwbaar
DETECT_MAX_SIDE = 640     # invoer voor YuNet verkleinen tot maximaal zoveel pixels
NAME_RE = re.compile(r"^[\w][\w \-]{0,39}$")
ID_RE = re.compile(r"[0-9a-f]{12}")

AUTO_MAX = 20             # zelf bijgeleerde foto's per persoon (oudste gaan eruit)
AUTO_QUALITY = 0.5        # alleen goede beelden bijleren ...
AUTO_MARGIN = 0.15        # ... die duidelijk bij één persoon horen
AUTO_DUPLICATE = 0.85     # lijkt al zo sterk op een bestaande foto: voegt niets toe
UNKNOWN_QUALITY = 0.3     # slechtere onbekende gezichten niet bewaren
UNKNOWN_MAX = 400
UNKNOWN_KEEP_DAYS = 30
ASSIGN_MAX = 8            # zoveel van de beste foto's gaan mee bij een naam geven


@dataclass(frozen=True)
class Backend:
    name: str
    match: float    # cosinus-gelijkenis vanaf hier = deze persoon ...
    margin: float   # ... en zoveel hoger dan de beste andere persoon
    learn: float    # vanaf hier zeker genoeg om zelf bij te leren
    group: float    # onbekende gezichten vanaf hier als dezelfde persoon samenvoegen


# SFace: 0.363 = zelfde persoon volgens de paper; iets strenger tegen verwisselingen.
SFACE = Backend("sface", 0.42, 0.05, 0.60, 0.45)
# ArcFace: andere personen scoren hier rond 0.0-0.1, dezelfde persoon met camerabeelden 0.2-0.8.
ARCFACE = Backend("arcface", 0.26, 0.08, 0.45, 0.30)


@dataclass
class FaceMatch:
    name: str | None          # None = onbekend
    score: float              # cosinus-gelijkenis met de best passende persoon
    crop_jpg: bytes           # uitgelijnde 112x112-uitsnede, om later mee te leren
    quality: float = 0.0      # 0-1: grootte, scherpte en hoe recht van voren
    margin: float = 0.0       # voorsprong op de beste andere persoon
    embedding: np.ndarray | None = field(default=None, repr=False)


@dataclass
class FaceGroup:
    """Alle beelden van één persoon binnen een event, met de gekozen naam."""
    name: str | None
    score: float
    best: FaceMatch                         # beste beeld (hoogste kwaliteit)
    members: list[FaceMatch]


def valid_name(name: str) -> bool:
    return bool(NAME_RE.match(name)) and name.strip() == name


def face_quality(face: np.ndarray, aligned: np.ndarray) -> float:
    """Ruwe kwaliteit van een YuNet-detectie (rij van 15 waarden) en de uitsnede."""
    size = np.clip((min(face[2], face[3]) - MIN_FACE_PX) / (96 - MIN_FACE_PX), 0, 1)
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    sharp = np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 150.0, 0, 1)
    rx, ry, lx, ly, nx = face[4], face[5], face[6], face[7], face[8]
    eyes = max(1.0, float(np.hypot(lx - rx, ly - ry)))
    frontal = np.clip(1 - abs(nx - (rx + lx) / 2) / eyes / 0.6, 0, 1)   # neus midden tussen de ogen
    conf = np.clip((face[14] - 0.8) / 0.2, 0, 1)
    q = (0.3 + 0.7 * size) * (0.3 + 0.7 * sharp) * (0.2 + 0.8 * frontal) * (0.6 + 0.4 * conf)
    return round(float(q), 3)


def summarize(matches: list[FaceMatch], group_threshold: float, limit: int = 4) -> list[FaceGroup]:
    """Beelden uit één event per persoon groeperen en per groep een naam kiezen.

    Elke poging levert een paar beelden op; door ze samen te nemen telt het scherpste,
    meest frontale beeld het zwaarst en geeft één toevallige misser geen verkeerde naam."""
    groups: list[list[FaceMatch]] = []
    for m in sorted(matches, key=lambda m: -m.quality):
        home = None
        for g in groups:
            if (m.name and any(o.name == m.name for o in g)) or (
                    m.embedding is not None and any(
                        o.embedding is not None and float(o.embedding @ m.embedding) >= group_threshold
                        for o in g)):
                home = g
                break
        if home is None:
            groups.append([m])
        else:
            home.append(m)
    out = []
    for g in groups:
        votes: dict[str, float] = {}
        for m in g:
            if m.name:
                votes[m.name] = votes.get(m.name, 0.0) + m.score * (0.5 + m.quality)
        name = None
        if votes:
            ranked = sorted(votes.items(), key=lambda kv: -kv[1])
            if len(ranked) == 1 or ranked[0][1] >= 2 * ranked[1][1]:
                name = ranked[0][0]
        score = max((m.score for m in g if m.name == name), default=max(m.score for m in g))
        out.append(FaceGroup(name, round(score, 3), g[0], g))
    out.sort(key=lambda fg: (fg.name is None, -fg.best.quality))
    return out[:limit]


class FaceRecognizer:
    def __init__(self, models_dir: Path, faces_dir: Path, unknown_dir: Path | None = None):
        self.faces_dir = faces_dir
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._detector = cv2.FaceDetectorYN.create(
            str(models_dir / DETECTOR_FILE), "", (320, 320), score_threshold=0.8)
        self._sface = cv2.FaceRecognizerSF.create(str(models_dir / RECOGNIZER_FILE), "")
        self._arcface = None
        if (models_dir / ARCFACE_FILE).exists():
            import onnxruntime as ort
            ort.disable_telemetry_events()
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.log_severity_level = 3
            opts.enable_cpu_mem_arena = False   # zuiniger: geen vastgehouden werkgeheugen
            self._arcface = ort.InferenceSession(str(models_dir / ARCFACE_FILE), opts,
                                                 providers=["CPUExecutionProvider"])
            self._arc_input = self._arcface.get_inputs()[0].name
        self.backend = ARCFACE if self._arcface is not None else SFACE
        self._cache_file = self.faces_dir / f".emb-{self.backend.name}.json"
        self._names: list[str] = []
        self._mat = np.zeros((0, 1), np.float32)
        self.reload()
        self.unknown = UnknownFaces(unknown_dir or faces_dir.parent / "faces_unknown", self)

    @staticmethod
    def available(models_dir: Path) -> bool:
        return (models_dir / DETECTOR_FILE).exists() and (models_dir / RECOGNIZER_FILE).exists()

    # ------------------------------------------------------------------ kenmerken
    def embed(self, aligned: np.ndarray) -> np.ndarray:
        """Genormaliseerde kenmerkvector van een uitgelijnde 112x112-uitsnede (BGR)."""
        if aligned.shape[:2] != (112, 112):
            aligned = cv2.resize(aligned, (112, 112))
        with self._lock:
            if self._arcface is not None:
                x = (cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
                v = self._arcface.run(None, {self._arc_input: x.transpose(2, 0, 1)[None]})[0][0]
            else:
                v = self._sface.feature(aligned)[0]
        v = np.asarray(v, np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-6)

    def _per_person(self, emb: np.ndarray) -> dict[str, float]:
        with self._lock:
            names, mat = self._names, self._mat
        per: dict[str, float] = {}
        if names:
            for n, s in zip(names, (mat @ emb).tolist()):
                per[n] = max(per.get(n, -1.0), s)
        return per

    def identify(self, emb: np.ndarray) -> tuple[str | None, float, float]:
        """(naam of None, beste gelijkenis, voorsprong op de beste andere persoon)."""
        per = self._per_person(emb)
        if not per:
            return None, 0.0, 0.0
        ranked = sorted(per.values(), reverse=True)
        best = ranked[0]
        margin = best - (ranked[1] if len(ranked) > 1 else 0.0)
        ok = best >= self.backend.match and margin >= self.backend.margin
        return (max(per, key=per.get) if ok else None), round(best, 3), round(margin, 3)

    def suggest(self, embs: list[np.ndarray]) -> str | None:
        """Op wie lijkt een groep onbekende gezichten het meest (voorzichtige hint)?"""
        best_name, best = None, 0.0
        for e in embs:
            for name, score in self._per_person(e).items():
                if score > best:
                    best_name, best = name, score
        return best_name if best >= self.backend.match * 0.75 else None

    # ------------------------------------------------------------------ bekende personen
    def _samples(self) -> list[tuple[str, str, bool, Path]]:
        out = []
        for person in sorted(p for p in self.faces_dir.iterdir() if p.is_dir()):
            for img_path in sorted(person.glob("*.jpg")):
                out.append((person.name, img_path.stem, False, img_path))
            for img_path in sorted((person / "auto").glob("*.jpg")):
                out.append((person.name, img_path.stem, True, img_path))
        return out

    def reload(self) -> None:
        try:
            cache = json.loads(self._cache_file.read_text())
        except (FileNotFoundError, ValueError):
            cache = {}
        names, vecs, new_cache = [], [], {}
        for name, _sid, _auto, path in self._samples():
            key = str(path.relative_to(self.faces_dir))
            mtime = path.stat().st_mtime
            hit = cache.get(key)
            if hit and hit[0] == mtime:
                emb = np.asarray(hit[1], np.float32)
            else:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                emb = self.embed(img)
            new_cache[key] = [mtime, [round(float(x), 5) for x in emb]]
            names.append(name)
            vecs.append(emb)
        with self._lock:
            self._names = names
            self._mat = np.stack(vecs) if vecs else np.zeros((0, 1), np.float32)
        try:
            tmp = self._cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(new_cache))
            tmp.replace(self._cache_file)
        except OSError:
            log.warning("Kan %s niet schrijven", self._cache_file)
        log.info("Gezichten (%s): %d foto's van %d personen geladen", self.backend.name,
                 len(names), len(set(names)))

    def people(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name, sid, auto, _ in self._samples():
            out.setdefault(name, [])
            if not auto:
                out[name].append(sid)
        return out

    def auto_samples(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name, sid, auto, _ in self._samples():
            if auto:
                out.setdefault(name, []).append(sid)
        return out

    def sample_path(self, name: str, sample_id: str, auto: bool = False) -> Path | None:
        if not valid_name(name) or not ID_RE.fullmatch(sample_id):
            return None
        path = self.faces_dir / name / ("auto" if auto else "") / f"{sample_id}.jpg"
        return path if path.is_file() else None

    def add(self, name: str, crop_jpg: bytes, reload: bool = True) -> str:
        if not valid_name(name):
            raise ValueError("Naam mag letters, cijfers, spaties en - bevatten (max. 40 tekens)")
        img = cv2.imdecode(np.frombuffer(crop_jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Geen geldige afbeelding")
        sample_id = uuid.uuid4().hex[:12]
        person = self.faces_dir / name
        person.mkdir(exist_ok=True)
        cv2.imwrite(str(person / f"{sample_id}.jpg"), cv2.resize(img, (112, 112)))
        if reload:
            self.reload()
        return sample_id

    def learn_auto(self, group: FaceGroup) -> bool:
        """Een zeker herkend gezicht zelf bijleren, zodat herkenning meegroeit met licht,
        seizoen, bril en kapsel. Alleen goede, duidelijk eenduidige beelden die iets nieuws
        toevoegen; per persoon hooguit AUTO_MAX, de oudste gaan eruit."""
        name = group.name
        cands = [m for m in group.members if m.name == name and m.embedding is not None
                 and m.quality >= AUTO_QUALITY and m.margin >= AUTO_MARGIN
                 and m.score >= self.backend.learn]
        if not name or not cands:
            return False
        m = max(cands, key=lambda c: c.score * c.quality)
        with self._lock:
            own = self._mat[[i for i, n in enumerate(self._names) if n == name]]
        if len(own) and float((own @ m.embedding).max()) >= AUTO_DUPLICATE:
            return False
        auto_dir = self.faces_dir / name / "auto"
        auto_dir.mkdir(parents=True, exist_ok=True)
        (auto_dir / f"{uuid.uuid4().hex[:12]}.jpg").write_bytes(m.crop_jpg)
        for old in sorted(auto_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)[:-AUTO_MAX]:
            old.unlink()
        self.reload()
        log.info("Gezicht van %s bijgeleerd (score %.2f, kwaliteit %.2f)", name, m.score, m.quality)
        return True

    def remove(self, name: str, sample_id: str | None = None, auto: bool = False) -> bool:
        if not valid_name(name):
            return False
        person = self.faces_dir / name
        if sample_id is None:
            if not person.is_dir():
                return False
            shutil.rmtree(person)
        else:
            path = self.sample_path(name, sample_id, auto)
            if path is None:
                return False
            path.unlink()
            if path.parent.name == "auto" and not any(path.parent.iterdir()):
                path.parent.rmdir()
            if not any(person.iterdir()):
                person.rmdir()
        self.reload()
        return True

    # ------------------------------------------------------------------ herkennen
    def analyse(self, bgr: np.ndarray) -> list[FaceMatch]:
        """Alle gezichten in een (uitgesneden) beeld, met de best passende naam."""
        h, w = bgr.shape[:2]
        scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
        small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1 else bgr
        with self._lock:
            self._detector.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self._detector.detect(small)
        if faces is None:
            return []
        out = []
        for face in faces:
            face = face.copy()
            face[:14] /= scale                       # terug naar coördinaten van bgr
            if min(face[2], face[3]) < MIN_FACE_PX:
                continue
            with self._lock:
                aligned = self._sface.alignCrop(bgr, face)
            emb = self.embed(aligned)
            name, score, margin = self.identify(emb)
            ok, buf = cv2.imencode(".jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                continue
            out.append(FaceMatch(name, score, buf.tobytes(), face_quality(face, aligned),
                                 margin, emb))
        return out


class UnknownFaces:
    """Gezichten zonder naam, per persoon gegroepeerd. Per beeld <id>.jpg + <id>.json
    (camera, tijd, uitsnede in het event, kwaliteit en kenmerkvector)."""

    def __init__(self, folder: Path, recognizer: FaceRecognizer):
        self.dir = folder
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rec = recognizer
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        self._emb: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        for js in sorted(self.dir.glob("*.json")):
            try:
                meta = json.loads(js.read_text())
                if meta.get("model") == self.rec.backend.name:
                    emb = np.asarray(meta["emb"], np.float32)
                else:                                   # ander model: opnieuw berekenen
                    img = cv2.imread(str(js.with_suffix(".jpg")))
                    if img is None:
                        raise ValueError("geen beeld")
                    emb = self.rec.embed(img)
                    meta.update(model=self.rec.backend.name, emb=[round(float(x), 5) for x in emb])
                    js.write_text(json.dumps(meta))
            except Exception:  # noqa: BLE001
                log.warning("Onbekend gezicht %s overgeslagen", js.name)
                continue
            self._items[js.stem] = meta
            self._emb[js.stem] = emb
        self.prune()

    def _delete(self, sid: str) -> None:
        self._items.pop(sid, None)
        self._emb.pop(sid, None)
        for ext in (".json", ".jpg"):
            (self.dir / f"{sid}{ext}").unlink(missing_ok=True)

    def path(self, sid: str) -> Path | None:
        if not ID_RE.fullmatch(sid):
            return None
        p = self.dir / f"{sid}.jpg"
        return p if p.is_file() else None

    def add(self, m: FaceMatch, camera: str, when: datetime, crop_rel: str) -> str | None:
        """Bewaar een onbekend gezicht; geeft het id terug (None als het te slecht is)."""
        if m.embedding is None or m.quality < UNKNOWN_QUALITY:
            return None
        sid = uuid.uuid4().hex[:12]
        meta = {"camera": camera, "time": when.isoformat(timespec="seconds"), "crop": crop_rel,
                "quality": m.quality, "model": self.rec.backend.name,
                "emb": [round(float(x), 5) for x in m.embedding]}
        with self._lock:
            (self.dir / f"{sid}.jpg").write_bytes(m.crop_jpg)
            (self.dir / f"{sid}.json").write_text(json.dumps(meta))
            self._items[sid] = meta
            self._emb[sid] = m.embedding
        self.prune()
        return sid

    def group_of(self, sid: str) -> dict | None:
        """De groep (zelfde persoon) waar dit onbekende gezicht in zit."""
        return next((g for g in self.groups() if sid in g["ids"]), None)

    def prune(self) -> None:
        cutoff = (datetime.now().astimezone() - timedelta(days=UNKNOWN_KEEP_DAYS)).isoformat()
        with self._lock:
            for sid in sorted(self._items, key=lambda s: self._items[s]["time"]):
                if self._items[sid]["time"] < cutoff or len(self._items) > UNKNOWN_MAX:
                    self._delete(sid)

    def drop_crop(self, crop_rel: str) -> None:
        """Een event-gezicht kreeg via de speler een naam: niet meer als onbekend tonen."""
        with self._lock:
            for sid in [s for s, m in self._items.items() if m["crop"] == crop_rel]:
                self._delete(sid)

    def groups(self) -> list[dict]:
        """Groepen van (waarschijnlijk) dezelfde persoon, vaakst geziene eerst."""
        with self._lock:
            ids = list(self._items)
            if not ids:
                return []
            mat = np.stack([self._emb[s] for s in ids])
            items = dict(self._items)
        parent = list(range(len(ids)))

        def root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, j in np.argwhere(np.triu(mat @ mat.T, 1) >= self.rec.backend.group):
            parent[root(int(i))] = root(int(j))
        clusters: dict[int, list[int]] = {}
        for i in range(len(ids)):
            clusters.setdefault(root(i), []).append(i)
        out = []
        for members in clusters.values():
            members.sort(key=lambda i: -items[ids[i]]["quality"])
            times = sorted(items[ids[i]]["time"] for i in members)
            out.append({
                "ids": [ids[i] for i in members],
                "samples": [ids[i] for i in members[:6]],
                "count": len(members),
                "first": times[0], "last": times[-1],
                "cameras": sorted({items[ids[i]]["camera"] for i in members}),
                "suggest": self.rec.suggest([mat[i] for i in members[:3]]),
            })
        out.sort(key=lambda g: (g["count"], g["last"]), reverse=True)
        return out

    def assign(self, ids: list[str], name: str) -> list[tuple[str, str]]:
        """Geef een groep een naam: de beste beelden worden geleerd, de rest vervalt.
        Daarna worden de overige onbekenden opnieuw vergeleken; wie nu herkend wordt
        krijgt ook die naam. Geeft (uitsnede-in-event, naam) terug om events bij te werken."""
        if not valid_name(name):
            raise ValueError("Naam mag letters, cijfers, spaties en - bevatten (max. 40 tekens)")
        with self._lock:
            chosen = sorted((s for s in ids if s in self._items),
                            key=lambda s: -self._items[s]["quality"])
        if not chosen:
            raise KeyError("Niet gevonden")
        for sid in chosen[:ASSIGN_MAX]:
            self.rec.add(name, (self.dir / f"{sid}.jpg").read_bytes(), reload=False)
        self.rec.reload()
        updates = []
        with self._lock:
            for sid in chosen:
                updates.append((self._items[sid]["crop"], name))
                self._delete(sid)
            for sid in list(self._items):
                found, _, _ = self.rec.identify(self._emb[sid])
                if found:
                    updates.append((self._items[sid]["crop"], found))
                    self._delete(sid)
        return updates

    def dismiss(self, ids: list[str]) -> int:
        with self._lock:
            gone = [s for s in ids if s in self._items]
            for sid in gone:
                self._delete(sid)
        return len(gone)

    def count(self) -> int:
        with self._lock:
            return len(self._items)
