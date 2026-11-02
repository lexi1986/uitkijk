# Uitkijk – technical guide

For installing with Docker and the setup wizard, see the [README](../README.md). This guide covers
how it works, every option, the API, the Proxmox install, hardware acceleration and the optional
extras.

## How it works

```
camera ──main stream──► ring buffer in RAM (encoded, ~10 s) ──► MP4 (stream copy)
   │                                                               ▲
   └───sub stream──► decode 5 fps ──► motion? ──► YOLO11n ─────────┘ event starts/extends
                                                     │
                                                     └──► MQTT ──► Home Assistant
```

- The **main stream** (high quality) is never decoded. The encoded packets (video and audio) live in
  a ring buffer in memory. On an event they are written 1:1 to an MP4 (no transcoding, no temporary
  files, no disk I/O until there is a clip). Clips are fragmented MP4: after a crash or power cut the
  clip is still playable up to just before that moment and is recovered at the next start.
- The **sub stream** (low quality) is decoded. Every 1/5 second a small grey frame goes to motion
  detection. Only on motion does the object model run, at most 2× per second (5× with a Coral).
- **Event**: starts on a confirmed person, vehicle or animal (or on motion when object detection is
  off). The clip contains the pre-buffer and keeps recording until 30 s after the **last** detection
  (max. 5 min).
- **Storage**: `/clips/<camera>/<date>/<time>_<label>.mp4` + `.jpg` (snapshot with box) + `.json`.
  Kept 7 days by default (people and doorbell rings 30) and at most the configured GB; the oldest go
  first.
- **Doubtful detections** (score below `objects.verify_below`) are first verified on a sharp, zoomed
  crop from the main stream. A drainpipe or garden bag scores 0 there, a real person 0.8+.
- **Stationary objects**: vehicles and animals that stay in the same spot stop triggering events,
  also when the sun or a shadow makes the model see only part of a parked car.

## What appears in Home Assistant

One device per camera (e.g. "Front door") with:

| Entity | What |
|---|---|
| `binary_sensor.cam_<camera>_motion` | motion |
| `binary_sensor.cam_<camera>_person` / `_vehicle` / `_animal` | confirmed objects |
| `binary_sensor.cam_<camera>_loitering` | unrecognised person in view for a while |
| `binary_sensor.cam_<camera>_online` | both streams connected |
| `camera.cam_<camera>_snapshot` | last detection with box |
| `event.cam_<camera>_event` | one event per saved clip, plus `unknown_vehicle` and `unknown_face` alerts |
| `sensor.cam_<camera>_last_event` | start of the last event; attribute `events` with the last 20 |
| `sensor.cam_<camera>_face` / `_plate` / `_description` | recognised name, plate, AI description (when enabled) |
| `switch.cam_<camera>_detect` / `_record` | detection / recording on or off |

And one device "Uitkijk" for the server: `sensor.camera_daily_summary` (22:00),
`sensor.camera_today` (running total every 5 min), health sensors
`sensor.camera_service_cpu` / `_memory` / `_memory_mb` / `_disk` / `_disk_free_gb` / `_ai_server` /
`_cameras_online` / `_started`, and `event.camera_service_alert` (`restart_after_crash`).

MQTT topics the service listens to (prefix `camsvc`):

| Topic | Payload |
|---|---|
| `camsvc/<camera>/trigger` | `doorbell` (or `motion`): start a recording now |
| `camsvc/<camera>/<detect\|record>/set` | `ON` / `OFF` |
| `camsvc/server/face` | `{"sample": "<id>", "name": "Sam"}` or `{"sample": "<id>", "ignore": true}` |
| `camsvc/server/presence` | `{"name": "Alex", "home": true}` |
| `camsvc/server/external_ai` | `{"camera": "<camera>", "notify": true, "title": "…", "description": "…"}` |

## REST API

Without passkey login the API is open to your network; with login, only to logged-in browsers and
to the server itself (127.0.0.1).

| Path | What |
|---|---|
| `GET /api/status` | status per camera, storage, detector, AI, health |
| `GET /api/events?camera=&label=&person=&plate=&q=&day=&important=` | stored events, newest first |
| `GET /api/summary?day=YYYY-MM-DD` | daily summary |
| `POST /api/events/false-alarm` | `{"clip": "...", "false_alarm": true}` |
| `GET /api/ai-comparison?days=7` | external AI next to local AI |
| `GET /api/cameras/<camera>/latest.jpg` | last snapshot |
| `GET /api/cameras/<camera>/mask.jpg` | current frame with mask, zones and percentage grid |
| `POST /api/cameras/<camera>/trigger?label=doorbell` | start a recording |
| `GET /clips/<path>` | clip, snapshot or JSON |
| `GET /docs` | API documentation |

Links that open the web page in the right place (handy in notifications): `/?live=<camera>`,
`/?camera=<camera>&open=latest` (newest recording of that camera) and `/?clip=<path>.mp4`. When not
logged in, you get there after logging in with your passkey.

---

## Installing on Proxmox (without Docker)

### 1. Create the container (on the Proxmox host)

Copy this folder to the Proxmox host and run one script that creates the LXC and installs
everything:

```bash
rsync -a --exclude .venv --exclude .git ./ root@<proxmox-ip>:/root/camera-service/
ssh root@<proxmox-ip> bash /root/camera-service/deploy/lxc/proxmox-create.sh
```

DHCP is the default; a fixed IP is possible with `IP=192.168.1.20/24 GW=192.168.1.1` before `bash`.

The script creates:
- **Debian 12**, unprivileged, 2 cores, 4 GB RAM, root disk 8 GB.
- **A 32 GB clip disk on `/clips` with `backup=0`**, so it is not part of your Proxmox backups.
- **iGPU passthrough**: `/dev/dri/renderD128` with the right `render` group (Proxmox 8.2+).
- Installs the service (`deploy/lxc/install.sh`) and builds the **YOLO11n model** (~1 GB temporary
  download, cleaned up afterwards).

Other values via variables: `CTID`, `CT_NAME`, `STORAGE`, `CLIPS_STORAGE`, `CLIPS_GB`, `VLAN`,
`CORES`, `MEMORY`, `GPU=0` (no iGPU), `EXPORT_MODEL=0` (build the model later).

**Updating** after a code change: rsync again and then on the host
`CTID=<number> bash /root/camera-service/deploy/lxc/push-update.sh`. Config, clips and model stay.

### 2. Installing manually (existing container)

```bash
bash /path/to/camera-service/deploy/lxc/install.sh   # in the container, as root
bash /opt/camera-service/deploy/export-model.sh       # when the model is still missing
```

Add the clip disk and iGPU yourself: `pct set <ID> -mp0 local-lvm:32,mp=/clips,backup=0` and
`pct set <ID> -dev0 /dev/dri/renderD128,gid=<gid of render in the container>`.

### 3. Configure

Without a config the service starts the setup wizard on port 8080 (the setup code is in
`journalctl -u camera-service`). Or edit `/etc/camera-service/config.yaml` by hand; all options are
in [`config.example.yaml`](../config.example.yaml).

- Test RTSP URLs in VLC first.
  - **Reolink**: `rtsp://user:password@<ip>:554/h264Preview_01_main` and `..._sub` (enable
    RTSP/ONVIF in the Reolink app under Network → Advanced → Port settings).
  - **Foscam**: `rtsp://user:password@<ip>:88/videoMain` and `videoSub`. The G.711 audio of the
    doorbell is converted to AAC in clips automatically.
- Tip for Reolink: set the **frame rate of the sub stream to 10 fps**; only 5 fps are looked at.

### 4. Start and check

```bash
systemctl restart camera-service
journalctl -u camera-service -f
curl -s http://localhost:8080/api/status | python3 -m json.tool
```

The log should show `main stream connected`, `sub stream connected` and
`Object model yolo11n.onnx loaded via ONNX Runtime on CPU`.

---

## Web page

- **Camera tiles** with **▶ Live** (via go2rtc, proxied through the service's own address because
  browsers block pages from talking to other LAN addresses).
- **Timeline** with search (names, plates, AI description), filters, day picker and daily summary.
  **Important only** (on by default) hides "probably nothing" and your own car coming or going.
- **Player** with faces, plates, the AI description and **Nothing to worry about** (see below).
- **Manage**: storage, server health, settings and mask/zones per camera, AI comparison, passkeys.
  Settings apply immediately and are stored in `settings.json`, masks in `masks.json`, zones in
  `zones.json` (all in the state folder; they take precedence over the config).

### Masks and zones

Manage → **Mask and zones**: click points to draw a polygon, click the first point to close it,
**Save**. Red mask areas are ignored (no motion, objects whose bottom edge is there do not count).
Violet zones: when there are zones, only objects whose bottom edge is inside a zone count, and events
get the zone name. By hand in the config: `mask: - [[0, 0], [100, 0], [100, 20], [0, 20]]` (percent).

## Doorbell, loitering and stationary objects

- **Doorbell button**: publish `doorbell` to `camsvc/<camera>/trigger`, or
  `POST /api/cameras/<camera>/trigger`. A recording with label *doorbell* starts immediately and the
  service searches the whole sharp image for a face (3× one second apart).
- **Loitering**: an unrecognised person in view longer than `objects.loiter_seconds` (default 30 s)
  turns on `binary_sensor.cam_<camera>_loitering`. Works best with `objects.always: true`.
- **Stationary vehicles and animals** are saved every 30 s in `stationary_<camera>.json`, so a
  restart does not create new events. While idle, parked cars are re-checked every 5 minutes.

## Face recognition

Recognises people you teach it, fully local. YuNet (OpenCV Zoo) finds and aligns faces; **ArcFace**
(InsightFace `w600k_r50`) compares them with the taught photos, falling back to SFace without that
model. Note: the InsightFace models are for non-commercial use only (see [MODELS.md](../MODELS.md)).

- Enable per camera: `faces: {enabled: true}` (or in the wizard). Works best at the front door.
- **Best image per event**: for a confirmed person the service takes a sharp main-stream image every
  second, up to 8 attempts or until someone is recognised with confidence. Each face gets a quality
  score (size, sharpness, frontal); votes are combined per person.
- **Learning automatically** (`faces.auto_learn`): confident, unambiguous faces that add something
  new are stored in `faces/<name>/auto/` (max. 20 per person).
- **Unknown faces** (`faces.collect_unknown`): faces without a name are kept 30 days in
  `faces_unknown/` and grouped per person. Under **Faces** pick a name for a group; the best photos
  are learned and all events get the name. "Ignore" discards the group.
- **Home Assistant**: `sensor.cam_<camera>_face` = name, `unknown` or `none`.
- Faces never leave the server. Only teach housemates and family who know about it; turn off
  `faces.collect_unknown` if you do not want to keep faces of passers-by.

### Clothing, presence and "Who is this?"

- **Clothing and build** (YouTu-ReID, Apache-2.0): when a face is recognised with one person in
  view, the service remembers that day (12 hours) what that person looks like. A person without a
  recognised face (e.g. seen from behind) who clearly matches gets `likely: [name]` in the event.
- **Presence from HA**: blueprint *presence* sends `{"name": "Alex", "home": true}` to
  `camsvc/server/presence`. For 5 minutes after arriving, a person without a recognised face is
  "probably Alex" and does not trigger loitering. The person entity's name must match the face name.
- **"Who is this?"**: for a new unknown face (1st and 3rd time) the service sends
  `event_type: unknown_face`; the blueprint *who_is_this* shows buttons with known names, "New
  person…" and "Ignore" and sends the answer to `camsvc/server/face`.

## Zones, unknown vehicles and daily summary

- **Unknown vehicle**: a plate that is not known (not even with one misread character) gives
  `event_type: unknown_vehicle`: at night (22–07) immediately, during the day only when a vehicle is
  still at that spot after 15 minutes.
- **Daily summary**: `GET /api/summary?day=YYYY-MM-DD` and on the web page via Today/Yesterday/date.
  At `summary_time` (default 22:00) it is published as `sensor.camera_daily_summary`.

## False alarms and health

- **Nothing to worry about**: in the player, mark an event as a false alarm. For a person, vehicle
  or animal the camera remembers that spot (`false_spots_<camera>.json`): the same kind of object in
  almost the same spot (IoU ≥ 0.6) no longer counts. "Important after all" undoes it.
- **AI filter** (with AI descriptions): the local model reports `person_visible`, `vehicle_visible`
  and `animal_visible`. When it sees none of what triggered the event, the event gets
  `false_alarm: "ai"`: no loitering alert, greyed out in the timeline, left out of the daily summary.
  A doorbell ring is never a false alarm; for a weak infrared image the AI gives no verdict.
- **Health**: every minute CPU, memory (relative to the service's limit), disk, AI server and cameras
  online go to HA. When the service did not stop cleanly (crash, OOM kill), `event.camera_service_alert`
  fires `restart_after_crash` after the restart. Crash details end up in `crash.log` in the state folder.

## Log in with passkeys

Put a reverse proxy with HTTPS and your own domain in front (Caddy, Nginx Proxy Manager, Traefik)
and enable `auth` (the wizard does this when you fill in a domain). Then only browsers logged in with
a passkey get in; requests from the server itself (127.0.0.1, without `X-Forwarded-For`) are allowed.

- **First passkey**: the page asks for the one-time pairing code (`pairing_code` in the state folder,
  also in the log). With iCloud Keychain the passkey is then on your other Apple devices too.
- **More devices**: Manage → **Passkeys** (add, remove, log out everywhere).
- **Lost all passkeys**: remove them from `auth.json` (or the whole file) and restart; a new pairing
  code appears.

## Licence plates (NL)

Reads Dutch licence plates of vehicles that arrive or leave, fully local (fast-alpr: YOLOv9 detector
+ OCR via ONNX Runtime, ~15 MB of models downloaded on first use).

- Enable per camera: Manage → Settings → "Licence plate recognition (NL)", or `plates: {enabled: true}`.
  Only useful for a camera that sees cars close by from the front or back.
- Up to 4 sharp main-stream images per moving vehicle; the most frequent reading wins. Only text that
  fits a Dutch sidecode counts (with at most one correction of a typical misread like O/0 or B/8).
- **Known vehicles**: name a plate from an event; overview under **Plates**. Stored in `plates.json`.
- At night it usually fails: the reflective plate becomes a white blob in infrared.

## AI descriptions (local vision model)

Optional: a short description per event (person, vehicle, doorbell) by a local vision model, e.g.
`qwen3-vl` in Ollama on a separate AI server. Asynchronous: recordings, detection and notifications
never wait for it.

- **Connection**: Ollama keeps listening on `127.0.0.1:11434` on the AI server only. The service
  reaches it through an SSH tunnel (`deploy/camsvc-ai-tunnel.service`, user `aitunnel`) that listens
  on `127.0.0.1:11435` on the camera server. On the AI server the tunnel key is in `authorized_keys`
  with `restrict,port-forwarding,permitopen="127.0.0.1:11434",command="/bin/false"`: no shell, only
  forwarding to Ollama.
- **Image**: one sharp main-stream image without detection boxes, masked areas blacked out, a wide
  crop around the object, max. 1280 px and 2 MiB, re-encoded (no metadata). For black-and-white
  infrared images the model is told not to mention colours.
- **Limits**: one worker, queue max. 10 (one waiting image per camera), 1 per camera per 30 s, max. 6
  per minute, deadline 180 s, at most one retry on transient errors, a pause after 5 errors in a row.
- **Output**: schema-checked JSON (description, objects, attributes, uncertainties, visibility flags),
  stored with the event and published as `sensor.cam_<camera>_description`. Logs only contain status
  and duration, never descriptions.

**Enable**: `ai: {enabled: true}` in the config and restart. **Undo**: `enabled: false`, restart,
`systemctl disable --now camsvc-ai-tunnel`, and remove the tunnel key from `authorized_keys` on the AI
server.

## Hardware acceleration

- **Object model**: by default **ONNX Runtime on the CPU** (`yolo11n.onnx`); on an older Core i5 that
  is a few tens of milliseconds per frame, and only on motion. OpenVINO (`yolo11n.xml`,
  `FORMAT=openvino bash deploy/export-model.sh`) can use an Intel iGPU; the service tests it in a
  separate process first and only uses it when that test passes.
- **Decoding the sub stream**: the pip build of PyAV has no VAAPI, so the sub stream is decoded on the
  CPU (a few percent of one core at 640×360). `hwaccel: auto` uses hardware when the PyAV build
  supports it and silently falls back otherwise.

### Google Coral (USB)

With a Coral USB the object model runs on the Edge TPU, so detection can look more often:
`max_fps: auto` becomes 5 instead of 2, and `always: true` (per camera) keeps looking for people even
without motion. Without a working Coral the service falls back to `model.path` on the CPU.

**New container**: `CORAL=1 bash /root/camera-service/deploy/lxc/proxmox-create.sh`. **Existing
container** (on the Proxmox host, `<ID>` = container number):

```bash
cat > /etc/udev/rules.d/99-coral-edgetpu.rules <<'UDEV'
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1a6e", ATTRS{idProduct}=="089a", MODE="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="9302", MODE="0666"
UDEV
udevadm control --reload-rules && udevadm trigger
cat >> /etc/pve/lxc/<ID>.conf <<'LXC'
lxc.cgroup2.devices.allow: c 189:* rwm
lxc.mount.entry: /dev/bus/usb dev/bus/usb none bind,optional,create=dir
LXC
pct reboot <ID>
```

Then in the container: `CORAL=1 EXPORT_MODEL=1 bash /opt/camera-service/deploy/lxc/install.sh`. The
log should show `Object model yolo11n_edgetpu.tflite loaded via TFLite on Edge TPU`.

## Upgrading from the Dutch version

Installations from before the English version are migrated automatically on first start: labels and
fields in clips, events, faces, false-alarm spots, settings and passkeys get their English names, and
`config.yaml` is updated (a copy is kept as `config.yaml.before-english`). In Home Assistant the
entities get new English IDs (e.g. `binary_sensor.cam_<camera>_person`); update automations and
dashboards that use the old ones, and remove the old, unavailable entities.

## Development and tests

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m camsvc --config config.yaml
```

## Licences

AGPL-3.0, see [`LICENSE`](../LICENSE). Models and their licences are listed in
[`MODELS.md`](../MODELS.md).
