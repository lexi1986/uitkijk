# Uitkijk

Your own lightweight camera system for home: records what happens, recognises people, vehicles and
animals, sends notifications through Home Assistant and has its own web page for phone and laptop.
Everything runs locally; no video goes to the cloud.

*Uitkijk* is Dutch for "lookout".

- **Recording with the seconds before**: a buffer of ~10 s, so you also see who was arriving.
- **Detection** of people, vehicles and (optionally) animals, with filters against plants, shadows and
  parked cars. Doubtful detections are double-checked on the sharp image.
- **Faces** (optional): teach it who your housemates are, and answer "Who is this?" from a notification.
- **Dutch licence plates** (optional); known cars get a name.
- **Web page** with timeline, search ("package", a name, a plate), daily summary, live view, masks and
  zones, and "Nothing to worry about" to ignore a false alarm from then on.
- **Home Assistant** via MQTT: sensors per camera, snapshots, a daily summary and ready-made
  blueprints for notifications.
- **Log in with a passkey** (Face ID / Touch ID) when you make the page reachable over HTTPS with
  your own domain.

![Licence: AGPL-3.0](https://img.shields.io/badge/licence-AGPL--3.0-blue)

## What you need

- A computer that is always on with **Docker** (a mini PC, a NAS with Docker, or a server). Two CPU
  cores and 2 GB of memory are enough for 3 cameras; with face recognition preferably 4 GB. An Intel
  CPU with an iGPU helps (hardware decoding) but is not required.
- **Cameras with RTSP**: e.g. Reolink, Foscam, Hikvision/Annke, Dahua/Amcrest, TP-Link Tapo/VIGI,
  UniFi Protect. Preferably with a second, low-resolution stream ("substream").
- Optional: **Home Assistant** with the Mosquitto broker (MQTT) for notifications.

## Install

```bash
git clone https://github.com/lexi1986/uitkijk.git
cd uitkijk
docker compose up -d
docker compose logs camsvc | grep "setup code"
```

Then open `http://<ip-of-that-computer>:8080`. The **setup wizard** asks for the setup code from the
log and then helps you with:

1. **Cameras**: pick the brand, enter IP address and login and press **Test**: you see an image right
   away. Another brand? Enter the RTSP address yourself.
2. **Home Assistant** (optional): address and user of the MQTT server, with a test button.
3. **Options**: face recognition, licence plates, live view.
4. **Storage**: how long recordings are kept and how much space they may use.
5. **Address**: how you open the page, and optionally your own domain for HTTPS and passkeys.

After **Save and start** the service restarts and shows your cameras. The first start takes a bit
longer when you enabled face recognition (~500 MB of models to download).

Everything then lives in the folders `data/` (settings, faces, plates) and `clips/` (recordings).
Back up `data/` now and then. Per-camera settings, masks and zones are changed under **Manage** on
the web page; all other options are listed in [`config.example.yaml`](config.example.yaml).

### Updating

```bash
git pull && docker compose up -d --build
```

### Without Docker (Proxmox container)

There is also a script that creates a Debian container on Proxmox and installs everything, see
[`docs/GUIDE.md`](docs/GUIDE.md#installing-on-proxmox-without-docker).

## Connecting Home Assistant

1. Make sure the **MQTT integration** is set up in Home Assistant (with the Mosquitto app) and enter
   those details in the wizard. The cameras appear as devices by themselves, with sensors such as
   `binary_sensor.cam_front_door_person` and a snapshot camera.
2. Import the **blueprints** for notifications (Settings → Automations → Blueprints → Import
   blueprint) with the link to the file in this repository:

| Blueprint | What |
|---|---|
| [person_alert](homeassistant/blueprints/automation/uitkijk/person_alert.yaml) | Notification with photo when there is a person, with "Watch live" and "Recording" |
| [doorbell](homeassistant/blueprints/automation/uitkijk/doorbell.yaml) | Recording and notification on a doorbell ring, with the name when someone is recognised |
| [loitering](homeassistant/blueprints/automation/uitkijk/loitering.yaml) | Urgent notification when an unknown person stays for a long time |
| [who_is_this](homeassistant/blueprints/automation/uitkijk/who_is_this.yaml) | Name an unknown face from the notification |
| [unknown_vehicle](homeassistant/blueprints/automation/uitkijk/unknown_vehicle.yaml) | Unknown plate at night, or during the day after 15 minutes |
| [daily_summary](homeassistant/blueprints/automation/uitkijk/daily_summary.yaml) | Summary of the day at 22:00 |
| [presence](homeassistant/blueprints/automation/uitkijk/presence.yaml) | No alarm when a resident just arrived home |
| [server_health](homeassistant/blueprints/automation/uitkijk/server_health.yaml) | Notification on a crash, full disk or offline camera |

For "Notification services" enter your phone's notify action, e.g.
`notify.mobile_app_sams_iphone` (found under Developer tools → Actions).

## Access from outside and login

Without your own domain the web page is open to everyone on your home network. To use it away from
home, put a reverse proxy with HTTPS in front (Caddy, Nginx Proxy Manager, Traefik) or use a VPN
(Tailscale, WireGuard). Enter your domain in the wizard; then you log in with a **passkey**. The
one-time pairing code for the first passkey is in the log. Never expose port 8080 to the internet
without such protection.

## Privacy

- Images, faces and plates stay on your own computer.
- If you film the public road or the neighbours, privacy rules apply (in the EU the GDPR): point
  cameras at your own property as much as possible, mask the rest, and do not keep recordings longer
  than needed. Recognising faces of anyone other than your housemates is especially sensitive.
- Optionally a **local** AI model (Ollama) can describe recordings; see the guide.

## Licence

AGPL-3.0, see [`LICENSE`](LICENSE). You may use, change and share it; whoever distributes a modified
version or offers it as a service must make the source code available too. The models have their own
licences, see [`MODELS.md`](MODELS.md). The InsightFace face model is for non-commercial use only.

More details (all options, API, hardware acceleration, Coral, AI descriptions):
[`docs/GUIDE.md`](docs/GUIDE.md).
