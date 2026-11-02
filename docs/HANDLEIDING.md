# Uitkijk – technische handleiding

Technische handleiding. Voor installeren met Docker en de wizard: zie de [README](../README.md).

Lichtgewicht videobuffer- en detectieservice, als Docker-container of in een Proxmox-LXC. Werkt met elke RTSP/ONVIF-camera en praat via MQTT met Home Assistant.

## Hoe het werkt

```
camera ──main-stream──► ringbuffer in RAM (gecodeerd, ~10 s) ──► MP4 (stream copy)
   │                                                               ▲
   └───sub-stream──► decoderen 5 fps ──► beweging? ──► YOLO11n ────┘ event start/verlengt
                                                         │
                                                         └──► MQTT ──► Home Assistant
```

- **Main-stream** (hoge kwaliteit) wordt nooit gedecodeerd. De gecodeerde pakketten (video en audio)
  staan in een ringbuffer in het geheugen. Bij een event worden ze 1-op-1 naar een MP4 geschreven
  (geen hercodering, geen tijdelijke bestanden, geen disk-I/O tot er een clip is).
- **Sub-stream** (laag) wordt gedecodeerd. Elke 1/5 seconde gaat een verkleind grijs frame naar de
  bewegingsdetectie. Pas bij beweging draait het objectmodel, maximaal 2× per seconde.
- **Met een Google Coral** draait het model op de Coral in plaats van de CPU. Daardoor kan het
  vaker kijken (5× per seconde, dus snellere bevestiging) en bij camera's met `always: true`
  ook zonder beweging naar personen blijven kijken. Zie [Google Coral](#google-coral-usb).
- **Event**: start bij een bevestigde persoon, voertuig of dier (of bij beweging als objectdetectie
  uit staat). De clip bevat de pre-buffer en loopt door tot 30 s na de **laatste** detectie
  (max. 5 min).
- **Opslag**: `/clips/<camera>/<datum>/<tijd>_<label>.mp4` + `.jpg` (snapshot met kader) +
  `.json`. Standaard 7 dagen bewaard en maximaal 25 GB; daarboven gaan de oudste weg.

## Wat er in Home Assistant verschijnt

Per camera een apparaat (bijv. "Deurbel") met:

| Entiteit | Wat |
|---|---|
| `binary_sensor.cam_<camera>_beweging` | beweging |
| `binary_sensor.cam_<camera>_persoon` / `_voertuig` / `_dier` | bevestigde objecten |
| `binary_sensor.cam_<camera>_online` | beide streams verbonden |
| `camera.cam_<camera>_snapshot` | laatste detectie met kader |
| `event.cam_<camera>_event` | één event per opgeslagen clip; volledige lijst in logboek en geschiedenis |
| `sensor.cam_<camera>_laatste_event` | starttijd laatste event; attribuut `events` met de laatste 20 (incl. clip- en snapshot-URL) |
| `switch.cam_<camera>_detect` | detectie aan/uit |
| `switch.cam_<camera>_record` | opname aan/uit |

Live beeld blijft rechtstreeks uit de camera komen (bij Reolink via de Reolink-integratie).
Voorbeelden staan in [`deploy/homeassistant/`](deploy/homeassistant).

## REST-API (alleen LAN/VPN, geen login)

| Pad | Wat |
|---|---|
| `GET /api/status` | status per camera: verbinding, buffer, fps, detector, schijfgebruik |
| `GET /api/cameras/<camera>/events` | laatste 20 events |
| `GET /api/cameras/<camera>/latest.jpg` | laatste snapshot |
| `GET /api/cameras/<camera>/mask.jpg` | huidig beeld met masker en procentenraster |
| `GET /clips/<pad>` | clip, snapshot of JSON |
| `GET /docs` | API-documentatie |

---

## Stappenplan

### 1. Container maken (op de Proxmox-host)

Kopieer deze map vanaf je Mac naar de Proxmox-host (zonder de lokale `.venv`) en draai daar
één script dat de LXC maakt en alles installeert:

```bash
rsync -a --exclude .venv --exclude .git ~/Documents/camera-service root@<proxmox-ip>:/root/
ssh root@<proxmox-ip> bash /root/camera-service/deploy/lxc/proxmox-create.sh
```

DHCP is standaard; een vast IP kan met `IP=192.168.1.20/24 GW=192.168.1.1` vóór `bash`.

Dit doet het script:
- **Debian 12**, unprivileged, 2 cores, 1 GB RAM, rootdisk 8 GB (ruim genoeg voor 3 camera's).
  Debian 12 heeft de Intel OpenCL-driver die ook oudere iGPU's (6e generatie) nog ondersteunt.
- **Clipschijf van 32 GB op `/clips`, met `backup=0`**, dus niet in je Proxmox-backup.
- **iGPU doorgeven**: `/dev/dri/renderD128` met het juiste groepsnummer van `render` (Proxmox 8.2+).
- Installeert camera-service (`deploy/lxc/install.sh`) en maakt het **YOLO11n-model** (± 1 GB
  tijdelijke download, wordt daarna opgeruimd).

Andere waarden kunnen via variabelen: `CTID`, `CT_NAME`, `STORAGE`, `CLIPS_STORAGE`, `CLIPS_GB`,
`VLAN`, `CORES`, `MEMORY`, `GPU=0` (geen iGPU), `EXPORT_MODEL=0` (model later maken).

**Bijwerken** na een wijziging in de code: opnieuw rsyncen en dan op de host
`CTID=<nummer> bash /root/camera-service/deploy/lxc/push-update.sh`. Config, clips en model blijven staan.

**Liever via je eigen GitHub-repo?** Zet de map in een (private) repo en gebruik
`REPO=<jij>/camera-service GITHUB_TOKEN=github_pat_xxx bash proxmox-create.sh`. Dan komt er ook
een timer die elke 5 minuten nieuwe commits ophaalt en alleen dan de service herstart.

### 2. Handmatig installeren (bestaande container)

```bash
bash /pad/naar/camera-service/deploy/lxc/install.sh   # in de container, als root
bash /opt/camera-service/deploy/export-model.sh       # als het model nog ontbreekt
```

Voor een bestaande container moet je de clipschijf en de iGPU zelf toevoegen:
`pct set <ID> -mp0 local-lvm:32,mp=/clips,backup=0` en
`pct set <ID> -dev0 /dev/dri/renderD128,gid=<gid van render in de container>`.

### 3. MQTT-gebruiker

Maak in HA een gebruiker voor de service, bijvoorbeeld `camera-service` (Instellingen → Personen → Gebruikers,
of in de Mosquitto-add-on onder *logins*). Vul die in bij `mqtt:` in de config.

### 4. Configureren

```bash
nano /etc/camera-service/config.yaml
```

- Vul per camera de RTSP-URL's in, of gebruik het `onvif:`-blok. Test een URL eerst in VLC.
  - **Reolink**: `rtsp://gebruiker:wachtwoord@<ip>:554/h264Preview_01_main` en `..._sub`
    (RTSP/ONVIF aanzetten in de Reolink-app onder Netwerk → Geavanceerd → Poortinstellingen).
  - **Foscam VD1**: `rtsp://gebruiker:wachtwoord@<ip>:88/videoMain` en `videoSub`
    (poort kan ook 554 zijn; ONVIF meestal poort 888). De G.711-audio van de deurbel wordt in de
    clips automatisch naar AAC omgezet.
- Zet `api.public_url` op het IP-adres van de LXC.
- Tip voor de Reolink-app: zet de **framerate van de sub-stream ("Vloeiend") op 10 fps**. Er wordt
  toch maar 5 fps bekeken, en minder frames is minder decoderen.

### 5. Starten en controleren

```bash
systemctl restart camera-service
journalctl -u camera-service -f
curl -s http://localhost:8080/api/status | python3 -m json.tool
```

In de log moet staan: `main-stream verbonden`, `sub-stream verbonden` en
`Objectmodel yolo11n.onnx geladen via ONNX Runtime op CPU`.

### 6. Masker instellen

**Makkelijkst via de webpagina:** klik rechtsboven op een camerategel op het maskerknopje, klik
punten om een vlak te tekenen, klik op het eerste punt om het te sluiten en kies **Opslaan**. Het
masker werkt meteen (geen herstart) en wordt bewaard in `/var/lib/camera-service/masks.json`; dat
gaat voor op `mask:` in de config. Hieronder de handmatige manier.

Open per camera `http://<lxc-ip>:8080/api/cameras/<naam>/mask.jpg`. Het raster toont procenten. Zet
gebieden die genegeerd moeten worden (straat, bomen) in de config als polygonen:

```yaml
    mask:
      - [[0, 0], [100, 0], [100, 20], [0, 20]]
```

Herstart de service en controleer het plaatje opnieuw.

### 7. Home Assistant

1. De entiteiten verschijnen vanzelf onder MQTT → per camera een apparaat (Deurbel, Oprit, Tuin).
2. Voeg de automation en de kaart uit `deploy/homeassistant/` toe.
3. Zet bij de Reolink-integratie de detectie-entiteiten uit (persoon, voertuig, huisdier, beweging).
   De integratie zelf blijft voor live beeld en instellingen. De belknop van de Foscam VD1 komt
   niet via deze service; die blijft via de Foscam-app lopen.

### 8. Firewall

Laat poort 8080 van de LXC alleen toe vanaf je LAN en je WireGuard-VPN (UniFi zone-firewall).
Er zit bewust geen login op de API.

### 9. Overgang

Draai je nu Frigate? Laat het een week parallel draaien. Draait camera-service stabiel, verwijder dan Frigate
(LXC + HA-integratie).

---

## Hardwareversnelling

- **Objectmodel**: standaard **ONNX Runtime op de CPU** (`yolo11n.onnx`). Dat werkt in elke
  container; op een i5-6500T kost YOLO11n op 320 px enkele tientallen milliseconden per beeld, en
  dat alleen bij beweging. OpenVINO (`yolo11n.xml`, `FORMAT=openvino bash deploy/export-model.sh`)
  kan de iGPU gebruiken, maar crashte in de LXC op de HD 530. De service test OpenVINO daarom eerst
  in een apart proces en laadt het model alleen als die test slaagt.
- **Decoderen van de sub-stream**: de standaard PyAV-installatie (pip) heeft **geen VAAPI**
  ingebouwd, dus de sub-stream wordt op de CPU gedecodeerd. Bij een sub-stream van 640×360 is dat
  een paar procent van één core. `hwaccel: auto` gebruikt hardware zodra de PyAV-build het
  ondersteunt (bijv. PyAV zelf gebouwd tegen de ffmpeg van Debian) en valt anders zonder fout terug.

## Webpagina: live, instellingen en filters

Op `http://<lxc-ip>:8080/` heeft elke camerategel drie knoppen:

- **▶ Live**: live beeld via go2rtc (bijv. de go2rtc-app in HA; `live: go2rtc_url:` in de config).
  camera-service geeft go2rtc door via zijn eigen adres (`/live/…`), omdat browsers een http-pagina
  geen andere adressen in het LAN laten aanspreken. Alleen de streams van de eigen camera's gaan door.
- **⚙ Instellingen**: opnameduur, gevoeligheid, labels, rondhangen en gezichtsherkenning. Werkt
  meteen en wordt bewaard in `/var/lib/camera-service/settings.json` (gaat voor op de config).
- **Masker**: zie stap 6.

Boven de tijdlijn kun je filteren op camera, soort en (met gezichtsherkenning) persoon of "Onbekend".

## Deurbelknop, rondhangen en stilstaande objecten

- **Deurbelknop**: publiceer `aanbellen` op `camsvc/<camera>/trigger` (MQTT), of
  `POST /api/cameras/<camera>/trigger`. Er start direct een opname met label *aanbellen*.
- **Rondhangen**: is een niet-herkende persoon langer dan `objects.loiter_seconds` (standaard 30 s)
  in beeld, dan gaat `binary_sensor.cam_<camera>_rondhangen` aan. Werkt het best met
  `objects.always: true` (ook zonder beweging 1x per seconde naar personen kijken, ± 1,5% CPU).
- **Stilstaande voertuigen en dieren** worden elke 30 s bewaard in
  `/var/lib/camera-service/stationary_<camera>.json`, zodat een herstart geen nieuwe events geeft.

## Gezichtsherkenning

Herkent geleerde personen (bijv. gezinsleden en familie), volledig lokaal. YuNet (OpenCV Zoo) zoekt
en lijnt het gezicht uit; **ArcFace** (InsightFace `w600k_r50`, 174 MB, ± 160 ms per gezicht op de
CPU) vergelijkt het met de geleerde foto's. Ontbreekt dat model, dan valt de service terug op SFace
(OpenCV Zoo). `install.sh` haalt alles op via `deploy/fetch-face-models.sh`. Let op: de
InsightFace-modellen zijn alleen voor niet-commercieel gebruik.

- Aanzetten per camera in de config: `faces: {enabled: true}`. Werkt het best bij de deurbel; bij
  camera's op afstand zijn gezichten vaak te klein of van opzij.
- **Beste beeld per event**: bij een bevestigde persoon haalt de service elke seconde een scherp
  beeld uit de main-stream (`source: main`, of `sub` voor minder CPU), tot 8 pogingen of tot iemand
  zeker herkend is. Elk gezicht krijgt een kwaliteitsscore (grootte, scherpte, recht van voren);
  per persoon wordt over alle beelden gestemd, het beste beeld telt het zwaarst en één misser geeft
  geen verkeerde naam. Per event worden hooguit 4 personen (beste uitsnede) bewaard.
- **Zelf bijleren** (`faces.auto_learn`, standaard aan): een gezicht dat zeker en eenduidig herkend is
  (goede kwaliteit, duidelijk niet iemand anders) en iets nieuws toevoegt, wordt bewaard in
  `faces/<naam>/auto/`. Maximaal 20 per persoon, oudste eruit. Gestippeld zichtbaar onder
  **Gezichten** en daar los te verwijderen als er toch iets verkeerds tussen zit.
- **Onbekende gezichten** (`faces.collect_unknown`, standaard aan): gezichten zonder naam worden
  30 dagen bewaard in `/var/lib/camera-service/faces_unknown/` en gegroepeerd per persoon. Onder
  **Gezichten** zie je bijv. "6× gezien"; kies een naam en de beste 8 foto's worden geleerd, alle
  events met die persoon krijgen de naam en de overige onbekenden worden opnieuw vergeleken.
  "Negeren" gooit de groep weg (bijv. een bezorger).
- **Leren via een event** kan nog steeds: open een event met een persoon en kies bij het gezicht een
  naam of "+ Nieuwe persoon…".
- **Home Assistant**: `sensor.cam_<camera>_gezicht` = naam, `onbekend` of `geen`. Events krijgen
  een lijst `faces` met naam, score, kwaliteit en uitsnede.
- Gezichten staan alleen op de server en verlaten die niet. Leer alleen huisgenoten en familie die
  ervan weten; zet `faces.collect_unknown` uit als je gezichten van voorbijgangers niet wilt bewaren.

## Wie is het? Kleding, aanwezigheid en vragen via de iPhone

- **Kleding en postuur** (YouTu-ReID uit OpenCV Zoo, Apache-2.0): bij een persoon maakt de service tot
  3 scherpe uitsneden uit de main-stream. Is het gezicht herkend en staat er maar één persoon in beeld,
  dan onthoudt hij die dag (12 uur) hoe die persoon eruitziet. Een persoon zonder herkend gezicht
  (bijv. van achteren bij het hek) die duidelijk lijkt (≥ 0,5 en 0,15 beter dan de volgende) krijgt in
  het event `vermoedelijk: [naam]` ("waarschijnlijk Sam" op de webpagina). Opslag: `kleding.json`.
- **Aanwezigheid uit HA**: automation "Camera – Aanwezigheid naar camera-service" stuurt
  `{"name": "Alex", "home": true}` naar `camsvc/server/aanwezig`. In de 5 minuten daarna is een
  persoon zonder herkend gezicht "waarschijnlijk Alex" en geeft geen rondhang-alarm. De naam van de
  person-entiteit moet gelijk zijn aan die bij Gezichten.
- **"Wie is dit?" op de iPhone**: bij een nieuw onbekend gezicht (1e en 3e keer) stuurt de service
  `event_type: onbekend_gezicht` via de event-entiteit; HA-automation "Camera – Wie is dit?" toont
  knoppen met de bekende namen, "Nieuwe persoon…" en "Negeren" en stuurt het antwoord naar
  `camsvc/server/gezicht`. De hele groep wordt dan geleerd of weggegooid.
- **Aanbellen**: bij een druk op de deurbel zoekt de service meteen in het hele scherpe beeld naar een
  gezicht (3× met 1 s ertussen), zodat de HA-melding binnen een paar seconden een naam krijgt.

## Zones, onbekende auto's en dagoverzicht

- **Zones** (⧉ op een camera → tab Zones): staan er zones, dan tellen alleen objecten die met hun
  onderkant in een zone staan. Events krijgen de zonenamen (zoeken kan ook op zone). Opslag `zones.json`.
- **Onbekende auto**: een gelezen kenteken dat niet bekend is (ook niet met één leesfout) geeft
  `event_type: onbekende_auto`: 's nachts (22-07) meteen, overdag pas als er na 15 minuten op die
  plek nog een auto staat.
- **Dagoverzicht**: `GET /api/summary?day=JJJJ-MM-DD` en op de webpagina via Vandaag/Gisteren/datum.
  Om `summary_time` (standaard 22:00) gaat het als `sensor.camera_dagoverzicht` naar HA.

## Inloggen (passkeys)

De webpagina staat op `https://camera.voorbeeld.nl` (Caddy in de container, certificaat van
Let's Encrypt via certbot met een DNS-check bij Cloudflare, `/etc/letsencrypt/renewal-hooks/deploy/caddy`).
Met `auth.enabled` mag alleen wie met een passkey is ingelogd erin; direct via `http://<ip>:8080`
geeft 401. Verzoeken vanaf de server zelf (127.0.0.1, niet via Caddy) mogen wel.

- **Eerste passkey**: de pagina vraagt om de eenmalige koppelcode (`/var/lib/camera-service/koppelcode`,
  ook in het log). Met iCloud-sleutelhanger staat de passkey daarna ook op je andere Apple-apparaten.
- **Meer apparaten / beheren**: knop **Passkeys** rechtsboven (toevoegen, verwijderen, overal uitloggen).
- **Alle passkeys kwijt**: verwijder ze uit `/var/lib/camera-service/auth.json` (of het hele bestand)
  en herstart; er komt een nieuwe koppelcode.

## Kentekenherkenning (NL)

Leest Nederlandse kentekens van voertuigen die aankomen of wegrijden, volledig lokaal
(fast-alpr: YOLOv9-detector + OCR via ONNX Runtime, ± 15 MB modellen die bij het eerste gebruik
naar `~/.cache` worden gedownload).

- Aanzetten per camera: ⚙ op de webpagina → "Kentekenherkenning (NL)", of `plates: {enabled: true}`
  in de config. Alleen zinvol bij een camera die auto's van voren/achteren en dichtbij ziet.
- Bij een bevestigd bewegend voertuig: tot 4 scherpe beelden uit de main-stream, het meest gelezen
  kenteken wint. Alleen tekst die in een NL-sidecode past telt (met hooguit één correctie van een
  typische leesfout zoals O/0 of B/8).
- **Bekende auto's**: geef een kenteken een naam via een event op de webpagina; overzicht onder
  **Kentekens** rechtsboven. Opslag: `/var/lib/camera-service/plates.json`.
- **Home Assistant**: `sensor.cam_<camera>_kenteken` (kenteken, attribuut `naam`); events krijgen een
  lijst `plates`. Filter op bekende auto in de tijdlijn.
- 's Nachts lukt het meestal niet: de reflecterende plaat wordt op infraroodbeeld een witte vlek.

## AI-beschrijvingen (lokaal vision-model)

Optioneel: een korte Nederlandse beschrijving per event (persoon, voertuig, aanbellen) door een
lokaal vision-model, bijv. `gemma3:4b` in Ollama op een aparte AI-server. Asynchroon: opnames,
detectie en meldingen wachten er nooit op.

- **Verbinding**: Ollama blijft op de AI-server alleen op `127.0.0.1:11434` luisteren. camera-service
  komt erbij via een SSH-tunnel (`deploy/camsvc-ai-tunnel.service`, gebruiker `aitunnel`) die in de
  container op `127.0.0.1:11435` luistert. De tunnelsleutel staat op de AI-server in
  `authorized_keys` met `restrict,port-forwarding,permitopen="127.0.0.1:11434",command="/bin/false"`:
  geen shell, alleen doorverbinden naar Ollama.
- **Beeld**: één scherp beeld uit de main-stream zonder detectiekaders, maskers zwart gemaakt,
  max. 1280 px en 2 MiB, opnieuw gecodeerd (geen metadata). Bij een zwart-wit/infraroodbeeld krijgt
  het model de opdracht geen kleuren te noemen; doet het dat toch, dan wordt dat gemarkeerd.
- **Limieten**: één worker, wachtrij max. 10 (één wachtend beeld per camera), 1 per camera per 30 s,
  max. 6 per minuut, deadline 180 s, hooguit één retry bij tijdelijke fouten, pauze na 5 fouten op rij.
- **Uitvoer**: schema-gecontroleerde JSON (beschrijving, objecten, beeldkwaliteit, onzekerheden), bij
  het event bewaard (zelfde bewaartermijn als de clip), zichtbaar in de speler en als
  `sensor.cam_<camera>_beschrijving`. Logs bevatten alleen status en duur, nooit beschrijvingen.

**Aanzetten**: `ai: {enabled: true}` in de config en `systemctl restart camera-service`.
**Terugdraaien**: `enabled: false` (of het blok weghalen) en herstarten; daarna
`systemctl disable --now camsvc-ai-tunnel` en op de AI-server de regel met `camsvc-ai-tunnel@<container>`
uit `~/.ssh/authorized_keys` halen. Bestaande clips en beschrijvingen blijven staan.

## Vals alarm, AI-filter en servergezondheid

- **Vals alarm**: in de speler markeer je een event als vals. Bij een persoon, voertuig of dier
  onthoudt de camera die plek (`vals_<camera>.json` in de state-map): hetzelfde soort object op
  vrijwel dezelfde plek (IoU ≥ 0,6) telt daarna niet meer. "Toch echt" draait het terug.
- **AI-filter**: de lokale AI geeft per beeld `persoon_zichtbaar`, `voertuig_zichtbaar` en
  `dier_zichtbaar`. Ziet hij niets van waarvoor het event startte, dan krijgt het event
  `vals: "ai"`: geen rondhangen-melding, grijs in de tijdlijn en niet in het dagoverzicht.
  Aanbellen is nooit vals; bij een onbruikbaar of zwak infraroodbeeld velt de AI geen oordeel.
- **Vergelijking extern/lokaal**: het HA-script `camera_ai_check` stuurt zijn oordeel naar
  `camsvc/server/extern_ai`; dat komt als `extern` bij het event. `/api/ai-vergelijking` en de
  AI-knop in de webpagina tonen hoe vaak beide het eens zijn.
- **Gezondheid**: elke minuut `camsvc/server/gezondheid` (CPU, geheugen t.o.v. MemoryMax,
  schijf, AI-server, camera's online) als sensoren in HA. Stopte de service niet netjes
  (crash, OOM-kill), dan volgt na de herstart `event.camera_service_melding` (`herstart_na_crash`).

## Links vanuit meldingen

- `/?live=<camera>` opent meteen het live-beeld, `/?camera=<camera>&open=laatste` de nieuwste
  opname van die camera, `/?clip=<camera>/<datum>/<bestand>.mp4` precies die opname.
- Niet ingelogd? Na het inloggen met je passkey ga je alsnog naar die link.
- "Alleen belangrijk" (standaard aan) verbergt "waarschijnlijk niets" en de eigen auto die komt
  of gaat (`/api/events?belangrijk=true`).

## Google Coral (USB)

Met een Coral USB draait het objectmodel op de Edge TPU. Dat scheelt een beetje CPU, maar
vooral kan de detectie vaker en beter:

- **`max_fps: auto`** wordt 5 in plaats van 2: een persoon is na ± 0,4 s bevestigd in plaats van ± 1 s.
- **`always: true`** (per camera, bijv. de deurbel): het model kijkt ook zonder beweging. Zonder
  beweging telt dan alleen een **persoon**; een geparkeerde auto of slapende kat geeft zo geen
  eindeloze events. Een poster of beeld van een persoon in beeld: wegmaskeren.
- Zonder werkende Coral valt de service automatisch terug op `model.path` (ONNX op de CPU) en
  wordt `always` genegeerd, zodat het licht blijft.

**Nieuwe container**: `CORAL=1 bash /root/camera-service/deploy/lxc/proxmox-create.sh`. Dat zet een
udev-regel op de host, geeft de USB-bus door aan de container, installeert de driver
(`libedgetpu1-std`) en `ai-edge-litert`, maakt het Coral-model en zet `model.coral_path` in de config.

**Bestaande container** (op de Proxmox-host, `<ID>` = containernummer):

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

Daarna in de container: `CORAL=1 EXPORT_MODEL=1 bash /opt/camera-service/deploy/lxc/install.sh`.

**Controleren**: in de log moet `Objectmodel yolo11n_edgetpu.tflite geladen via TFLite op Edge TPU`
staan. In `/api/status` hoort `detector` op `Edge TPU` te staan, met `infer_ms` rond 10–30 ms.
Staat er `Coral niet bruikbaar` in de log, dan ziet de container de Coral niet (USB-doorgifte,
udev-regel). Steek de Coral bij voorkeur in een USB 3-poort.

Het Coral-model is int8 en daardoor iets minder nauwkeurig dan het ONNX-model. Dat wordt ruim
goedgemaakt door de hogere kijkfrequentie. Ook `SIZE=640` (beter op afstand) is mogelijk:
`FORMAT=edgetpu SIZE=640 bash deploy/export-model.sh`. Kijk daarna naar `infer_ms`, omdat een deel
van het model dan op de CPU kan draaien.

## Ontwikkelen en testen

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest -q
.venv/bin/python -m camsvc --config config.example.yaml
```

## Licenties

AGPL-3.0, zie [`LICENSE`](../LICENSE). De modellen en hun licenties staan in
[`MODELLEN.md`](../MODELLEN.md).
