# Uitkijk

Je eigen lichte camerasysteem voor thuis: neemt op wat er gebeurt, herkent personen, auto's en
dieren, stuurt meldingen via Home Assistant en heeft een eigen webpagina voor telefoon en laptop.
Alles draait lokaal; er gaat geen beeld naar de cloud.

- **Opnemen met de seconden ervoor**: een buffer van ± 10 s, zodat je ook ziet wie er aankwam.
- **Herkennen** van personen, voertuigen en (optioneel) dieren, met filters tegen planten, schaduw
  en geparkeerde auto's. Twijfelgevallen worden op het scherpe beeld nog eens nagekeken.
- **Gezichten** leren herkennen (optioneel), ook "wie is dit?" vanuit een melding beantwoorden.
- **Nederlandse kentekens** lezen (optioneel); bekende auto's krijgen een naam.
- **Webpagina** met tijdlijn, zoeken ("pakket", een naam, een kenteken), dagoverzicht, live beeld,
  maskers en zones tekenen, en "Niets aan de hand" om een vals alarm voortaan te negeren.
- **Home Assistant** via MQTT: sensoren per camera, snapshots, een dagoverzicht en kant-en-klare
  blueprints voor meldingen.
- **Inloggen met een passkey** (Face ID / Touch ID) als je de pagina via HTTPS met een eigen
  domein bereikbaar maakt.

![Licentie: AGPL-3.0](https://img.shields.io/badge/licentie-AGPL--3.0-blue)

## Wat heb je nodig

- Een computer die altijd aan staat met **Docker** (een mini-pc, NAS met Docker, of een server).
  Twee CPU-kernen en 2 GB geheugen is genoeg voor 3 camera's; met gezichtsherkenning liever 4 GB.
  Een Intel-processor met iGPU helpt (hardware-decoding), maar is niet nodig.
- **Camera's met RTSP**: o.a. Reolink, Foscam, Hikvision/Annke, Dahua/Amcrest, TP-Link Tapo/VIGI,
  UniFi Protect. Bij voorkeur met een tweede, lage-resolutiestroom ("substream").
- Optioneel: **Home Assistant** met de Mosquitto-broker (MQTT) voor meldingen.

## Installeren

```bash
git clone https://github.com/lexi1986/uitkijk.git
cd uitkijk
docker compose up -d
docker compose logs camsvc | grep setupcode
```

Open daarna `http://<ip-van-die-computer>:8080`. De **installatiewizard** vraagt om de setupcode
uit het log en helpt je dan met:

1. **Camera's**: kies het merk, vul IP-adres en inloggegevens in en druk op **Test**: je ziet
   meteen een beeld. Ander merk? Vul zelf het RTSP-adres in.
2. **Home Assistant** (optioneel): adres en gebruiker van de MQTT-server, met een testknop.
3. **Opties**: gezichtsherkenning, kentekens, live beeld.
4. **Opslag**: hoe lang opnames bewaard blijven en hoeveel ruimte ze mogen gebruiken.
5. **Adres**: hoe je de pagina opent, en eventueel een eigen domein voor HTTPS en passkeys.

Na **Opslaan en starten** start de service opnieuw en zie je je camera's. De eerste keer duurt
dat even langer als je gezichtsherkenning aanzette (± 500 MB aan modellen downloaden).

Alles staat daarna in de map `data/` (instellingen, gezichten, kentekens) en `clips/` (opnames).
Maak van `data/` af en toe een back-up. Instellingen per camera, maskers en zones pas je aan via
**Beheer** op de webpagina; alle overige opties staan in [`config.example.yaml`](config.example.yaml).

### Bijwerken

```bash
git pull && docker compose up -d --build
```

### Zonder Docker (Proxmox-container)

Er is ook een script dat een Debian-container op Proxmox aanmaakt en alles installeert, zie
[`docs/HANDLEIDING.md`](docs/HANDLEIDING.md#stappenplan).

## Home Assistant koppelen

1. Zorg dat de **MQTT-integratie** in Home Assistant aanstaat (met de Mosquitto-app) en vul die
   gegevens in de wizard in. De camera's verschijnen vanzelf als apparaten, met sensoren zoals
   `binary_sensor.cam_voordeur_persoon` en een snapshot-camera.
2. Importeer de **blueprints** voor meldingen (Instellingen → Automatiseringen → Blueprints →
   Blueprint importeren) met de link naar het bestand in deze repository:

| Blueprint | Wat |
|---|---|
| [persoon_melding](homeassistant/blueprints/automation/camera_service/persoon_melding.yaml) | Melding met foto als er een persoon is, met "Live bekijken" en "Opname" |
| [aanbellen](homeassistant/blueprints/automation/camera_service/aanbellen.yaml) | Opname en melding bij aanbellen, met de naam als iemand herkend wordt |
| [rondhangen](homeassistant/blueprints/automation/camera_service/rondhangen.yaml) | Dringende melding als een onbekende lang blijft staan |
| [wie_is_dit](homeassistant/blueprints/automation/camera_service/wie_is_dit.yaml) | Onbekend gezicht een naam geven vanuit de melding |
| [onbekende_auto](homeassistant/blueprints/automation/camera_service/onbekende_auto.yaml) | Onbekend kenteken 's nachts, of overdag na een kwartier |
| [dagoverzicht](homeassistant/blueprints/automation/camera_service/dagoverzicht.yaml) | Samenvatting van de dag om 22:00 |
| [aanwezigheid](homeassistant/blueprints/automation/camera_service/aanwezigheid.yaml) | Geen alarm als een bewoner net thuiskomt |
| [servergezondheid](homeassistant/blueprints/automation/camera_service/servergezondheid.yaml) | Melding bij een crash, volle schijf of offline camera |

Bij "Meldingsdiensten" vul je de notify-actie van je telefoon in, bijv.
`notify.mobile_app_iphone_van_sam` (te vinden onder Ontwikkelaarshulpmiddelen → Acties).

## Van buitenaf en inloggen

Zonder eigen domein is de webpagina open voor iedereen op je thuisnetwerk. Wil je er ook buiten
de deur bij, zet er dan een reverse proxy met HTTPS voor (Caddy, Nginx Proxy Manager, Traefik) of
gebruik een VPN (Tailscale, WireGuard). Vul in de wizard je domein in; dan moet je inloggen met
een **passkey**. De eenmalige koppelcode voor de eerste passkey staat in het log. Zet poort 8080
nooit zonder zo'n beveiliging open op internet.

## Privacy

- Beelden, gezichten en kentekens blijven op je eigen computer.
- Film je de openbare weg of de buren, dan gelden de privacyregels (in Nederland de AVG): richt
  camera's zoveel mogelijk op je eigen terrein, gebruik maskers voor de rest en bewaar opnames
  niet langer dan nodig. Gezichtsherkenning van anderen dan je huisgenoten is daarbij extra gevoelig.
- Optioneel kan een **lokaal** AI-model (Ollama) beschrijvingen bij opnames maken; zie de handleiding.

## Licentie

AGPL-3.0, zie [`LICENSE`](LICENSE). Je mag het gebruiken, aanpassen en delen; wie een aangepaste
versie verspreidt of als dienst aanbiedt, moet de broncode ook beschikbaar maken. De gebruikte
modellen hebben hun eigen licenties, zie [`MODELLEN.md`](MODELLEN.md). Het gezichtsmodel van
InsightFace is alleen voor niet-commercieel gebruik.

Meer details (alle opties, API, hardwareversnelling, Coral, AI-beschrijvingen):
[`docs/HANDLEIDING.md`](docs/HANDLEIDING.md).
