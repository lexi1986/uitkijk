"""MQTT connection to Home Assistant via MQTT discovery."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from typing import TYPE_CHECKING, Callable

import paho.mqtt.client as mqtt

from . import __version__

if TYPE_CHECKING:
    from .config import CameraConfig, Config
    from .events import EventRecord, EventStore

log = logging.getLogger(__name__)

BINARY_SENSORS = {
    # key: (name, device_class, entity_category)
    "motion": ("Motion", "motion", None),
    "person": ("Person", "occupancy", None),
    "vehicle": ("Vehicle", "occupancy", None),
    "animal": ("Animal", "occupancy", None),
    "loitering": ("Loitering", "occupancy", None),
    "online": ("Online", "connectivity", "diagnostic"),
}
SWITCHES = {"detect": "Detection", "record": "Recording"}
EVENT_TYPES = ["doorbell", "person", "vehicle", "animal", "motion", "unknown_vehicle", "unknown_face"]

# Health of the server itself: one JSON message, one sensor per key.
HEALTH_SENSORS = {
    "started": {"name": "Started", "device_class": "timestamp", "icon": "mdi:restart"},
    "cpu": {"name": "CPU", "unit_of_measurement": "%", "state_class": "measurement", "icon": "mdi:cpu-64-bit"},
    "memory": {"name": "Memory", "unit_of_measurement": "%", "state_class": "measurement",
               "icon": "mdi:memory"},
    "memory_mb": {"name": "Memory used", "unit_of_measurement": "MB", "state_class": "measurement",
                  "icon": "mdi:memory"},
    "disk": {"name": "Disk", "unit_of_measurement": "%", "state_class": "measurement",
             "icon": "mdi:harddisk"},
    "disk_free_gb": {"name": "Disk free", "unit_of_measurement": "GB", "state_class": "measurement",
                     "icon": "mdi:harddisk"},
    "ai_server": {"name": "AI server", "icon": "mdi:robot-outline"},
    "cameras_online": {"name": "Cameras online", "state_class": "measurement", "icon": "mdi:cctv"},
}
SERVER_EVENTS = ["restart_after_crash"]

SwitchHandler = Callable[[str, str, bool], None]  # (camera, switch, on)


class MqttBridge:
    def __init__(self, cfg: "Config", store: "EventStore"):
        self.cfg = cfg
        self.store = store
        self.prefix = cfg.mqtt["topic_prefix"]
        self.discovery = cfg.mqtt["discovery_prefix"]
        self.status_topic = f"{self.prefix}/status"
        self._switch_handler: SwitchHandler | None = None
        self._trigger_handler: Callable[[str, str], None] | None = None
        self._cameras: list[CameraConfig] = []
        self._state_getter: Callable[[str], dict[str, bool]] | None = None
        self.face_handler = None       # (sample, name, ignore) from a phone notification
        self.presence_handler = None   # (name, home) from HA
        self.external_handler = None   # (camera, verdict) from the external AI in HA
        self.connected = threading.Event()

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="camera-service")
        if cfg.mqtt.get("username"):
            self.client.username_pw_set(cfg.mqtt["username"], cfg.mqtt.get("password"))
        self.client.will_set(self.status_topic, "offline", qos=1, retain=True)
        self.client.reconnect_delay_set(1, 60)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    # ------------------------------------------------------------------ setup
    def start(self, cameras: list["CameraConfig"], switch_handler: SwitchHandler,
              state_getter: Callable[[str], dict[str, bool]],
              trigger_handler: Callable[[str, str], None] | None = None) -> None:
        self._cameras = cameras
        self._switch_handler = switch_handler
        self._trigger_handler = trigger_handler
        self._state_getter = state_getter
        self.client.connect_async(self.cfg.mqtt["host"], int(self.cfg.mqtt["port"]), keepalive=30)
        self.client.loop_start()

    def stop(self) -> None:
        self.client.publish(self.status_topic, "offline", qos=1, retain=True)
        self.client.disconnect()
        self.client.loop_stop()

    def _on_connect(self, client, _userdata, _flags, reason_code, _props):
        if reason_code.is_failure:
            log.error("MQTT connection refused: %s", reason_code)
            return
        log.info("MQTT connected to %s", self.cfg.mqtt["host"])
        client.subscribe(f"{self.prefix}/+/+/set", qos=1)
        client.subscribe(f"{self.prefix}/+/trigger", qos=1)          # e.g. doorbell button from HA
        client.subscribe(f"{self.prefix}/server/face", qos=1)         # naming a face from a notification
        client.subscribe(f"{self.prefix}/server/presence", qos=1)     # resident home/away from HA
        client.subscribe(f"{self.prefix}/server/external_ai", qos=1)  # verdict of the external AI
        client.subscribe(f"{self.discovery}/status", qos=1)  # HA restarted -> announce again
        self._announce_all()
        client.publish(self.status_topic, "online", qos=1, retain=True)
        self.connected.set()

    def _on_disconnect(self, _client, _userdata, _flags, reason_code, _props):
        self.connected.clear()
        log.warning("MQTT connection lost (%s), reconnecting…", reason_code)

    def _on_message(self, client, _userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode(errors="ignore").strip()
        if topic == f"{self.discovery}/status":
            if payload == "online":
                self._announce_all()
                client.publish(self.status_topic, "online", qos=1, retain=True)
            return
        if topic == f"{self.prefix}/server/face" and self.face_handler:
            # {"sample": "<id>", "name": "Sam"} or {"sample": "<id>", "ignore": true}
            try:
                data = json.loads(payload)
                self.face_handler(str(data.get("sample", "")), (data.get("name") or "").strip() or None,
                                  bool(data.get("ignore")))
            except (ValueError, AttributeError):
                log.warning("Invalid face message: %s", payload[:100])
            return
        if topic == f"{self.prefix}/server/presence" and self.presence_handler:
            # {"name": "Alex", "home": true}; skip retained messages at startup
            try:
                data = json.loads(payload)
                if not msg.retain and data.get("name"):
                    self.presence_handler(str(data["name"]), bool(data.get("home")))
            except (ValueError, AttributeError):
                log.warning("Invalid presence message: %s", payload[:100])
            return
        if topic == f"{self.prefix}/server/external_ai" and self.external_handler:
            # {"camera": "doorbell", "notify": true, "title": "...", "description": "..."}
            try:
                data = json.loads(payload)
                if not msg.retain and data.get("camera"):
                    self.external_handler(str(data["camera"]).lower(), {
                        "notify": bool(data.get("notify")), "title": str(data.get("title") or "")[:120],
                        "description": str(data.get("description") or "")[:600],
                        "model": str(data.get("model") or "")[:60]})
            except (ValueError, AttributeError):
                log.warning("Invalid external AI message")
            return
        parts = topic.split("/")
        # <prefix>/<camera>/trigger  (payload = label, default doorbell)
        if len(parts) == 3 and parts[2] == "trigger" and self._trigger_handler:
            self._trigger_handler(parts[1], payload or "doorbell")
            return
        # <prefix>/<camera>/<switch>/set
        if len(parts) == 4 and parts[3] == "set" and parts[2] in SWITCHES and self._switch_handler:
            self._switch_handler(parts[1], parts[2], payload.upper() == "ON")

    # ------------------------------------------------------------------ discovery
    def _device(self, cam: "CameraConfig") -> dict:
        return {
            "identifiers": [f"camsvc_{cam.name}"],
            "name": cam.friendly_name,
            "manufacturer": "Uitkijk",
            "model": "Video buffer & detection",
            "sw_version": __version__,
        }

    def _avail(self, cam_name: str) -> dict:
        return {
            "availability": [{"topic": self.status_topic}],
        }

    def reannounce(self) -> None:
        if self.connected.is_set():
            self._announce_all()

    def _announce_server(self) -> None:
        """Entities of the service itself (not per camera)."""
        dev = {"identifiers": ["camsvc_server"], "name": "Uitkijk",
               "manufacturer": "Uitkijk", "model": "Server", "sw_version": __version__}
        for key, name, icon in (("daily_summary", "Daily summary", "mdi:calendar-text"),
                                ("today", "Today", "mdi:calendar-today")):
            self.client.publish(f"{self.discovery}/sensor/camsvc_server/{key}/config", json.dumps({
                "name": name,
                "unique_id": f"camsvc_server_{key}",
                "default_entity_id": f"sensor.camera_{key}",
                "state_topic": f"{self.prefix}/server/{key}",
                "value_template": "{{ value_json.text[:250] }}",
                "json_attributes_topic": f"{self.prefix}/server/{key}",
                "icon": icon,
                "device": dev,
                **self._avail("server"),
            }), qos=1, retain=True)
        health = f"{self.prefix}/server/health"
        for key, extra in HEALTH_SENSORS.items():
            self.client.publish(f"{self.discovery}/sensor/camsvc_server/{key}/config", json.dumps({
                "unique_id": f"camsvc_server_{key}",
                "default_entity_id": f"sensor.camera_service_{key}",
                "state_topic": health,
                "value_template": f"{{{{ value_json.{key} }}}}",
                "entity_category": "diagnostic",
                "device": dev,
                **extra,
                **self._avail("server"),
            }), qos=1, retain=True)
        self.client.publish(f"{self.discovery}/event/camsvc_server/alert/config", json.dumps({
            "name": "Server alert",
            "unique_id": "camsvc_server_alert",
            "default_entity_id": "event.camera_service_alert",
            "state_topic": f"{self.prefix}/server/alert",
            "event_types": SERVER_EVENTS,
            "icon": "mdi:server-network-alert",
            "device": dev,
            **self._avail("server"),
        }), qos=1, retain=True)

    def publish_health(self, health: dict) -> None:
        self.client.publish(f"{self.prefix}/server/health", json.dumps(health), qos=1, retain=True)

    def publish_server_event(self, event_type: str, data: dict) -> None:
        self.client.publish(f"{self.prefix}/server/alert",
                            json.dumps({"event_type": event_type, **data}), qos=1, retain=False)

    def publish_today(self, summary: dict) -> None:
        """Running total for today (separate from the daily summary, which notifies at 22:00)."""
        self.client.publish(f"{self.prefix}/server/today", json.dumps(summary, ensure_ascii=False),
                            qos=1, retain=True)

    def publish_summary(self, summary: dict) -> None:
        self.client.publish(f"{self.prefix}/server/daily_summary", json.dumps(summary, ensure_ascii=False),
                            qos=1, retain=True)

    def _announce_all(self) -> None:
        self._announce_server()
        for cam in self._cameras:
            self._announce(cam)
            # publish the current state again (retained)
            if self._state_getter:
                for which, on in self._state_getter(cam.name).items():
                    if which in SWITCHES:
                        self.publish_switch(cam.name, which, on)
                    elif which in BINARY_SENSORS:
                        self.publish_binary(cam.name, which, on)
                    elif which == "face" and cam.settings["faces"]["enabled"]:
                        self.publish_face(cam.name, on)
                    elif which == "plate" and cam.settings["plates"]["enabled"]:
                        self.publish_plate(cam.name, on, None)
            self.publish_recent(cam.name)

    def _cfg_topic(self, component: str, cam: str, key: str) -> str:
        return f"{self.discovery}/{component}/camsvc_{cam}/{key}/config"

    def _announce(self, cam: "CameraConfig") -> None:
        dev = self._device(cam)
        base = f"{self.prefix}/{cam.name}"
        pub = lambda comp, key, cfg: self.client.publish(  # noqa: E731
            self._cfg_topic(comp, cam.name, key), json.dumps(cfg), qos=1, retain=True)

        for key, (name, dclass, category) in BINARY_SENSORS.items():
            cfg = {
                "name": name,
                "unique_id": f"camsvc_{cam.name}_{key}",
                "default_entity_id": f"binary_sensor.cam_{cam.name}_{key}",
                "state_topic": f"{base}/{key}",
                "device_class": dclass,
                "device": dev,
                **self._avail(cam.name),
            }
            if category:
                cfg["entity_category"] = category
            pub("binary_sensor", key, cfg)

        for key, name in SWITCHES.items():
            pub("switch", key, {
                "name": name,
                "unique_id": f"camsvc_{cam.name}_{key}",
                "default_entity_id": f"switch.cam_{cam.name}_{key}",
                "state_topic": f"{base}/{key}",
                "command_topic": f"{base}/{key}/set",
                "icon": "mdi:motion-sensor" if key == "detect" else "mdi:record-rec",
                "device": dev,
                "entity_category": "config",
                **self._avail(cam.name),
            })

        if not cam.settings["faces"]["enabled"]:
            self.client.publish(self._cfg_topic("sensor", cam.name, "face"), "", qos=1, retain=True)
        else:
            pub("sensor", "face", {
                "name": "Face",
                "unique_id": f"camsvc_{cam.name}_face",
                "default_entity_id": f"sensor.cam_{cam.name}_face",
                "state_topic": f"{base}/face",
                "icon": "mdi:face-recognition",
                "device": dev,
                **self._avail(cam.name),
            })

        if not cam.settings["plates"]["enabled"]:
            self.client.publish(self._cfg_topic("sensor", cam.name, "plate"), "", qos=1, retain=True)
        else:
            pub("sensor", "plate", {
                "name": "Licence plate",
                "unique_id": f"camsvc_{cam.name}_plate",
                "default_entity_id": f"sensor.cam_{cam.name}_plate",
                "state_topic": f"{base}/plate",
                "value_template": "{{ value_json.plate }}",
                "json_attributes_topic": f"{base}/plate",
                "json_attributes_template": "{{ {'name': value_json.name} | tojson }}",
                "icon": "mdi:card-text-outline",
                "device": dev,
                **self._avail(cam.name),
            })

        if self.cfg.ai.get("enabled"):
            pub("sensor", "ai", {
                "name": "AI description",
                "unique_id": f"camsvc_{cam.name}_ai",
                "default_entity_id": f"sensor.cam_{cam.name}_description",
                "state_topic": f"{base}/description",
                "value_template": "{{ value_json.description[:250] }}",
                "json_attributes_topic": f"{base}/description",
                "icon": "mdi:text-box-search-outline",
                "device": dev,
                **self._avail(cam.name),
            })

        pub("camera", "snapshot", {
            "name": "Snapshot",
            "unique_id": f"camsvc_{cam.name}_snapshot",
            "default_entity_id": f"camera.cam_{cam.name}_snapshot",
            "topic": f"{base}/snapshot",
            "device": dev,
            **self._avail(cam.name),
        })

        pub("event", "event", {
            "name": "Event",
            "unique_id": f"camsvc_{cam.name}_event",
            "default_entity_id": f"event.cam_{cam.name}_event",
            "state_topic": f"{base}/event",
            "event_types": EVENT_TYPES,
            "icon": "mdi:cctv",
            "device": dev,
            **self._avail(cam.name),
        })

        pub("sensor", "last_event", {
            "name": "Last event",
            "unique_id": f"camsvc_{cam.name}_last_event",
            "default_entity_id": f"sensor.cam_{cam.name}_last_event",
            "state_topic": f"{base}/events",
            "value_template": "{{ value_json.state }}",
            "json_attributes_topic": f"{base}/events",
            "json_attributes_template": "{{ {'events': value_json.events} | tojson }}",
            "icon": "mdi:history",
            "device": dev,
            **self._avail(cam.name),
        })

    # ------------------------------------------------------------------ publishing
    def publish_binary(self, cam: str, key: str, on: bool) -> None:
        self.client.publish(f"{self.prefix}/{cam}/{key}", "ON" if on else "OFF", qos=1, retain=True)

    def publish_switch(self, cam: str, key: str, on: bool) -> None:
        self.client.publish(f"{self.prefix}/{cam}/{key}", "ON" if on else "OFF", qos=1, retain=True)

    def publish_ai(self, cam: str, event_id: str, result: dict) -> None:
        payload = {k: result.get(k) for k in ("description", "objects", "attributes", "image_quality",
                                               "uncertainties", "infrared", "captured", "model",
                                               "person_visible", "vehicle_visible",
                                               "animal_visible", "false_alarm")}
        payload["event_id"] = event_id
        self.client.publish(f"{self.prefix}/{cam}/description", json.dumps(payload, ensure_ascii=False),
                            qos=1, retain=True)

    def publish_plate(self, cam: str, plate: str, name: str | None) -> None:
        self.client.publish(f"{self.prefix}/{cam}/plate",
                            json.dumps({"plate": plate, "name": name}), qos=1, retain=True)

    def publish_alert(self, cam: str, event_type: str, data: dict) -> None:
        """Separate alert via the event entity, e.g. an unknown vehicle on the driveway."""
        self.client.publish(f"{self.prefix}/{cam}/event",
                            json.dumps({"event_type": event_type, **data}), qos=1, retain=False)

    def publish_face(self, cam: str, name: str) -> None:
        self.client.publish(f"{self.prefix}/{cam}/face", name, qos=1, retain=True)

    def publish_snapshot(self, cam: str, jpg: bytes) -> None:
        self.client.publish(f"{self.prefix}/{cam}/snapshot", jpg, qos=0, retain=True)

    def publish_event(self, cam: str, record: "EventRecord") -> None:
        payload = {"event_type": record.label, **asdict(record), **record.urls(self.store.public_url)}
        self.client.publish(f"{self.prefix}/{cam}/event", json.dumps(payload), qos=1, retain=False)
        self.publish_recent(cam)

    def publish_recent(self, cam: str) -> None:
        events = self.store.as_payload(cam)
        # keep the attributes compact (the HA recorder truncates large attributes)
        slim = [{**{k: e[k] for k in ("label", "start", "duration", "score", "clip_url", "snapshot_url")},
                 "names": [f["name"] for f in (e.get("faces") or []) if f["name"]]
                 or (e.get("likely") or []),
                 "plates": [p["plate"] for p in (e.get("plates") or [])],
                 "false_alarm": e.get("false_alarm")}
                for e in events]
        state = events[0]["start"] if events else "none"
        self.client.publish(f"{self.prefix}/{cam}/events",
                            json.dumps({"state": state, "events": slim}), qos=1, retain=True)
