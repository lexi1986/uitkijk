"""Starten: python -m camsvc --config /etc/camera-service/config.yaml"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import signal
from pathlib import Path

import uvicorn

from .api import create_app
from .config import load_config
from .service import Service


def run_setup(config_path: Path, reason: str) -> None:
    """Zonder bruikbare config: alleen de installatiewizard draaien."""
    from .setup import create_setup_app
    state_dir = Path(os.environ.get("CAMSVC_STATE", "/data/state"))
    port = int(os.environ.get("CAMSVC_PORT", "8080"))
    uvicorn.run(create_setup_app(config_path, state_dir, reason), host="0.0.0.0", port=port,
                log_level="warning", access_log=False)


def main() -> None:
    faulthandler.enable()  # bij een crash in een C-bibliotheek toch zien waar het gebeurde
    parser = argparse.ArgumentParser(prog="camsvc")
    parser.add_argument("--config", default=os.environ.get("CAMSVC_CONFIG",
                                                           "/etc/camera-service/config.yaml"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        run_setup(Path(args.config), "nog geen configuratie")
        return
    except (ValueError, KeyError, TypeError) as exc:
        run_setup(Path(args.config), f"configuratie onbruikbaar: {exc}")
        return
    logging.getLogger().setLevel(cfg.log_level)

    from .models import ensure_models
    ensure_models(cfg)
    # PyAV/ffmpeg zijn erg spraakzaam
    logging.getLogger("libav").setLevel(logging.ERROR)

    # een crash in C-code (ffmpeg, ONNX Runtime) ook na het lezen van het log nog terug kunnen vinden
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    faulthandler.enable(file=open(cfg.state_dir / "crash.log", "a"), all_threads=True)  # noqa: SIM115

    svc = Service(cfg)
    svc.start()

    server = uvicorn.Server(uvicorn.Config(create_app(svc), host=cfg.api_host, port=cfg.api_port,
                                           log_level="warning", access_log=False))

    def shutdown(*_):
        server.should_exit = True

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.run()
    finally:
        svc.stop()
        # Niet de interpreter laten opruimen: decoder-threads van PyAV/ffmpeg kunnen dan nog
        # lopen terwijl hun objecten al weg zijn (segfault bij het stoppen). Alles is bewaard.
        logging.shutdown()
        os._exit(0)


if __name__ == "__main__":
    main()
