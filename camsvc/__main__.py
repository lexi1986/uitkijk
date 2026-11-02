"""Start: python -m camsvc --config /etc/camera-service/config.yaml"""

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
    """Without a usable config: run only the setup wizard."""
    from .setup import create_setup_app
    state_dir = Path(os.environ.get("CAMSVC_STATE", "/data/state"))
    port = int(os.environ.get("CAMSVC_PORT", "8080"))
    uvicorn.run(create_setup_app(config_path, state_dir, reason), host="0.0.0.0", port=port,
                log_level="warning", access_log=False)


def migrate_old_data(config_path: Path) -> None:
    """Data from the earlier Dutch version -> English names (once, before loading the config)."""
    import yaml
    from .migrate import migrate
    try:
        storage = (yaml.safe_load(config_path.read_text()) or {}).get("storage") or {}
        migrate(config_path, Path(storage.get("clips_dir", "/clips")),
                Path(storage.get("state_dir", "/var/lib/camera-service")))
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001 - a failed migration must not block startup
        logging.getLogger(__name__).exception("Migrating old data failed")


def main() -> None:
    faulthandler.enable()  # on a crash in a C library, still see where it happened
    parser = argparse.ArgumentParser(prog="camsvc")
    parser.add_argument("--config", default=os.environ.get("CAMSVC_CONFIG",
                                                           "/etc/camera-service/config.yaml"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    migrate_old_data(Path(args.config))
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        run_setup(Path(args.config), "no configuration yet")
        return
    except (ValueError, KeyError, TypeError) as exc:
        run_setup(Path(args.config), f"configuration not usable: {exc}")
        return
    logging.getLogger().setLevel(cfg.log_level)

    from .models import ensure_models
    ensure_models(cfg)
    # PyAV/ffmpeg are very chatty
    logging.getLogger("libav").setLevel(logging.ERROR)

    # keep crashes in C code (ffmpeg, ONNX Runtime) findable after the journal has rotated
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
        # Do not let the interpreter tear down: PyAV/ffmpeg decoder threads may still run while
        # their objects are gone (segfault on stop). Everything has been saved.
        logging.shutdown()
        os._exit(0)


if __name__ == "__main__":
    main()
