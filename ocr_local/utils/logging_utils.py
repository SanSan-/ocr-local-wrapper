from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    """Настраивает вывод логов в консоль и файл."""
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(logs_dir / "ocr_local.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
