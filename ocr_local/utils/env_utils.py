from __future__ import annotations

import os
from pathlib import Path

from ocr_local.constants import DEFAULT_MODEL_PATH


def load_environment(env_path: Path | None = None) -> bool:
    """Загружает переменные окружения из .env, если доступен python-dotenv."""
    path = env_path or Path.cwd() / ".env"
    if not path.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(path, override=False))


def get_default_model_path() -> Path:
    """Возвращает путь к модели из окружения или проектный путь по умолчанию."""
    raw_path = os.getenv("OCR_MODEL_PATH")
    if raw_path:
        return Path(raw_path)
    return DEFAULT_MODEL_PATH

