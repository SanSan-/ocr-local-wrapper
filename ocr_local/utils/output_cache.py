from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from ocr_local.constants import OCR_CACHE_FILE
from ocr_local.models import OcrMetrics
from ocr_local.utils.io_utils import read_text_utf8, write_text_utf8_no_bom


def build_cache_key(input_path: Path, settings: dict[str, object] | None = None) -> str:
    """Старые ключи доступны явно; новый сервис всегда передаёт настройки."""
    path_key = str(input_path.expanduser().resolve()).casefold()
    if settings is None:
        return path_key
    payload = json.dumps(settings, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"v2:{path_key}:{hashlib.sha256(payload).hexdigest()}"


def load_cache_snapshot(logger: logging.Logger) -> dict[str, Any]:
    if not OCR_CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(read_text_utf8(OCR_CACHE_FILE))
    except (OSError, ValueError) as exc:
        logger.warning("Не удалось прочитать кеш OCR: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def get_cached_text(
    cache: dict[str, Any], input_path: Path, *, settings: dict[str, object] | None = None,
) -> str | None:
    entry = cache.get(build_cache_key(input_path, settings))
    if not isinstance(entry, dict) or entry.get("settings") != settings:
        return None
    text = entry.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        stat = input_path.stat()
    except OSError:
        return None
    if entry.get("input_size") != stat.st_size or entry.get("input_mtime_ns") != stat.st_mtime_ns:
        return None
    return text


def apply_output_cache(
    input_path: Path, output_path: Path, logger: logging.Logger, *,
    settings: dict[str, object] | None = None, metrics: OcrMetrics | None = None,
) -> str | None:
    """Возвращает только совпавший кеш; готовый TXT не доказывает модель/параметры."""
    cache = load_cache_snapshot(logger)
    text = get_cached_text(cache, input_path, settings=settings)
    if text is None:
        return None
    if output_path.is_dir():
        return None
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_text_utf8_no_bom(output_path, text)
    if metrics is not None:
        entry = cache[build_cache_key(input_path, settings)]
        metrics.limit_reached = bool(entry.get("limit_reached", False))
    logger.info("Найден кеш OCR выбранной модели и параметров.")
    return text


def update_output_cache(
    input_path: Path, output_path: Path, text: str, logger: logging.Logger, *,
    settings: dict[str, object] | None = None, limit_reached: bool = False,
) -> None:
    """Добавляет запись, сохраняя прежние ключи и результаты."""
    cache = load_cache_snapshot(logger)
    stat = input_path.stat()
    cache[build_cache_key(input_path, settings)] = {
        "input_path": str(input_path), "output_path": str(output_path),
        "input_size": stat.st_size, "input_mtime_ns": stat.st_mtime_ns,
        "settings": settings, "text": text, "limit_reached": limit_reached,
    }
    try:
        OCR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        OCR_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Не удалось сохранить кеш OCR: %s", exc)
