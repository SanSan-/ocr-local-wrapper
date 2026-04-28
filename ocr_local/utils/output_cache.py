from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ocr_local.constants import OCR_CACHE_FILE
from ocr_local.utils.io_utils import read_text_utf8, write_text_utf8_no_bom


def build_cache_key(input_path: Path) -> str:
    """Строит ключ кеша по абсолютному пути изображения."""
    return str(input_path.expanduser().resolve()).casefold()


def load_cache_snapshot(logger: logging.Logger) -> dict[str, Any]:
    """Читает кеш OCR в память."""
    return _load_cache(logger)


def get_cached_text(
    cache: dict[str, Any],
    input_path: Path,
) -> str | None:
    """Возвращает текст из кеша, если файл изображения не изменился."""
    cache_key = build_cache_key(input_path)
    return _extract_cached_text(cache.get(cache_key), input_path)


def apply_output_cache(
    input_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> str | None:
    """Восстанавливает результат OCR из кеша и возвращает текст."""
    cache_key = build_cache_key(input_path)
    cache = _load_cache(logger)
    cached_text = _extract_cached_text(cache.get(cache_key), input_path)
    if cached_text is None:
        if output_path.exists():
            try:
                existing_text = read_text_utf8(output_path)
            except OSError as exc:
                logger.warning("Не удалось прочитать файл результата для кеша (%s): %s", cache_key, exc)
                return None
            if not existing_text.strip():
                logger.warning("Файл результата пустой, кеш OCR не обновлен (%s).", cache_key)
                return None
            cache[cache_key] = _build_cache_entry(input_path, output_path, existing_text)
            _save_cache(cache, logger)
            logger.info("Файл результата уже существует, сохранен в кеш OCR (%s).", cache_key)
            return existing_text
        return None

    if output_path.exists():
        logger.info("Кеш OCR найден (%s), файл результата уже существует.", cache_key)
        return cached_text

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_text_utf8_no_bom(output_path, cached_text)
    except OSError as exc:
        logger.warning("Не удалось восстановить OCR из кеша (%s): %s", cache_key, exc)
        return None
    logger.info("OCR восстановлен из кеша (%s).", cache_key)
    return cached_text


def update_output_cache(
    input_path: Path,
    output_path: Path,
    text: str,
    logger: logging.Logger,
) -> None:
    """Сохраняет результат OCR в кеш."""
    cache_key = build_cache_key(input_path)
    cache = _load_cache(logger)
    entry = _build_cache_entry(input_path, output_path, text)
    if cache.get(cache_key) == entry:
        return
    cache[cache_key] = entry
    _save_cache(cache, logger)


def _load_cache(logger: logging.Logger) -> dict[str, Any]:
    if not OCR_CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(read_text_utf8(OCR_CACHE_FILE))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Не удалось прочитать кеш OCR: %s", exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("Формат кеша OCR не распознан, начинаю с пустого состояния.")
        return {}
    return data


def _save_cache(cache: dict[str, Any], logger: logging.Logger) -> None:
    try:
        OCR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        OCR_CACHE_FILE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Не удалось сохранить кеш OCR: %s", exc)


def _extract_cached_text(entry: Any, input_path: Path) -> str | None:
    if not isinstance(entry, dict):
        return None
    text = entry.get("text")
    if not isinstance(text, str) or not text:
        return None
    if not _is_same_input(entry, input_path):
        return None
    return text


def _is_same_input(entry: dict[str, Any], input_path: Path) -> bool:
    try:
        stat = input_path.stat()
    except OSError:
        return False
    return (
        entry.get("input_size") == stat.st_size
        and entry.get("input_mtime_ns") == stat.st_mtime_ns
    )


def _build_cache_entry(input_path: Path, output_path: Path, text: str) -> dict[str, Any]:
    stat = input_path.stat()
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "text": text,
    }


__all__ = [
    "apply_output_cache",
    "build_cache_key",
    "get_cached_text",
    "load_cache_snapshot",
    "update_output_cache",
]
