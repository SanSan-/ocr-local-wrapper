from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from ocr_local.constants import SUPPORTED_IMAGE_EXTENSIONS
from ocr_local.exceptions import OcrValidationError
from ocr_local.model_profiles import resolve_options, settings_identity
from ocr_local.models import OcrOptions, OcrResult
from ocr_local.utils.io_utils import build_output_path, write_text_utf8_no_bom
from ocr_local.utils.output_cache import build_cache_key, get_cached_text, serialize_cache_update
from ocr_local.web.errors import WebError
from ocr_local.web.limits import DEFAULT_LIMITS, WebLimits


@dataclass(frozen=True)
class StorageUsage:
    upload_bytes: int = 0
    upload_count: int = 0
    result_bytes: int = 0
    result_count: int = 0
    cache_bytes: int = 0


def _quota_error() -> WebError:
    return WebError("storage_quota_exceeded", 507,
                    "Недостаточно места в квоте хранения. Освободите место вручную и повторите операцию.")


@contextmanager
def _storage_errors() -> Iterator[None]:
    try:
        yield
    except OSError as exc:
        raise WebError("storage_unavailable", 507,
                       "Хранилище недоступно. Проверьте свободное место и права доступа.") from exc


def _text_size(text: str, limit: int) -> int:
    if len(text) > limit:
        raise WebError("result_too_large", 413, "Текст результата превышает допустимый размер.")
    size = len(text.encode("utf-8"))
    if size > limit:
        raise WebError("result_too_large", 413, "Текст результата превышает допустимый размер.")
    return size


class StorageBudget:
    """Файлы и резервы одного веб-процесса; внешние записи не блокируются."""

    def __init__(self, uploads_dir: Path, cache_file: Path, limits: WebLimits = DEFAULT_LIMITS) -> None:
        self.uploads_dir = uploads_dir.resolve()
        self.cache_file = cache_file.resolve()
        self.limits = limits
        self._lock = threading.RLock()
        self._reservations: set[StorageReservation] = set()

    def _managed_path(self, path: Path) -> Path:
        path = path.resolve()
        if not path.is_relative_to(self.uploads_dir) or path == self.uploads_dir:
            raise WebError("invalid_upload", 400, "Изображение или результат недоступны.")
        return path

    def _usage(self) -> StorageUsage:
        upload_bytes = upload_count = text_bytes = text_count = 0
        if self.uploads_dir.exists():
            for path in self.uploads_dir.rglob("*"):
                if path.relative_to(self.uploads_dir).parts[0] == ".ocr-local-tmp":
                    continue
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix in SUPPORTED_IMAGE_EXTENSIONS:
                    upload_bytes += path.stat().st_size
                    upload_count += 1
                elif suffix == ".txt":
                    text_bytes += path.stat().st_size
                    text_count += 1
        cache_bytes = self.cache_file.stat().st_size if self.cache_file.exists() else 0
        return StorageUsage(upload_bytes, upload_count, text_bytes + cache_bytes, text_count, cache_bytes)

    def usage(self) -> StorageUsage:
        with self._lock, _storage_errors():
            return self._usage()

    def _read_bounded(self, path: Path, limit: int, status_code: int) -> str:
        if path.stat().st_size > limit:
            raise WebError("stored_data_too_large", status_code,
                           "Сохранённые данные превышают допустимый размер. Освободите место вручную.")
        with path.open("rb") as source:
            data = source.read(limit + 1)
        if len(data) > limit:
            raise WebError("stored_data_too_large", status_code,
                           "Сохранённые данные превышают допустимый размер. Освободите место вручную.")
        return data.decode("utf-8")

    def load_cache(self) -> dict[str, Any]:
        with self._lock:
            if not self.cache_file.exists():
                return {}
            try:
                cache = json.loads(self._read_bounded(self.cache_file, self.limits.cache_bytes, 507))
                if not isinstance(cache, dict):
                    raise ValueError("Кеш должен содержать объект.")
            except (OSError, ValueError) as exc:
                raise WebError("cache_unavailable", 507,
                               "Сохранённый кеш недоступен. Проверьте его в локальном хранилище.") from exc
            return cache

    def read_text(self, path: Path) -> str:
        with self._lock, _storage_errors():
            return self._read_bounded(self._managed_path(path), self.limits.text_bytes, 413)

    def _check_growth(
        self, usage: StorageUsage, text_growth: int, cache_growth: int, count: int,
        *, exclude: StorageReservation | None = None,
    ) -> None:
        others = [item for item in self._reservations if item is not exclude]
        reserved_text = sum(item.text_growth for item in others)
        reserved_cache = sum(item.cache_growth for item in others)
        reserved_count = sum(item.count for item in others)
        if (usage.result_bytes + reserved_text + reserved_cache + text_growth + cache_growth > self.limits.result_bytes
                or usage.result_count + reserved_count + count > self.limits.result_count
                or usage.cache_bytes + reserved_cache + cache_growth > self.limits.cache_bytes):
            raise _quota_error()

    def reserve(
        self, options: OcrOptions, *, validate_result: Callable[[OcrResult], None] | None = None,
    ) -> StorageReservation:
        options = resolve_options(options)
        with self._lock, _storage_errors():
            input_path = self._managed_path(options.input_path)
            output_path = self._managed_path(build_output_path(input_path, options.output_path))
            settings = settings_identity(options)
            cache = self.load_cache()
            usage = self._usage()
            cached_text = None if options.force else get_cached_text(cache, input_path, settings=settings)
            exists = output_path.is_file()
            if cached_text is not None and not output_path.is_dir():
                size = _text_size(cached_text, self.limits.text_bytes)
                text_growth, cache_growth, count = (0, 0, 0) if exists else (size, 0, 1)
                writes = not exists
            else:
                old_text_size = output_path.stat().st_size if exists else 0
                text_growth = max(0, self.limits.text_bytes - old_text_size)
                unchanged = dict(cache)
                unchanged.pop(build_cache_key(input_path, settings), None)
                base_size = len(json.dumps(unchanged, ensure_ascii=False, indent=2).encode("utf-8")) if unchanged else 0
                cache_growth = max(0, base_size + self.limits.cache_entry_bytes - usage.cache_bytes)
                count, writes = int(not exists), True
            if writes:
                self._check_growth(usage, text_growth, cache_growth, count)
            reservation = StorageReservation(self, input_path, output_path, text_growth, cache_growth, count, validate_result)
            self._reservations.add(reservation)
            return reservation

    def publish_upload(self, temp_path: Path, digest: str, suffix: str) -> Path:
        suffix = suffix.lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None or suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            raise WebError("invalid_upload", 400, "Недопустимый идентификатор изображения.")
        with self._lock, _storage_errors():
            target = self._managed_path(self.uploads_dir / (digest + suffix))
            if target.is_file():
                return target
            size = temp_path.stat().st_size
            if size <= 0 or size > self.limits.image_bytes:
                raise WebError("upload_too_large", 413, "Изображение пусто или превышает допустимый размер.")
            usage = self._usage()
            if usage.upload_bytes + size > self.limits.upload_bytes or usage.upload_count + 1 > self.limits.upload_count:
                raise _quota_error()
            self.uploads_dir.mkdir(parents=True, exist_ok=True)
            # Жёсткая ссылка публикует проверенный файл целиком и не заменяет существующий.
            try:
                os.link(temp_path, target)
            except FileExistsError:
                if not target.is_file():
                    raise
            return target


class StorageReservation:
    """Резерв до допуска OCR и точная проверка непосредственно перед записью."""

    def __init__(
        self, budget: StorageBudget, input_path: Path, output_path: Path,
        text_growth: int, cache_growth: int, count: int,
        validate_result: Callable[[OcrResult], None] | None,
    ) -> None:
        self.budget = budget
        self.input_path, self.output_path = input_path, output_path
        self.text_growth, self.cache_growth, self.count = text_growth, cache_growth, count
        self.validate_result = validate_result
        self._saved = False

    def load_cache(self) -> dict[str, Any]:
        return self.budget.load_cache()

    def release(self) -> None:
        with self.budget._lock:
            self.budget._reservations.discard(self)
            self.text_growth = self.cache_growth = self.count = 0

    def _validate(self, result: OcrResult) -> int:
        if self not in self.budget._reservations or self._saved:
            raise RuntimeError("Резерв хранения уже освобождён или использован.")
        if result.input_path.resolve() != self.input_path or result.output_path.resolve() != self.output_path:
            raise RuntimeError("Результат не соответствует резерву хранения.")
        size = _text_size(result.text, self.budget.limits.text_bytes)
        if self.validate_result is not None:
            self.validate_result(result)
        return size

    def _check_exact(self, text_size: int, cache_size: int | None) -> None:
        old_text_size = self.output_path.stat().st_size if self.output_path.is_file() else 0
        usage = self.budget._usage()
        text_growth = max(0, text_size - old_text_size)
        cache_growth = max(0, cache_size - usage.cache_bytes) if cache_size is not None else 0
        count = int(not self.output_path.exists())
        if (text_growth > self.text_growth or cache_growth > self.cache_growth or count > self.count
                or (cache_size is not None and cache_size > self.budget.limits.cache_bytes)):
            raise _quota_error()
        self.budget._check_growth(usage, text_growth, cache_growth, count, exclude=self)

    def _mark_saved(self) -> None:
        self._saved = True
        self.text_growth = self.cache_growth = self.count = 0

    def restore_result(self, result: OcrResult) -> None:
        with self.budget._lock, _storage_errors():
            text_size = self._validate(result)
            if self.output_path.is_dir():
                raise OcrValidationError("Путь результата указывает на каталог.")
            if not self.output_path.exists():
                self._check_exact(text_size, None)
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                write_text_utf8_no_bom(self.output_path, result.text)
            self._mark_saved()

    def save_result(self, result: OcrResult) -> None:
        with self.budget._lock, _storage_errors():
            text_size = self._validate(result)
            serialized = serialize_cache_update(
                self.budget.load_cache(), self.input_path, self.output_path, result.text,
                settings=result.settings, limit_reached=result.metrics.limit_reached,
            )
            self._check_exact(text_size, len(serialized.encode("utf-8")))
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            write_text_utf8_no_bom(self.output_path, result.text)
            self.budget.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.budget.cache_file.write_text(serialized, encoding="utf-8", newline="")
            self._mark_saved()
