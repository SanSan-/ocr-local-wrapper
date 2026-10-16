from __future__ import annotations

import asyncio
import hashlib
import re
import threading
import uuid
from contextlib import suppress
from pathlib import Path

from PIL import Image
from starlette.datastructures import Headers, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser, parse_options_header

from ocr_local.constants import SUPPORTED_IMAGE_EXTENSIONS
from ocr_local.web.errors import WebError
from ocr_local.web.limits import WebLimits


class TemporaryUploads:
    """Удаляет только собственные незавершённые файлы выделенного каталога."""

    marker_text = "ocr-local-upload-v1\n"
    file_pattern = re.compile(r"(?:form|image)-[0-9a-f]{32}\.(?:part|jpg|jpeg|png|webp|bmp)")

    def __init__(self, uploads_dir: Path, limits: WebLimits) -> None:
        self.path, self.limits = uploads_dir / ".ocr-local-tmp", limits

    def prepare(self) -> None:
        marker = self.path / ".owner"
        if self.path.exists():
            if self.path.is_symlink() or not marker.is_file() or marker.read_text(encoding="utf-8") != self.marker_text:
                raise WebError("storage_unavailable", 507, "Временное хранилище недоступно. Проверьте локальные каталоги.")
        else:
            self.path.mkdir(parents=True)
            marker.write_text(self.marker_text, encoding="utf-8")

    def cleanup(self) -> None:
        self.prepare()
        for path in self.path.iterdir():
            if self.file_pattern.fullmatch(path.name) and path.is_file() and not path.is_symlink():
                path.unlink()

    def new_path(self, kind: str, suffix: str) -> Path:
        self.prepare()
        return self.path / f"{kind}-{uuid.uuid4().hex}{suffix}"

    def check_size(self, extra: int = 0) -> None:
        total = sum(path.stat().st_size for path in self.path.iterdir()
                    if path.is_file() and self.file_pattern.fullmatch(path.name))
        if total + extra > self.limits.temp_bytes:
            raise WebError("storage_quota_exceeded", 507, "Недостаточно места во временном хранилище.")


class UploadParser(MultiPartParser):
    """Разбирает ровно один файл и ограниченное поле настроек в собственном хранилище."""

    def __init__(self, headers, stream, workspace: TemporaryUploads) -> None:
        super().__init__(headers, stream, max_files=1, max_fields=1)
        self.max_part_size = workspace.limits.form_field_bytes
        self.workspace = workspace
        self.paths: list[Path] = []
        self.part_bytes = 0
        self.header_bytes = 0
        self.finished = False

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self.part_bytes = self.header_bytes = 0

    def on_header_field(self, data, start, end) -> None:
        self._count_header(end - start)
        super().on_header_field(data, start, end)

    def on_header_value(self, data, start, end) -> None:
        self._count_header(end - start)
        super().on_header_value(data, start, end)

    def _count_header(self, size: int) -> None:
        self.header_bytes += size
        if self.header_bytes > self.workspace.limits.form_field_bytes:
            raise WebError("request_too_large", 413, "Служебная часть загрузки слишком велика.")

    def on_headers_finished(self) -> None:
        _, options = parse_options_header(self._current_part.content_disposition)
        name = options.get(b"name", b"")
        if b"filename" not in options:
            if name != b"settings":
                raise MultiPartException("Неверное поле формы.")
            super().on_headers_finished()
            return
        if name != b"file" or self._current_files:
            raise MultiPartException("Допускается только одно изображение.")
        self._current_files += 1
        self._current_part.field_name = "file"
        path = self.workspace.new_path("form", ".part")
        self.paths.append(path)
        handle = path.open("x+b", buffering=0)
        self._files_to_close_on_error.append(handle)
        self._current_part.file = UploadFile(file=handle, size=0,
                                             filename=options[b"filename"].decode("utf-8", errors="replace"),
                                             headers=Headers(raw=self._current_part.item_headers))

    def on_part_data(self, data, start, end) -> None:
        self.part_bytes += end - start
        limit = self.workspace.limits.image_bytes if self._current_part.file else self.workspace.limits.form_field_bytes
        if self.part_bytes > limit:
            raise WebError("request_too_large", 413, "Изображение или настройки превышают допустимый размер.")
        if self._current_part.file:
            pending = sum(len(chunk) for _, chunk in self._file_parts_to_write)
            self.workspace.check_size(pending + end - start)
        super().on_part_data(data, start, end)

    def on_end(self) -> None:
        self.finished = True

    def close(self) -> None:
        for handle in self._files_to_close_on_error:
            handle.close()
        for path in self.paths:
            path.unlink(missing_ok=True)


class _StagingCancelled(Exception):
    """Рабочий поток прекращает подготовку по запросу отменённого обработчика."""


class _ImageStaging:
    """Владеет временной копией до передачи результата асинхронному обработчику."""

    def __init__(self, file: UploadFile, workspace: TemporaryUploads) -> None:
        self.file, self.workspace = file, workspace
        self.cancelled = threading.Event()
        self.path: Path | None = None

    def check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise _StagingCancelled()

    def cleanup(self) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)

    def copy(self, path: Path) -> str:
        digest, size = hashlib.sha256(), 0
        with path.open("xb") as output:
            self.path = path
            while True:
                self.check_cancelled()
                chunk = self.file.file.read(64 * 1024)
                if not chunk:
                    break
                self.check_cancelled()
                size += len(chunk)
                if size > self.workspace.limits.image_bytes:
                    raise WebError("request_too_large", 413, "Изображение превышает допустимый размер.")
                self.workspace.check_size(len(chunk))
                output.write(chunk)
                output.flush()
                digest.update(chunk)
        if not size:
            raise WebError("invalid_image", 400, "Файл изображения пуст.")
        return digest.hexdigest()

    def run(self) -> tuple[Path, str, str]:
        suffix = Path(self.file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            raise WebError("invalid_image", 400, "Неподдерживаемый формат изображения.")
        self.check_cancelled()
        path = self.workspace.new_path("image", suffix)
        try:
            digest = self.copy(path)
            self.check_cancelled()
            _verify_image(path, self.workspace.limits)
            self.check_cancelled()
            return path, digest, suffix
        except BaseException:
            self.cleanup()
            raise


def _verify_image(path: Path, limits: WebLimits) -> None:
    try:
        with Image.open(path) as image:
            if image.width * image.height > limits.image_pixels:
                raise WebError("image_too_large", 413, "Изображение содержит слишком много пикселей.")
            image.verify()
    except WebError:
        raise
    except Image.DecompressionBombError as exc:
        raise WebError("image_too_large", 413, "Изображение содержит слишком много пикселей.") from exc
    except Exception as exc:
        raise WebError("invalid_image", 400, "Файл не читается как изображение.") from exc


class _WorkerCompletion(asyncio.Future):
    """Откладывает отмену ожидающей задачи до фактического завершения потока."""

    def cancel(self, msg=None) -> bool:
        return False


async def _wait_for_worker(worker: asyncio.Future):
    completion = _WorkerCompletion()
    worker.add_done_callback(lambda _: completion.set_result(None))
    try:
        await completion
    except asyncio.CancelledError as cancellation:
        if not worker.cancelled():
            try:
                worker.result()
            except Exception as failure:
                cancellation.__cause__ = failure
        raise
    return worker.result()


async def stage_image(file: UploadFile, workspace: TemporaryUploads) -> tuple[Path, str, str]:
    staging = _ImageStaging(file, workspace)
    loop = asyncio.get_running_loop()
    worker = loop.run_in_executor(None, staging.run)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        staging.cancelled.set()
        try:
            with suppress(Exception):
                await _wait_for_worker(worker)
        finally:
            # Поток уже остановлен; очистка покрывает и отмену после его успешного возврата.
            cleanup = loop.run_in_executor(None, staging.cleanup)
            await _wait_for_worker(cleanup)
        raise
