from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable

from starlette.responses import JSONResponse

from ocr_local.web.errors import WebError, public_error
from ocr_local.web.limits import WebLimits


class RequestLimits:
    """Ограничивает тело до разбора FastAPI и резервирует единственный приём."""

    def __init__(self, app, *, limits: WebLimits, is_busy: Callable[[], bool], logger) -> None:
        self.app, self.limits, self.is_busy, self.logger = app, limits, is_busy, logger
        self.upload_slot = threading.Lock()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        upload = scope["path"] == "/api/upload"
        acquired = False
        try:
            if upload:
                if self.is_busy():
                    raise WebError("ocr_busy", 409, "Распознавание уже выполняется.")
                acquired = self.upload_slot.acquire(blocking=False)
                if not acquired:
                    raise WebError("upload_busy", 429, "Другое изображение уже загружается. Повторите позже.")
            limit = self.limits.total_body_bytes if upload else self.limits.json_bytes
            self._check_declared_length(scope, limit)
            started, last_data, received = time.monotonic(), time.monotonic(), 0

            async def bounded_receive():
                nonlocal received, last_data
                remaining = min(self.limits.upload_total_seconds - (time.monotonic() - started),
                                self.limits.upload_idle_seconds - (time.monotonic() - last_data))
                if remaining <= 0:
                    raise WebError("upload_timeout", 408, "Истекло время приёма изображения или запроса.")
                try:
                    message = await asyncio.wait_for(receive(), timeout=remaining)
                except TimeoutError as exc:
                    raise WebError("upload_timeout", 408, "Истекло время приёма изображения или запроса.") from exc
                if message["type"] == "http.request":
                    body = message.get("body", b"")
                    received += len(body)
                    if received > limit:
                        raise WebError("request_too_large", 413, "Размер запроса превышает допустимый предел.")
                    if body:
                        last_data = time.monotonic()
                return message

            await self.app(scope, bounded_receive, send)
        except WebError as exc:
            status, payload = public_error(exc, self.logger)
            await JSONResponse(payload, status_code=status)(scope, receive, send)
        finally:
            if acquired:
                self.upload_slot.release()

    @staticmethod
    def _check_declared_length(scope, limit: int) -> None:
        values = [v for k, v in scope["headers"] if k.lower() == b"content-length"]
        if len(values) > 1 or (values and not values[0].isdigit()):
            raise WebError("invalid_request", 400, "Неверный размер запроса.")
        if values and int(values[0]) > limit:
            raise WebError("request_too_large", 413, "Размер запроса превышает допустимый предел.")
