from __future__ import annotations

import asyncio
import json
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from starlette.responses import StreamingResponse
from starlette.requests import ClientDisconnect

from ocr_local.web.errors import WebError
from ocr_local.web.limits import WebLimits


def encode_event(event: dict) -> bytes:
    return json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


@dataclass
class OcrWebJob:
    job_id: str
    options: object = None
    reservation: object = None
    thread: threading.Thread | None = None
    events: deque[bytes] = field(default_factory=deque)
    log_size: int = 0
    dropped: int = 0
    terminal_event: bytes | None = None
    completed_at: float | None = None


class JobRegistry:
    """Один активный запуск и ограниченное окно повторной доставки его итога."""

    def __init__(self, limits: WebLimits, clock=time.monotonic) -> None:
        self.limits, self.clock = limits, clock
        self.lock = threading.RLock()
        self.jobs: dict[str, OcrWebJob] = {}
        self.active_id: str | None = None
        self.subscribers = 0

    def is_busy(self) -> bool:
        with self.lock:
            return self.active_id is not None

    def cleanup(self) -> None:
        with self.lock:
            now = self.clock()
            for job_id, job in list(self.jobs.items()):
                if job.completed_at is not None and now - job.completed_at >= self.limits.job_ttl_seconds:
                    job.events.clear()
                    job.terminal_event = None
                    job.log_size = 0
                    del self.jobs[job_id]

    def check_admission(self) -> None:
        self.cleanup()
        if self.active_id is not None:
            raise WebError("ocr_busy", 409, "Распознавание уже выполняется.")
        if len(self.jobs) >= self.limits.job_count:
            deadlines = [j.completed_at + self.limits.job_ttl_seconds for j in self.jobs.values()
                         if j.completed_at is not None]
            retry = max(1, math.ceil(min(deadlines, default=self.clock() + 1) - self.clock()))
            raise WebError("jobs_full", 429, "Сохранённые итоги занимают все места. Повторите позже.",
                           headers={"Retry-After": str(retry)})

    def add(self, options, reservation) -> OcrWebJob:
        with self.lock:
            self.check_admission()
            job = OcrWebJob(uuid.uuid4().hex, options, reservation)
            self.jobs[job.job_id] = job
            self.active_id = job.job_id
            return job

    def discard_start(self, job: OcrWebJob) -> None:
        with self.lock:
            self.jobs.pop(job.job_id, None)
            if self.active_id == job.job_id:
                self.active_id = None

    def emit(self, job: OcrWebJob, event: dict) -> None:
        with self.lock:
            if job.completed_at is not None or job.job_id not in self.jobs:
                return
            if event.get("type") == "done":
                encoded = encode_event(event)
                if len(encoded) > self.limits.terminal_bytes:
                    raise WebError("result_too_large", 400, "Результат превышает допустимый размер.")
                if self.active_id == job.job_id:
                    self.active_id = None
                job.terminal_event = encoded
                job.completed_at = self.clock()
                job.events.clear()
                job.log_size = job.dropped = 0
                job.options = job.reservation = job.thread = None
            elif event.get("type") == "log" and job.completed_at is None:
                text = str(event.get("message", ""))
                raw = text.encode("utf-8")
                if len(raw) > self.limits.log_line_bytes:
                    text = raw[:max(0, self.limits.log_line_bytes - 3)].decode("utf-8", errors="ignore") + "…"
                encoded = encode_event({"type": "log", "message": text})
                while job.events and (len(job.events) >= self.limits.log_count or
                                      job.log_size + len(encoded) > self.limits.log_bytes):
                    job.log_size -= len(job.events.popleft())
                    job.dropped += 1
                if len(encoded) <= self.limits.log_bytes and self.limits.log_count:
                    job.events.append(encoded)
                    job.log_size += len(encoded)
                else:
                    job.dropped += 1

    def subscribe(self, job_id: str) -> OcrWebJob:
        with self.lock:
            self.cleanup()
            job = self.jobs.get(job_id)
            if job is None:
                raise WebError("job_not_found", 404, "Состояние задачи недоступно. Обновите страницу.")
            if self.subscribers >= self.limits.subscriber_count:
                raise WebError("streams_full", 429, "Открыто слишком много потоков. Повторите позже.")
            self.subscribers += 1
            return job

    async def events(self, job: OcrWebJob):
        while True:
            with self.lock:
                self.cleanup()
                if job.job_id not in self.jobs:
                    return
                terminal = job.terminal_event
                if terminal is not None:
                    payload = terminal
                elif job.dropped:
                    payload = encode_event({"type": "log", "message": f"Пропущено старых сообщений: {job.dropped}."})
                    job.dropped = 0
                elif job.events:
                    payload = job.events.popleft()
                    job.log_size -= len(payload)
                else:
                    payload = None
            yield b"data: " + payload + b"\n\n" if payload is not None else b": keep-alive\n\n"
            if terminal is not None:
                return
            if payload is None:
                await asyncio.sleep(0.25)


class JobStreamResponse(StreamingResponse):
    """Медленная отправка ограничена по времени и освобождает место подписчика."""

    def __init__(self, registry: JobRegistry, job: OcrWebJob) -> None:
        super().__init__(registry.events(job), media_type="text/event-stream",
                         headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        self.registry, self.job = registry, job

    async def __call__(self, scope, receive, send) -> None:
        async def bounded_send(message):
            timeout = self.registry.limits.send_timeout_seconds
            if self.job.completed_at is not None:
                timeout = min(timeout, self.job.completed_at + self.registry.limits.job_ttl_seconds - self.registry.clock())
            if timeout <= 0:
                raise TimeoutError
            await asyncio.wait_for(send(message), timeout)

        try:
            await super().__call__(scope, receive, bounded_send)
        except* (TimeoutError, OSError, ClientDisconnect):
            pass
        finally:
            with self.registry.lock:
                self.registry.subscribers -= 1
