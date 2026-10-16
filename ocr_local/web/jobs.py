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
    log_count: int = 0
    dropped: int = 0
    terminal_event: bytes | None = None
    completed_at: float | None = None
    readers: int = 0
    subscriptions: set[JobSubscription] = field(default_factory=set, repr=False)


@dataclass(eq=False)
class JobSubscription:
    job: OcrWebJob
    drains_logs: bool
    closed: bool = False
    in_flight: bytes | None = None


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
            expired = [(job_id, job) for job_id, job in self.jobs.items()
                       if job.completed_at is not None and now - job.completed_at >= self.limits.job_ttl_seconds]
            for job_id, job in expired:
                for subscription in tuple(job.subscriptions):
                    self.unsubscribe(subscription)
                self._clear_journal(job)
                job.terminal_event = None
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
                if not job.readers:
                    self._clear_journal(job)
                job.options = job.reservation = job.thread = None
            elif event.get("type") == "log":
                text = str(event.get("message", ""))
                raw = text.encode("utf-8")
                if len(raw) > self.limits.log_line_bytes:
                    text = raw[:max(0, self.limits.log_line_bytes - 3)].decode("utf-8", errors="ignore") + "…"
                encoded = encode_event({"type": "log", "message": text})
                if self._make_room(job, len(encoded)):
                    job.events.append(encoded)
                    job.log_size += len(encoded)
                    job.log_count += 1
                else:
                    self._record_drop(job)

    @staticmethod
    def _clear_journal(job: OcrWebJob) -> None:
        job.events.clear()
        job.log_size = job.log_count = job.dropped = 0

    @staticmethod
    def _record_drop(job: OcrWebJob) -> None:
        job.dropped = min(job.dropped + 1, 2**63 - 1)

    def _fits(self, job: OcrWebJob, size: int) -> bool:
        return job.log_count < self.limits.log_count and job.log_size + size <= self.limits.log_bytes

    def _evict_oldest(self, job: OcrWebJob) -> None:
        job.log_size -= len(job.events.popleft())
        job.log_count -= 1
        self._record_drop(job)

    def _make_room(self, job: OcrWebJob, size: int) -> bool:
        while job.events and not self._fits(job, size):
            self._evict_oldest(job)
        return self._fits(job, size)

    def subscribe(self, job_id: str) -> JobSubscription:
        with self.lock:
            self.cleanup()
            job = self.jobs.get(job_id)
            if job is None:
                raise WebError("job_not_found", 404, "Состояние задачи недоступно. Обновите страницу.")
            if self.subscribers >= self.limits.subscriber_count:
                raise WebError("streams_full", 429, "Открыто слишком много потоков. Повторите позже.")
            subscription = JobSubscription(job, drains_logs=job.completed_at is None)
            job.subscriptions.add(subscription)
            job.readers += int(subscription.drains_logs)
            self.subscribers += 1
            return subscription

    def acknowledge(self, subscription: JobSubscription) -> None:
        with self.lock:
            if subscription.in_flight is not None:
                job = subscription.job
                job.log_size -= len(subscription.in_flight)
                job.log_count -= 1
                subscription.in_flight = None

    def unsubscribe(self, subscription: JobSubscription) -> None:
        with self.lock:
            if subscription.closed:
                return
            self.acknowledge(subscription)
            subscription.closed = True
            job = subscription.job
            job.subscriptions.discard(subscription)
            job.readers -= int(subscription.drains_logs)
            self.subscribers -= 1
            if job.completed_at is not None and not job.readers:
                self._clear_journal(job)

    def _next_log(self, job: OcrWebJob) -> bytes | None:
        if job.dropped:
            while True:
                count = f"не менее {job.dropped}" if job.dropped == 2**63 - 1 else str(job.dropped)
                payload = encode_event({"type": "log", "message": f"Пропущено старых сообщений: {count}."})
                if self._fits(job, len(payload)):
                    job.log_size += len(payload)
                    job.log_count += 1
                    job.dropped = 0
                    return payload
                if not job.events:
                    return None
                self._evict_oldest(job)
        return job.events.popleft() if job.events else None

    def _next_payload(self, subscription: JobSubscription) -> tuple[bytes | None, bool]:
        # Вызывается под замком реестра; строка учитывается до подтверждения отправки.
        job = subscription.job
        if subscription.drains_logs and (job.dropped or job.events):
            payload = self._next_log(job)
            subscription.in_flight = payload
            return payload, False
        return job.terminal_event, job.terminal_event is not None

    async def events(self, subscription: JobSubscription):
        job = subscription.job
        try:
            while True:
                with self.lock:
                    self.cleanup()
                    if subscription.closed or job.job_id not in self.jobs:
                        return
                    if subscription.in_flight is not None:
                        raise RuntimeError("Предыдущая строка журнала ещё не отправлена.")
                    payload, terminal = self._next_payload(subscription)
                yield b"data: " + payload + b"\n\n" if payload is not None else b": keep-alive\n\n"
                waiting = payload is None
                payload = None
                if terminal:
                    return
                if waiting:
                    await asyncio.sleep(0.25)
        finally:
            self.unsubscribe(subscription)


class JobStreamResponse(StreamingResponse):
    """Медленная отправка ограничена по времени и освобождает место подписчика."""

    def __init__(self, registry: JobRegistry, subscription: JobSubscription) -> None:
        self.registry, self.subscription = registry, subscription
        self.job = subscription.job
        try:
            super().__init__(registry.events(subscription), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        except BaseException:
            registry.unsubscribe(subscription)
            raise

    async def __call__(self, scope, receive, send) -> None:
        async def bounded_send(message):
            if self.subscription.closed and not (
                    message["type"] == "http.response.body" and not message.get("more_body", False)):
                raise ClientDisconnect
            timeout = self.registry.limits.send_timeout_seconds
            if self.job.completed_at is not None:
                timeout = min(timeout, self.job.completed_at + self.registry.limits.job_ttl_seconds - self.registry.clock())
            if timeout <= 0:
                raise TimeoutError
            await asyncio.wait_for(send(message), timeout)
            if message["type"] == "http.response.body":
                self.registry.acknowledge(self.subscription)

        try:
            await super().__call__(scope, receive, bounded_send)
        except* (OSError, ClientDisconnect):
            # Разрыв и тайм-аут отправки завершают поток; подписка освобождается в finally.
            pass
        finally:
            try:
                await self.body_iterator.aclose()
            finally:
                self.registry.unsubscribe(self.subscription)
