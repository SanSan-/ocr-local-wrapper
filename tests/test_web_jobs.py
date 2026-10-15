from __future__ import annotations

import asyncio
import gc
import json
import threading
import weakref
from dataclasses import replace

import pytest
from starlette.requests import ClientDisconnect

from ocr_local.web import app as web_app
from ocr_local.web.errors import WebError
from ocr_local.web.jobs import JobRegistry, JobStreamResponse
from ocr_local.web.limits import DEFAULT_LIMITS


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class HeavyReference:
    pass


def done(status="ok"):
    if status == "error":
        return {"type": "done", "status": "error", "error": "Проверочный отказ.",
                "code": "model_error", "diagnostic_id": "test-diagnostic"}
    return {"type": "done", "status": "ok", "result": {"text": "Результат", "model_id": "glm-ocr"}}


def registry_with_job(*, clock=None, limits=DEFAULT_LIMITS):
    registry = JobRegistry(limits, clock=clock or Clock())
    return registry, registry.add(HeavyReference(), HeavyReference())


def scope(version="2.4"):
    return {"type": "http", "method": "GET", "path": "/api/stream/test", "headers": [],
            "asgi": {"version": "3.0", "spec_version": version}}


async def pending_receive():
    await asyncio.Future()


async def collect_response(registry, job, *, version="2.4"):
    messages = []

    async def send(message):
        messages.append(message)

    response = JobStreamResponse(registry, registry.subscribe(job.job_id))
    await response(scope(version), pending_receive, send)
    return b"".join(message.get("body", b"") for message in messages)


def event_payload(chunk):
    assert chunk.startswith(b"data: ") and chunk.endswith(b"\n\n")
    return json.loads(chunk[6:-2])


def exception_leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in exception_leaves(child)]
    return [error]


def test_active_job_does_not_expire_and_ttl_starts_at_completion():
    clock = Clock()
    registry, job = registry_with_job(clock=clock)
    clock.now += 4 * 1800
    registry.cleanup()
    assert registry.jobs[job.job_id] is job
    assert registry.active_id == job.job_id
    assert job.completed_at is None

    registry.emit(job, done())
    completed_at = clock.now
    assert job.completed_at == completed_at
    assert registry.active_id is None
    clock.now = completed_at + 1800 - 0.001
    registry.cleanup()
    assert job.job_id in registry.jobs
    clock.now = completed_at + 1800
    registry.cleanup()
    assert job.job_id not in registry.jobs
    assert job.terminal_event is None
    with pytest.raises(WebError) as failure:
        registry.subscribe(job.job_id)
    assert (failure.value.status_code, failure.value.code) == (404, "job_not_found")
    assert registry.subscribers == 0


def test_periodic_cleanup_runs_within_thirty_seconds_after_expiry(monkeypatch):
    clock = Clock()
    registry, job = registry_with_job(clock=clock)
    registry.emit(job, done())
    expiry = clock.now + 1800
    clock.now = expiry - 0.001
    intervals = []

    async def advance_then_stop(seconds):
        if intervals:
            raise asyncio.CancelledError
        intervals.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(web_app, "_registry", registry)
    monkeypatch.setattr(web_app, "LIMITS", DEFAULT_LIMITS)
    monkeypatch.setattr(web_app.asyncio, "sleep", advance_then_stop)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(web_app._clean_jobs())
    assert intervals == [30.0]
    assert expiry <= clock.now <= expiry + 30
    assert job.job_id not in registry.jobs


def test_thirty_two_slots_are_retained_and_retry_after_tracks_nearest_expiry():
    clock = Clock(0.0)
    registry = JobRegistry(DEFAULT_LIMITS, clock=clock)
    jobs = []
    for index in range(32):
        clock.now = float(index)
        job = registry.add(HeavyReference(), HeavyReference())
        registry.emit(job, done())
        jobs.append(job)
    original = {job.job_id: job.terminal_event for job in jobs}
    clock.now = 31.25
    with pytest.raises(WebError) as failure:
        registry.add(HeavyReference(), HeavyReference())
    assert (failure.value.status_code, failure.value.code) == (429, "jobs_full")
    assert failure.value.headers["Retry-After"] == "1769"
    assert {job_id: job.terminal_event for job_id, job in registry.jobs.items()} == original
    assert registry.active_id is None

    clock.now = 1800.0
    next_job = registry.add(HeavyReference(), HeavyReference())
    assert len(registry.jobs) == 32
    assert jobs[0].job_id not in registry.jobs
    assert all(job.job_id in registry.jobs for job in jobs[1:])
    assert registry.active_id == next_job.job_id


def test_active_slot_refuses_second_job_and_failed_start_frees_slot():
    registry, job = registry_with_job()
    with pytest.raises(WebError) as failure:
        registry.add(None, None)
    assert (failure.value.status_code, failure.value.code) == (409, "ocr_busy")
    assert list(registry.jobs) == [job.job_id]
    registry.discard_start(job)
    assert not registry.jobs and registry.active_id is None
    replacement = registry.add(None, None)
    assert registry.active_id == replacement.job_id


@pytest.mark.parametrize("status", ["ok", "error"])
@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_terminal_replay_is_not_consumed_and_does_not_create_jobs(status, version):
    registry, job = registry_with_job()
    registry.emit(job, {"type": "log", "message": "Предшествующая строка."})
    event = done(status)
    registry.emit(job, event)
    original = job.terminal_event
    event["status"] = "changed-after-publication"

    async def reconnect():
        for _ in range(3):
            body = await collect_response(registry, job, version=version)
            assert body == b"data: " + original + b"\n\n"
            assert registry.subscribers == 0

    asyncio.run(reconnect())
    assert job.terminal_event == original
    assert list(registry.jobs) == [job.job_id]
    assert registry.active_id is None
    assert not job.events and job.log_size == 0


def test_terminal_publication_releases_heavy_references_and_ignores_late_logs():
    registry, job = registry_with_job()
    job.thread = threading.Thread()
    references = [weakref.ref(job.options), weakref.ref(job.reservation), weakref.ref(job.thread)]
    registry.emit(job, done())
    registry.emit(job, {"type": "log", "message": "Опоздавшая строка."})
    gc.collect()
    assert all(reference() is None for reference in references)
    assert job.options is job.reservation is job.thread is None
    assert not job.events and job.log_size == 0


def test_repeated_completion_cannot_replace_terminal_or_extend_retention():
    clock = Clock()
    registry, job = registry_with_job(clock=clock)
    registry.emit(job, done("ok"))
    original, completed_at = job.terminal_event, job.completed_at
    clock.now += 1000
    registry.emit(job, done("error"))
    assert job.terminal_event == original
    assert job.completed_at == completed_at


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_four_subscriptions_reject_fifth_and_free_places_after_response(version):
    registry, job = registry_with_job()
    registry.emit(job, done())
    responses = [JobStreamResponse(registry, registry.subscribe(job.job_id)) for _ in range(4)]
    with pytest.raises(WebError) as failure:
        registry.subscribe(job.job_id)
    assert (failure.value.status_code, failure.value.code) == (429, "streams_full")
    assert registry.subscribers == 4

    async def run_responses():
        async def send(message):
            pass

        await responses.pop()(scope(version), pending_receive, send)
        assert registry.subscribers == 3
        responses.append(JobStreamResponse(registry, registry.subscribe(job.job_id)))
        assert registry.subscribers == 4
        for response in responses:
            await response(scope(version), pending_receive, send)

    asyncio.run(run_responses())
    assert registry.subscribers == 0


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_cancelling_stream_frees_subscriber_and_keeps_active_job(version):
    registry, job = registry_with_job()

    async def cancel_response():
        entered = asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.body":
                entered.set()
                await asyncio.Future()

        response = JobStreamResponse(registry, registry.subscribe(job.job_id))
        task = asyncio.create_task(response(scope(version), pending_receive, send))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        try:
            await task
        except BaseException as error:
            assert all(isinstance(leaf, asyncio.CancelledError) for leaf in exception_leaves(error))
        else:
            pytest.fail("Отмена подписки не должна исчезать без уведомления вызывающей стороны.")

    asyncio.run(cancel_response())
    assert registry.subscribers == 0
    assert registry.active_id == job.job_id
    assert job.completed_at is None


def test_client_disconnect_frees_subscriber():
    registry, job = registry_with_job()

    async def disconnect_response():
        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        response = JobStreamResponse(registry, registry.subscribe(job.job_id))
        await asyncio.wait_for(response(scope("2.3"), receive, send), 1)

    asyncio.run(disconnect_response())
    assert registry.subscribers == 0
    assert registry.active_id == job.job_id


def test_log_count_evicts_oldest_and_reports_dropped_lines_once():
    registry, job = registry_with_job()
    for index in range(257):
        registry.emit(job, {"type": "log", "message": f"Строка {index}"})
    assert len(job.events) == 250
    assert job.dropped == 7
    assert job.log_size == sum(map(len, job.events)) <= DEFAULT_LIMITS.log_bytes

    async def read_prefix():
        events = registry.events(job)
        summary = event_payload(await anext(events))
        first = event_payload(await anext(events))
        await events.aclose()
        return summary, first

    summary, first = asyncio.run(read_prefix())
    assert summary == {"type": "log", "message": "Пропущено старых сообщений: 7."}
    assert first == {"type": "log", "message": "Строка 7"}
    assert job.dropped == 0
    assert len(job.events) == 249
    assert job.log_size == sum(map(len, job.events))


def test_log_total_bytes_evict_independently_of_count():
    registry, job = registry_with_job(limits=replace(DEFAULT_LIMITS, log_bytes=1000))
    for index in range(3):
        registry.emit(job, {"type": "log", "message": str(index) + "я" * 300})
    assert len(job.events) == 1
    assert job.dropped == 2
    assert job.log_size == sum(map(len, job.events)) <= 1000
    assert json.loads(job.events[0])["message"].startswith("2")


@pytest.mark.parametrize("text", ["a" * 4097, "я" * 3000, "😀" * 2000])
def test_long_log_is_truncated_on_utf8_boundary_without_replacement_characters(text):
    registry, job = registry_with_job()
    registry.emit(job, {"type": "log", "message": text})
    message = json.loads(job.events[0])["message"]
    assert len(message.encode("utf-8")) <= 4096
    assert message.endswith("…")
    assert text.startswith(message[:-1])
    assert "\ufffd" not in message
    assert job.log_size == sum(map(len, job.events))


@pytest.mark.parametrize("length", [4095, 4096])
def test_log_at_or_below_line_limit_is_not_truncated(length):
    registry, job = registry_with_job()
    text = "a" * length
    registry.emit(job, {"type": "log", "message": text})
    assert json.loads(job.events[0])["message"] == text


def test_unsubscribed_log_flood_does_not_prevent_terminal_publication():
    registry, job = registry_with_job(limits=replace(DEFAULT_LIMITS, log_count=2, log_bytes=256))
    for index in range(1000):
        registry.emit(job, {"type": "log", "message": f"Сообщение {index}"})
    registry.emit(job, done())
    assert registry.subscribers == 0
    assert job.terminal_event is not None
    assert not job.events and job.log_size == job.dropped == 0
    assert registry.active_id is None


@pytest.mark.parametrize("oversize", [False, True])
def test_terminal_size_is_checked_before_publication(oversize):
    event = done()
    size = len(json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    registry, job = registry_with_job(limits=replace(DEFAULT_LIMITS, terminal_bytes=size - int(oversize)))
    if oversize:
        with pytest.raises(WebError) as failure:
            registry.emit(job, event)
        assert failure.value.code == "result_too_large"
        assert job.terminal_event is None and job.completed_at is None
        assert registry.active_id == job.job_id
    else:
        registry.emit(job, event)
        assert len(job.terminal_event) == size
        assert registry.active_id is None


@pytest.mark.parametrize("version", ["2.3", "2.4"])
@pytest.mark.parametrize("blocked_type", ["http.response.start", "http.response.body"])
def test_slow_send_timeout_frees_subscriber_in_both_asgi_paths(version, blocked_type):
    registry, job = registry_with_job(limits=replace(DEFAULT_LIMITS, send_timeout_seconds=0.01))

    async def slow_response():
        cancelled = asyncio.Event()

        async def send(message):
            if message["type"] == blocked_type:
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()

        response = JobStreamResponse(registry, registry.subscribe(job.job_id))
        await asyncio.wait_for(response(scope(version), pending_receive, send), 1)
        assert cancelled.is_set()

    asyncio.run(slow_response())
    assert registry.subscribers == 0
    assert job.completed_at is None and registry.active_id == job.job_id


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_slow_terminal_send_is_limited_by_remaining_ttl(version):
    clock = Clock()
    registry, job = registry_with_job(clock=clock)
    registry.emit(job, done())
    clock.now += 1800 - 0.01

    async def slow_response():
        cancelled = asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.body":
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()

        response = JobStreamResponse(registry, registry.subscribe(job.job_id))
        # Общий таймаут отправки равен 10 с; завершение до 1 с доказывает применение остатка TTL.
        await asyncio.wait_for(response(scope(version), pending_receive, send), 1)
        assert cancelled.is_set()

    asyncio.run(slow_response())
    assert registry.subscribers == 0
    clock.now += 0.02
    registry.cleanup()
    assert not registry.jobs and job.terminal_event is None


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_stream_admitted_before_expiry_stops_without_sending_after_expiry(version):
    clock = Clock()
    registry, job = registry_with_job(clock=clock)
    registry.emit(job, done())
    response = JobStreamResponse(registry, registry.subscribe(job.job_id))
    clock.now += 1800

    async def expired_response():
        async def send(message):
            pytest.fail("Истёкший итог не должен передаваться клиенту.")

        await response(scope(version), pending_receive, send)

    asyncio.run(expired_response())
    assert registry.subscribers == 0


def test_nested_transport_exception_group_is_handled_and_subscriber_released():
    registry, job = registry_with_job()

    async def grouped_failure():
        async def send(message):
            raise ExceptionGroup("Транспорт", [TimeoutError(), ExceptionGroup("Вложенная", [
                OSError("Соединение закрыто"), ClientDisconnect(),
            ])])

        response = JobStreamResponse(registry, registry.subscribe(job.job_id))
        await response(scope(), pending_receive, send)

    asyncio.run(grouped_failure())
    assert registry.subscribers == 0


def test_unexpected_exception_in_group_is_not_silenced_and_subscriber_released():
    registry, job = registry_with_job()

    async def grouped_failure():
        async def send(message):
            raise ExceptionGroup("Смешанная ошибка", [TimeoutError(), RuntimeError("Ошибка реализации")])

        response = JobStreamResponse(registry, registry.subscribe(job.job_id))
        await response(scope(), pending_receive, send)

    with pytest.raises(ExceptionGroup) as failure:
        asyncio.run(grouped_failure())
    leaves = exception_leaves(failure.value)
    assert len(leaves) == 1 and isinstance(leaves[0], RuntimeError)
    assert registry.subscribers == 0
