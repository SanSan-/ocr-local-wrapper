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
    return await collect_subscription_response(registry, registry.subscribe(job.job_id), version=version)


async def collect_subscription_response(registry, subscription, *, version="2.4"):
    messages = []

    async def send(message):
        messages.append(message)

    response = JobStreamResponse(registry, subscription)
    await response(scope(version), pending_receive, send)
    return b"".join(message.get("body", b"") for message in messages)


def response_events(body):
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b"data: ")]


def assert_log_accounting(job, subscriptions):
    sending = [item.in_flight for item in subscriptions if item.in_flight is not None]
    assert job.log_count == len(job.events) + len(sending)
    assert job.log_size == sum(map(len, job.events)) + sum(map(len, sending))


async def cancel_stream(task):
    if not task.done():
        task.cancel()
    try:
        await task
    except BaseException as error:
        assert all(isinstance(leaf, asyncio.CancelledError) for leaf in exception_leaves(error))


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
        subscription = registry.subscribe(job.job_id)
        events = registry.events(subscription)
        try:
            summary = event_payload(await anext(events))
            assert_log_accounting(job, [subscription])
            assert job.log_count <= DEFAULT_LIMITS.log_count
            registry.acknowledge(subscription)
            first = event_payload(await anext(events))
            registry.acknowledge(subscription)
            return summary, first
        finally:
            await events.aclose()
            registry.unsubscribe(subscription)

    summary, first = asyncio.run(read_prefix())
    # Само уведомление занимает место: при полной очереди вытесняется ещё одна строка.
    assert summary == {"type": "log", "message": "Пропущено старых сообщений: 8."}
    assert first == {"type": "log", "message": "Строка 8"}
    assert job.dropped == 0
    assert len(job.events) == 248
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


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_nested_transport_exception_group_is_handled_and_subscriber_released(version):
    registry, job = registry_with_job()
    subscription = registry.subscribe(job.job_id)
    registry.emit(job, {"type": "log", "message": "Строка до отказа транспорта."})
    registry.emit(job, done())

    async def grouped_failure():
        async def send(message):
            if not message.get("body", b"").startswith(b"data: "):
                return
            assert subscription.in_flight is not None
            raise ExceptionGroup("Транспорт", [TimeoutError(), ExceptionGroup("Вложенная", [
                OSError("Соединение закрыто"), ClientDisconnect(),
            ])])

        response = JobStreamResponse(registry, subscription)
        await response(scope(version), pending_receive, send)

    asyncio.run(grouped_failure())
    assert registry.subscribers == job.readers == job.log_count == job.log_size == 0
    assert subscription.closed and subscription.in_flight is None


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_unexpected_exception_in_group_is_not_silenced_and_subscriber_released(version):
    registry, job = registry_with_job()
    subscription = registry.subscribe(job.job_id)
    registry.emit(job, {"type": "log", "message": "Строка до непредусмотренной ошибки."})
    registry.emit(job, done())

    async def grouped_failure():
        async def send(message):
            if not message.get("body", b"").startswith(b"data: "):
                return
            assert subscription.in_flight is not None
            raise ExceptionGroup("Смешанная ошибка", [TimeoutError(), RuntimeError("Ошибка реализации")])

        response = JobStreamResponse(registry, subscription)
        await response(scope(version), pending_receive, send)

    with pytest.raises(ExceptionGroup) as failure:
        asyncio.run(grouped_failure())
    leaves = exception_leaves(failure.value)
    assert len(leaves) == 1 and isinstance(leaves[0], RuntimeError)
    assert registry.subscribers == job.readers == job.log_count == job.log_size == 0
    assert subscription.closed and subscription.in_flight is None


@pytest.mark.parametrize(("status", "version"), [
    ("ok", "2.4"), ("error", "2.4"), ("ok", "2.3"), ("error", "2.3"),
], ids=["ok-asgi24", "error-asgi24", "ok-asgi23", "error-asgi23"])
def test_pending_logs_precede_done_after_keepalive(status, version):
    registry, job = registry_with_job()
    expected_logs = [
        {"type": "log", "message": "Первая ожидающая строка."},
        {"type": "log", "message": "Вторая ожидающая строка."},
    ]
    terminal = done(status)
    delivered = []

    async def controlled_response():
        waiting = asyncio.Event()
        resume = asyncio.Event()

        async def send(message):
            body = message.get("body", b"")
            if body == b": keep-alive\n\n" and not waiting.is_set():
                waiting.set()
                await resume.wait()
            if body.startswith(b"data: "):
                delivered.append(event_payload(body))

        response = JobStreamResponse(registry, registry.subscribe(job.job_id))
        task = asyncio.create_task(response(scope(version), pending_receive, send))
        try:
            await asyncio.wait_for(waiting.wait(), 1)
            for event in expected_logs:
                registry.emit(job, event)
            registry.emit(job, terminal)
        finally:
            resume.set()
        await asyncio.wait_for(task, 1)

    asyncio.run(controlled_response())
    assert delivered == [*expected_logs, terminal]
    assert registry.subscribers == 0
    assert not job.events and job.log_size == 0


@pytest.mark.parametrize("version", ["2.3", "2.4"])
@pytest.mark.parametrize("status", ["ok", "error"])
def test_early_subscription_with_empty_queue_receives_only_done(version, status):
    registry, job = registry_with_job()
    subscription = registry.subscribe(job.job_id)
    registry.emit(job, done(status))

    body = asyncio.run(collect_subscription_response(registry, subscription, version=version))

    assert response_events(body) == [done(status)]
    assert subscription.closed and subscription.in_flight is None
    assert job.readers == job.log_count == job.log_size == registry.subscribers == 0


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_two_early_streams_share_logs_and_fast_done_does_not_wait(version):
    registry, job = registry_with_job()
    slow = registry.subscribe(job.job_id)
    fast = registry.subscribe(job.job_id)
    first = {"type": "log", "message": "Строка медленного потока."}
    second = {"type": "log", "message": "Строка быстрого потока."}
    registry.emit(job, first)
    registry.emit(job, second)
    registry.emit(job, done())
    original_terminal, original_completion = job.terminal_event, job.completed_at
    slow_delivered = []

    async def run_streams():
        selected = asyncio.Event()
        release = asyncio.Event()

        async def send_slow(message):
            body = message.get("body", b"")
            if not body.startswith(b"data: "):
                return
            event = event_payload(body)
            if event["type"] == "log":
                selected.set()
                await release.wait()
            slow_delivered.append(event)

        task = asyncio.create_task(JobStreamResponse(registry, slow)(scope(version), pending_receive, send_slow))
        try:
            await asyncio.wait_for(selected.wait(), 1)
            assert slow.in_flight is not None
            assert_log_accounting(job, [slow, fast])
            assert job.log_count == 2
            body = await asyncio.wait_for(collect_subscription_response(registry, fast, version=version), 1)
            assert response_events(body) == [second, done()]
            assert not task.done() and slow_delivered == []
            assert job.readers == registry.subscribers == 1
            assert job.log_count == 1 and not job.events
            assert_log_accounting(job, [slow, fast])
        finally:
            release.set()
            await asyncio.wait_for(task, 1)

    asyncio.run(run_streams())
    assert slow_delivered == [first, done()]
    assert job.terminal_event == original_terminal and job.completed_at == original_completion
    assert job.readers == job.log_count == job.log_size == registry.subscribers == 0
    assert slow.closed and fast.closed


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_late_stream_neither_steals_nor_retains_early_tail(version):
    registry, job = registry_with_job()
    early = registry.subscribe(job.job_id)
    for text in ("Первая строка.", "Вторая строка."):
        registry.emit(job, {"type": "log", "message": text})
    registry.emit(job, done())
    original_queue = list(job.events)

    async def late_streams():
        body = await collect_response(registry, job, version=version)
        assert response_events(body) == [done()]
        assert list(job.events) == original_queue and job.readers == 1

        late = registry.subscribe(job.job_id)
        assert not late.drains_logs
        selected, release = asyncio.Event(), asyncio.Event()
        delivered = []

        async def send(message):
            body = message.get("body", b"")
            if body.startswith(b"data: "):
                selected.set()
                await release.wait()
                delivered.append(event_payload(body))

        task = asyncio.create_task(JobStreamResponse(registry, late)(scope(version), pending_receive, send))
        try:
            await asyncio.wait_for(selected.wait(), 1)
            assert list(job.events) == original_queue
            registry.unsubscribe(early)
            registry.unsubscribe(early)
            assert early.closed and job.readers == 0
            assert not job.events and job.log_count == job.log_size == job.dropped == 0
            assert registry.subscribers == 1 and not late.closed
        finally:
            release.set()
            await asyncio.wait_for(task, 1)
        assert delivered == [done()]

    asyncio.run(late_streams())
    assert registry.subscribers == 0
    assert job.terminal_event is not None


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_closing_one_early_reader_preserves_tail_for_other(version):
    registry, job = registry_with_job()
    first = registry.subscribe(job.job_id)
    second = registry.subscribe(job.job_id)
    event = {"type": "log", "message": "Оставшаяся строка."}
    registry.emit(job, event)
    registry.emit(job, done("error"))

    registry.unsubscribe(first)
    registry.unsubscribe(first)
    assert job.readers == registry.subscribers == 1
    assert job.log_count == len(job.events) == 1
    body = asyncio.run(collect_subscription_response(registry, second, version=version))

    assert response_events(body) == [event, done("error")]
    assert job.readers == job.log_count == job.log_size == registry.subscribers == 0


def test_other_job_subscription_does_not_retain_unsubscribed_completed_log():
    registry, old_job = registry_with_job()
    old_subscription = registry.subscribe(old_job.job_id)
    registry.emit(old_job, done())
    job = registry.add(None, None)
    registry.emit(job, {"type": "log", "message": "Журнал другой задачи."})

    registry.emit(job, done())

    assert registry.subscribers == old_job.readers == 1
    assert job.readers == job.log_count == job.log_size == job.dropped == 0
    assert not job.events and job.terminal_event is not None
    registry.unsubscribe(old_subscription)
    assert registry.subscribers == 0


def test_last_unsubscribe_during_active_job_preserves_bounded_log():
    registry, job = registry_with_job()
    subscription = registry.subscribe(job.job_id)
    event = {"type": "log", "message": "Строка активной задачи."}
    registry.emit(job, event)

    registry.unsubscribe(subscription)
    registry.unsubscribe(subscription)

    assert job.readers == registry.subscribers == 0
    assert len(job.events) == job.log_count == 1
    assert job.completed_at is None and registry.active_id == job.job_id
    replacement = registry.subscribe(job.job_id)
    registry.emit(job, done())
    assert response_events(asyncio.run(collect_subscription_response(registry, replacement))) == [event, done()]


@pytest.mark.parametrize("subscribe_first", [True, False])
def test_subscription_registration_has_unambiguous_completion_order(subscribe_first):
    registry, job = registry_with_job()
    event = {"type": "log", "message": "Строка до завершения."}
    registry.emit(job, event)
    if subscribe_first:
        subscription = registry.subscribe(job.job_id)
        registry.emit(job, done())
    else:
        registry.emit(job, done())
        subscription = registry.subscribe(job.job_id)

    assert subscription.drains_logs is subscribe_first
    assert job.readers == int(subscribe_first)
    expected = [event, done()] if subscribe_first else [done()]
    assert response_events(asyncio.run(collect_subscription_response(registry, subscription))) == expected


def test_concurrent_registration_and_completion_remain_consistent():
    registry, job = registry_with_job()
    registry.emit(job, {"type": "log", "message": "Строка на границе завершения."})
    barrier = threading.Barrier(2, timeout=2)
    subscriptions, failures = [], []

    def register():
        try:
            barrier.wait()
            subscriptions.append(registry.subscribe(job.job_id))
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=register)
    thread.start()
    try:
        barrier.wait()
        registry.emit(job, done())
    finally:
        thread.join(timeout=2)
    assert not thread.is_alive() and not failures
    assert len(subscriptions) == registry.subscribers == 1
    subscription = subscriptions[0]
    assert job.readers == int(subscription.drains_logs)
    assert bool(job.events) is subscription.drains_logs
    registry.unsubscribe(subscription)
    assert job.readers == registry.subscribers == job.log_count == job.log_size == 0


@pytest.mark.parametrize("version", ["2.3", "2.4"])
@pytest.mark.parametrize("capacity", [1, 2])
def test_blocked_send_counts_toward_quota_and_overflow_precedes_done(version, capacity):
    limits = replace(DEFAULT_LIMITS, log_count=capacity, log_bytes=256)
    registry, job = registry_with_job(limits=limits)
    subscription = registry.subscribe(job.job_id)
    first = {"type": "log", "message": "Отправляемая строка."}
    second = {"type": "log", "message": "Вытесненная строка."}
    third = {"type": "log", "message": "Последняя строка."}
    registry.emit(job, first)
    delivered = []

    async def controlled_send():
        selected, release = asyncio.Event(), asyncio.Event()

        async def send(message):
            body = message.get("body", b"")
            if not body.startswith(b"data: "):
                return
            event = event_payload(body)
            if event == first:
                selected.set()
                await release.wait()
            assert_log_accounting(job, [subscription])
            assert job.log_count <= capacity and job.log_size <= limits.log_bytes
            delivered.append(event)

        task = asyncio.create_task(JobStreamResponse(registry, subscription)(scope(version), pending_receive, send))
        try:
            await asyncio.wait_for(selected.wait(), 1)
            in_flight = subscription.in_flight
            assert in_flight is not None and job.log_count == 1
            assert job.log_size == len(in_flight)
            registry.emit(job, second)
            registry.emit(job, third)
            assert subscription.in_flight == in_flight
            assert_log_accounting(job, [subscription])
            assert job.log_count == capacity and job.dropped == 3 - capacity
            registry.emit(job, done())
            assert registry.active_id is None
            assert subscription.in_flight == in_flight
        finally:
            release.set()
            await asyncio.wait_for(task, 1)

    asyncio.run(controlled_send())
    notice = {"type": "log", "message": f"Пропущено старых сообщений: {3 - capacity}."}
    expected = [first, notice, third, done()] if capacity == 2 else [first, notice, done()]
    assert delivered == expected
    assert job.log_count == job.log_size == job.dropped == job.readers == registry.subscribers == 0
    assert subscription.in_flight is None and subscription.closed


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_byte_quota_includes_blocked_log_and_notice(version):
    limits = replace(DEFAULT_LIMITS, log_bytes=200)
    registry, job = registry_with_job(limits=limits)
    subscription = registry.subscribe(job.job_id)
    first = {"type": "log", "message": "я" * 80}
    registry.emit(job, first)
    delivered = []

    async def controlled_response():
        selected, release = asyncio.Event(), asyncio.Event()

        async def send(message):
            body = message.get("body", b"")
            if not body.startswith(b"data: "):
                return
            event = event_payload(body)
            if event == first:
                selected.set()
                await release.wait()
            assert_log_accounting(job, [subscription])
            assert job.log_size <= limits.log_bytes
            if event["type"] == "log":
                assert subscription.in_flight is not None and job.log_count == 1
            else:
                assert subscription.in_flight is None and job.log_count == job.log_size == 0
            delivered.append(event)

        task = asyncio.create_task(JobStreamResponse(registry, subscription)(scope(version), pending_receive, send))
        try:
            await asyncio.wait_for(selected.wait(), 1)
            registry.emit(job, {"type": "log", "message": "Б"})
            assert not job.events and job.dropped == 1
            assert_log_accounting(job, [subscription])
            registry.emit(job, done())
        finally:
            release.set()
            await asyncio.wait_for(task, 1)

    asyncio.run(controlled_response())
    assert delivered == [first, {"type": "log", "message": "Пропущено старых сообщений: 1."}, done()]
    assert registry.subscribers == job.readers == job.log_count == job.log_size == 0


@pytest.mark.parametrize("version", ["2.3", "2.4"])
@pytest.mark.parametrize("failure", ["cancel", "timeout", "disconnect", "transport"])
def test_interrupted_log_send_releases_only_its_reader_and_in_flight(version, failure):
    limits = replace(DEFAULT_LIMITS, send_timeout_seconds=0.01 if failure == "timeout" else 10)
    registry, job = registry_with_job(limits=limits)
    interrupted = registry.subscribe(job.job_id)
    remaining = registry.subscribe(job.job_id)
    first = {"type": "log", "message": "Прерванная отправка."}
    second = {"type": "log", "message": "Оставшийся хвост."}
    registry.emit(job, first)
    registry.emit(job, second)
    registry.emit(job, done("error"))
    original_terminal, original_completion = job.terminal_event, job.completed_at

    async def interrupted_response():
        selected = asyncio.Event()

        async def receive():
            if failure == "disconnect" and version == "2.3":
                await selected.wait()
                return {"type": "http.disconnect"}
            return await pending_receive()

        async def send(message):
            if not message.get("body", b"").startswith(b"data: "):
                return
            assert event_payload(message["body"]) == first
            assert interrupted.in_flight is not None
            assert_log_accounting(job, [interrupted, remaining])
            selected.set()
            if failure == "transport":
                raise OSError("Соединение потеряно во время отправки строки.")
            if failure == "disconnect" and version == "2.4":
                raise ClientDisconnect()
            await asyncio.Future()

        task = asyncio.create_task(JobStreamResponse(registry, interrupted)(scope(version), receive, send))
        await asyncio.wait_for(selected.wait(), 1)
        if failure == "cancel":
            await asyncio.wait_for(cancel_stream(task), 1)
        else:
            await asyncio.wait_for(task, 1)
        registry.unsubscribe(interrupted)
        registry.acknowledge(interrupted)
        assert interrupted.closed and interrupted.in_flight is None
        assert registry.subscribers == job.readers == 1
        assert len(job.events) == job.log_count == 1
        assert_log_accounting(job, [interrupted, remaining])
        body = await collect_subscription_response(registry, remaining, version=version)
        assert response_events(body) == [second, done("error")]

    asyncio.run(interrupted_response())
    assert job.terminal_event == original_terminal and job.completed_at == original_completion
    assert registry.subscribers == job.readers == job.log_count == job.log_size == 0


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_expiry_during_log_send_releases_state_without_double_unsubscribe(version):
    clock = Clock()
    registry, job = registry_with_job(clock=clock)
    subscription = registry.subscribe(job.job_id)
    registry.emit(job, {"type": "log", "message": "Строка у границы срока."})
    registry.emit(job, {"type": "log", "message": "Неотправленный хвост."})
    registry.emit(job, done())
    completed_at = job.completed_at
    clock.now = completed_at + DEFAULT_LIMITS.job_ttl_seconds - 0.01
    selected = []

    async def expires_while_sending():
        async def send(message):
            if not message.get("body", b"").startswith(b"data: "):
                return
            selected.append(event_payload(message["body"]))
            assert subscription.in_flight is not None
            clock.now += 0.02
            registry.cleanup()
            assert job.job_id not in registry.jobs and job.terminal_event is None
            assert subscription.closed and subscription.in_flight is None
            assert job.log_count == job.log_size == job.readers == registry.subscribers == 0
            await asyncio.Future()

        await asyncio.wait_for(JobStreamResponse(registry, subscription)(scope(version), pending_receive, send), 1)

    asyncio.run(expires_while_sending())
    assert len(selected) == 1 and selected[0]["type"] == "log"
    registry.unsubscribe(subscription)
    registry.acknowledge(subscription)
    assert registry.subscribers == job.readers == 0
    assert job.completed_at == completed_at
    with pytest.raises(WebError) as failure:
        registry.subscribe(job.job_id)
    assert (failure.value.status_code, failure.value.code) == (404, "job_not_found")


def test_response_constructor_failure_releases_subscription(monkeypatch):
    registry, job = registry_with_job()
    subscription = registry.subscribe(job.job_id)

    def fail(*args, **kwargs):
        raise RuntimeError("Не удалось создать ответ.")

    monkeypatch.setattr("starlette.responses.StreamingResponse.__init__", fail)
    with pytest.raises(RuntimeError, match="Не удалось создать ответ"):
        JobStreamResponse(registry, subscription)

    assert subscription.closed and subscription.in_flight is None
    assert registry.subscribers == job.readers == 0
    registry.unsubscribe(subscription)
    assert registry.subscribers == job.readers == 0
    assert registry.active_id == job.job_id


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_iterator_close_failure_does_not_leak_subscription(version):
    registry, job = registry_with_job()
    subscription = registry.subscribe(job.job_id)
    registry.emit(job, {"type": "log", "message": "Строка перед закрытием."})
    registry.emit(job, done())
    response = JobStreamResponse(registry, subscription)
    original_iterator = response.body_iterator

    class FailingClose:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return await anext(original_iterator)

        async def aclose(self):
            await original_iterator.aclose()
            raise RuntimeError("Ошибка закрытия итератора.")

    response.body_iterator = FailingClose()

    async def close_fails():
        async def send(message):
            pass

        await response(scope(version), pending_receive, send)

    with pytest.raises(RuntimeError, match="Ошибка закрытия итератора"):
        asyncio.run(close_fails())
    assert subscription.closed and subscription.in_flight is None
    assert job.readers == job.log_count == job.log_size == registry.subscribers == 0
    registry.unsubscribe(subscription)
    assert registry.subscribers == 0


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_drop_notice_waits_for_budget_occupied_by_other_stream(version):
    limits = replace(DEFAULT_LIMITS, log_count=1, log_bytes=256)
    registry, job = registry_with_job(limits=limits)
    first_reader = registry.subscribe(job.job_id)
    waiting_reader = registry.subscribe(job.job_id)
    event = {"type": "log", "message": "Единственная отправляемая строка."}
    registry.emit(job, event)
    first_delivered, waiting_delivered = [], []

    async def controlled_streams():
        first_selected, waiting = asyncio.Event(), asyncio.Event()
        release_first, release_waiting = asyncio.Event(), asyncio.Event()

        async def first_send(message):
            body = message.get("body", b"")
            if not body.startswith(b"data: "):
                return
            payload = event_payload(body)
            if payload == event:
                first_selected.set()
                await release_first.wait()
            assert_log_accounting(job, [first_reader, waiting_reader])
            assert job.log_count <= 1 and job.log_size <= limits.log_bytes
            first_delivered.append(payload)

        async def waiting_send(message):
            body = message.get("body", b"")
            if body == b": keep-alive\n\n":
                waiting.set()
                await release_waiting.wait()
            elif body.startswith(b"data: "):
                waiting_delivered.append(event_payload(body))

        first_task = asyncio.create_task(JobStreamResponse(registry, first_reader)(
            scope(version), pending_receive, first_send))
        waiting_task = None
        try:
            await asyncio.wait_for(first_selected.wait(), 1)
            registry.emit(job, {"type": "log", "message": "Отброшенная строка."})
            registry.emit(job, done())
            waiting_task = asyncio.create_task(JobStreamResponse(registry, waiting_reader)(
                scope(version), pending_receive, waiting_send))
            await asyncio.wait_for(waiting.wait(), 1)
            assert waiting_delivered == [] and not waiting_task.done()
            assert job.dropped == 1 and job.log_count == 1 and not job.events
            assert waiting_reader.in_flight is None
            assert_log_accounting(job, [first_reader, waiting_reader])
            release_first.set()
            await asyncio.wait_for(first_task, 1)
        finally:
            release_first.set()
            release_waiting.set()
            await asyncio.wait_for(first_task, 1)
            if waiting_task is not None:
                await asyncio.wait_for(waiting_task, 1)

    asyncio.run(controlled_streams())
    assert first_delivered == [event, {"type": "log", "message": "Пропущено старых сообщений: 1."}, done()]
    assert waiting_delivered == [done()]
    assert registry.subscribers == job.readers == job.log_count == job.log_size == job.dropped == 0
