from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

import pytest
from starlette.requests import Request
from starlette.responses import Response

from ocr_local.web import request_limits
from ocr_local.web.limits import DEFAULT_LIMITS
from ocr_local.web.request_limits import RequestLimits


def scope(path="/api/upload", headers=()):
    return {"type": "http", "method": "POST", "path": path, "headers": list(headers)}


async def consume(scope, receive, send):
    async for _ in Request(scope, receive).stream():
        pass
    await Response(status_code=204)(scope, receive, send)


def middleware(app=consume, **overrides):
    return RequestLimits(app, limits=replace(DEFAULT_LIMITS, **overrides), is_busy=lambda: False,
                         logger=logging.getLogger("request-limits-test"))


def invoke(app, messages, request_scope=None):
    sent, reads = [], []

    async def receive():
        reads.append(True)
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    asyncio.run(app(request_scope or scope(), receive, send))
    return sent[0]["status"], len(reads)


@pytest.mark.parametrize("path,field", [("/api/upload", "total_body_bytes"), ("/api/recognize", "json_bytes")])
@pytest.mark.parametrize("size,status", [(7, 204), (8, 204), (9, 413)])
@pytest.mark.parametrize("declared", [False, True])
def test_counts_actual_bytes_with_absent_or_false_length(path, field, size, status, declared):
    app = middleware(**{field: 8})
    headers = [(b"content-length", b"1")] if declared else []
    messages = [{"type": "http.request", "body": b"x" * 4, "more_body": True},
                {"type": "http.request", "body": b"x" * (size - 4), "more_body": False}]
    assert invoke(app, messages, scope(path, headers)) == (status, 2)
    assert not app.upload_slot.locked()


@pytest.mark.parametrize("headers,status", [
    ([(b"content-length", b"9")], 413),
    ([(b"content-length", b"-1")], 400),
    ([(b"content-length", b"0"), (b"content-length", b"0")], 400),
])
def test_invalid_declared_length_does_not_read_body(headers, status):
    app = middleware(total_body_bytes=8)
    assert invoke(app, [], scope(headers=headers)) == (status, 0)
    assert not app.upload_slot.locked()


def test_second_upload_is_rejected_without_consuming_body():
    app = middleware()
    app.upload_slot.acquire()
    try:
        assert invoke(app, []) == (429, 0)
        assert app.upload_slot.locked()
    finally:
        app.upload_slot.release()
    assert invoke(app, [{"type": "http.request", "body": b""}])[0] == 204


@pytest.mark.parametrize("advance,chunks", [(11, [b"", b""]), (6, [b"x", b"x", b"x"])])
def test_idle_and_total_deadlines_use_monotonic_time(monkeypatch, advance, chunks):
    now = [0.0]
    monkeypatch.setattr(request_limits.time, "monotonic", lambda: now[0])
    # Подменяем только ожидание: управляемое время не должно менять часы event loop.
    async def immediate(awaitable, *, timeout):
        assert timeout > 0
        return await awaitable

    monkeypatch.setattr(request_limits.asyncio, "wait_for", immediate)
    app = middleware(upload_total_seconds=12, upload_idle_seconds=10)
    sent = []

    async def receive():
        now[0] += advance
        return {"type": "http.request", "body": chunks.pop(0), "more_body": True}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope(), receive, send))
    assert sent[0]["status"] == 408
    assert not app.upload_slot.locked()


def test_silent_receive_is_cancelled_and_slot_released():
    app = middleware(upload_idle_seconds=0.01)
    cancelled, sent = [], []

    async def receive():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope(), receive, send))
    assert sent[0]["status"] == 408 and cancelled == [True]
    assert not app.upload_slot.locked()


@pytest.mark.parametrize("failure", [asyncio.CancelledError, OSError])
def test_cancellation_and_receive_failure_release_upload_slot(failure):
    app = middleware()

    async def receive():
        raise failure()

    async def send(message):
        pytest.fail("Ответ не ожидается после отмены транспорта")

    with pytest.raises(failure):
        asyncio.run(app(scope(), receive, send))
    assert not app.upload_slot.locked()
