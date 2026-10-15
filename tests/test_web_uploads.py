from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from starlette.datastructures import Headers, UploadFile
from starlette.requests import Request
from starlette.responses import Response

from ocr_local.web import app as web_app
from ocr_local.web import request_limits
from ocr_local.web.errors import WebError
from ocr_local.web.limits import DEFAULT_LIMITS, MIB
from ocr_local.web.request_limits import RequestLimits
from ocr_local.web.uploads import TemporaryUploads, UploadParser, stage_image


BOUNDARY = b"upload-boundary"
FILE_HEADER = (b"--" + BOUNDARY + b'\r\nContent-Disposition: form-data; name="file"; '
               b'filename="image.png"\r\nContent-Type: image/png\r\n\r\n')
FILE_END = b"\r\n--" + BOUNDARY + b"--\r\n"


@pytest.fixture
def workspace(tmp_path):
    result = TemporaryUploads(tmp_path / "uploads", replace(DEFAULT_LIMITS, temp_bytes=128))
    result.prepare()
    return result


def snapshot(paths):
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def assert_clean(workspace, parsers):
    assert parsers
    handles = [handle for parser in parsers for handle in parser._files_to_close_on_error]
    assert handles and all(handle.closed for handle in handles)
    assert all(not path.exists() for parser in parsers for path in parser.paths)
    assert sorted(path.name for path in workspace.path.iterdir()) == [".owner"]


def request_scope():
    return {"type": "http", "method": "POST", "path": "/api/upload",
            "headers": [(b"content-type", b"multipart/form-data; boundary=" + BOUNDARY)]}


def assert_partial_written(parsers):
    assert len(parsers) == 1 and len(parsers[0].paths) == 1
    handle = parsers[0]._files_to_close_on_error[0]
    assert not handle.closed and handle.tell() == len(b"partial")


@pytest.fixture
def upload_handler(workspace, monkeypatch):
    parsers = []

    class ObservedParser(UploadParser):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            parsers.append(self)

    monkeypatch.setattr(web_app, "_workspace", workspace)
    monkeypatch.setattr(web_app, "_registry", SimpleNamespace(is_busy=lambda: False))
    monkeypatch.setattr(web_app, "UploadParser", ObservedParser)

    async def app(scope, receive, send):
        await web_app.upload_image(Request(scope, receive))
        await Response(status_code=204)(scope, receive, send)

    middleware = RequestLimits(app, limits=DEFAULT_LIMITS, is_busy=lambda: False,
                               logger=logging.getLogger(__name__))
    return workspace, parsers, middleware


def test_temporary_quota_defaults_to_52_mib():
    assert DEFAULT_LIMITS.temp_bytes == 52 * MIB


@pytest.mark.parametrize("extra,allowed", [(4, True), (5, True), (6, False)])
def test_temporary_quota_counts_form_and_staged_copy(workspace, extra, allowed):
    workspace.limits = replace(workspace.limits, temp_bytes=10)
    form = workspace.new_path("form", ".part")
    staged = workspace.new_path("image", ".png")
    form.write_bytes(b"abc")
    staged.write_bytes(b"de")
    before = snapshot([form, staged])

    if allowed:
        workspace.check_size(extra)
    else:
        with pytest.raises(WebError) as caught:
            workspace.check_size(extra)
        assert caught.value.status_code == 507
        assert caught.value.code == "storage_quota_exceeded"

    assert snapshot([form, staged]) == before


def test_cleanup_removes_only_owned_orphans(workspace):
    owned = [workspace.new_path("form", ".part"), workspace.new_path("image", ".webp")]
    foreign = [workspace.path / "notes.txt", workspace.path / "form-not-a-uuid.part",
               workspace.path / ("image-" + "a" * 32 + ".txt"),
               workspace.path.parent / ("form-" + "b" * 32 + ".part")]
    matching_directory = workspace.path / ("form-" + "c" * 32 + ".part")
    matching_directory.mkdir()
    foreign.append(matching_directory / "original.png")
    for path in owned + foreign:
        path.write_bytes(b"preserved unless owned orphan")
    protected = [workspace.path / ".owner", *foreign]
    before = snapshot(protected)

    workspace.cleanup()
    workspace.cleanup()

    assert all(not path.exists() for path in owned)
    assert matching_directory.is_dir()
    assert snapshot(protected) == before


@pytest.mark.parametrize("marker", [None, "another-program\n"])
def test_unowned_directory_is_not_adopted_or_cleaned(tmp_path, marker):
    workspace = TemporaryUploads(tmp_path / "uploads", DEFAULT_LIMITS)
    workspace.path.mkdir(parents=True)
    foreign = workspace.path / ("form-" + "a" * 32 + ".part")
    foreign.write_bytes(b"original")
    protected = [foreign]
    if marker is not None:
        marker_path = workspace.path / ".owner"
        marker_path.write_text(marker, encoding="utf-8")
        protected.append(marker_path)
    before = snapshot(protected)

    with pytest.raises(WebError) as caught:
        workspace.cleanup()

    assert caught.value.status_code == 507 and caught.value.code == "storage_unavailable"
    assert snapshot(protected) == before
    assert (workspace.path / ".owner").exists() is (marker is not None)


def test_cleanup_preserves_symlink_and_external_target(workspace, tmp_path):
    target = tmp_path / "original.png"
    target.write_bytes(b"user-original")
    link = workspace.path / ("image-" + "a" * 32 + ".png")
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"Создание символической ссылки недоступно: {exc}")
    before = snapshot([target])

    workspace.cleanup()

    assert link.is_symlink() and snapshot([target]) == before


def open_parser(workspace):
    async def unused_stream():
        yield b""

    parser = UploadParser(Headers({"content-type": "multipart/form-data; boundary=upload-boundary"}),
                          unused_stream(), workspace)
    parser.on_part_begin()
    field, value = b"Content-Disposition", b'form-data; name="file"; filename="image.png"'
    parser.on_header_field(field, 0, len(field))
    parser.on_header_value(value, 0, len(value))
    parser.on_header_end()
    parser.on_headers_finished()
    return parser


def test_parser_counts_all_pending_callbacks_before_file_write(workspace):
    workspace.limits = replace(workspace.limits, temp_bytes=8)
    parser = open_parser(workspace)
    try:
        parser.on_part_data(b"abcdef", 0, 6)
        parser.on_part_data(b"gh", 0, 2)
        assert parser.paths[0].stat().st_size == 0
        with pytest.raises(WebError) as caught:
            parser.on_part_data(b"i", 0, 1)
        assert caught.value.status_code == 507
        assert caught.value.code == "storage_quota_exceeded"
        assert sum(len(data) for _, data in parser._file_parts_to_write) == 8
    finally:
        parser.close()

    assert_clean(workspace, [parser])


def test_parser_counts_written_bytes_without_explicit_flush(workspace):
    workspace.limits = replace(workspace.limits, temp_bytes=8)
    parser = open_parser(workspace)
    try:
        handle = parser._files_to_close_on_error[0]
        handle.write(b"abcdef")
        parser.on_part_data(b"gh", 0, 2)
        with pytest.raises(WebError) as caught:
            parser.on_part_data(b"i", 0, 1)
        assert caught.value.status_code == 507
        assert caught.value.code == "storage_quota_exceeded"
    finally:
        parser.close()

    assert_clean(workspace, [parser])


@pytest.mark.parametrize("size,allowed", [(7, True), (8, True), (9, False)])
def test_real_multipart_counts_small_chunks_before_accepting_next_byte(workspace, size, allowed):
    workspace.limits = replace(workspace.limits, temp_bytes=8)
    consumed = []

    async def body():
        yield FILE_HEADER
        for index in range(size):
            consumed.append(index)
            yield b"x"
        yield FILE_END

    parser = UploadParser(Headers({"content-type": "multipart/form-data; boundary=upload-boundary"}),
                          body(), workspace)

    async def parse():
        try:
            if allowed:
                form = await parser.parse()
                assert parser.finished and await form["file"].read() == b"x" * size
            else:
                with pytest.raises(WebError) as caught:
                    await parser.parse()
                assert caught.value.status_code == 507
                assert caught.value.code == "storage_quota_exceeded"
                assert parser.paths[0].stat().st_size == 8
        finally:
            parser.close()

    asyncio.run(parse())
    assert len(consumed) == size
    assert_clean(workspace, [parser])


@pytest.mark.parametrize("extra,allowed", [(-1, False), (0, True), (1, True)])
def test_stage_image_respects_combined_form_and_copy_quota(workspace, extra, allowed):
    content = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(content, format="PNG")
    data = content.getvalue()
    form_path = workspace.new_path("form", ".part")
    form_path.write_bytes(data)
    workspace.limits = replace(workspace.limits, temp_bytes=2 * len(data) + extra)
    before = snapshot([form_path])
    file = UploadFile(io.BytesIO(data), filename="scan.PNG")

    async def stage():
        try:
            if allowed:
                staged, digest, suffix = await stage_image(file, workspace)
                assert staged.read_bytes() == data and suffix == ".png"
                assert digest == hashlib.sha256(data).hexdigest()
                staged.unlink()
            else:
                with pytest.raises(WebError) as caught:
                    await stage_image(file, workspace)
                assert caught.value.status_code == 507
                assert caught.value.code == "storage_quota_exceeded"
        finally:
            await file.close()

    asyncio.run(stage())
    assert file.file.closed and snapshot([form_path]) == before
    assert sorted(path.name for path in workspace.path.iterdir()) == [".owner", form_path.name]


@pytest.mark.parametrize("tail", [
    b"",
    b'\r\n--upload-boundary\r\nContent-Disposition: form-data; name="extra"\r\n\r\nx' + FILE_END,
    b"\r\n--upload-boundary\r\nBad Header\r\n\r\nx" + FILE_END,
], ids=["unfinished", "unexpected-field", "malformed-header"])
def test_malformed_multipart_after_partial_write_closes_and_removes_files(upload_handler, tail):
    workspace, parsers, middleware = upload_handler
    reads, sent = [], []

    async def receive():
        reads.append(True)
        if len(reads) == 1:
            return {"type": "http.request", "body": FILE_HEADER + b"partial", "more_body": True}
        assert len(reads) == 2
        assert_partial_written(parsers)
        return {"type": "http.request", "body": tail, "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(request_scope(), receive, send))

    assert sent[0]["status"] == 400
    payload = json.loads(sent[1]["body"])
    assert payload["code"] == "invalid_request"
    assert len(reads) == 2 and not middleware.upload_slot.locked()
    assert_clean(workspace, parsers)


def test_cancel_after_partial_write_closes_files_and_releases_slot(upload_handler):
    workspace, parsers, middleware = upload_handler
    reads = []

    async def scenario():
        blocked = asyncio.Event()

        async def receive():
            reads.append(True)
            if len(reads) == 1:
                return {"type": "http.request", "body": FILE_HEADER + b"partial", "more_body": True}
            assert_partial_written(parsers)
            blocked.set()
            await asyncio.Future()

        async def send(message):
            pytest.fail("Отменённая загрузка не должна отправлять ответ")

        task = asyncio.create_task(middleware(request_scope(), receive, send))
        await asyncio.wait_for(blocked.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert len(reads) == 2 and not middleware.upload_slot.locked()
    assert_clean(workspace, parsers)


@pytest.mark.parametrize("deadline", ["idle", "total"])
def test_timeout_after_partial_write_closes_files_and_releases_slot(upload_handler, monkeypatch, deadline):
    workspace, parsers, middleware = upload_handler
    reads, sent, cancelled = [], [], []
    now = [0.0]
    if deadline == "idle":
        middleware.limits = replace(DEFAULT_LIMITS, upload_idle_seconds=0.02)
    else:
        middleware.limits = replace(DEFAULT_LIMITS, upload_idle_seconds=4, upload_total_seconds=5)
        monkeypatch.setattr(request_limits, "time", SimpleNamespace(monotonic=lambda: now[0]))

    async def receive():
        reads.append(True)
        now[0] += 3
        if len(reads) == 1:
            return {"type": "http.request", "body": FILE_HEADER + b"partial", "more_body": True}
        assert len(reads) == 2
        assert_partial_written(parsers)
        if deadline == "total":
            return {"type": "http.request", "body": b"x", "more_body": True}
        try:
            await asyncio.Future()
        finally:
            cancelled.append(True)

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(request_scope(), receive, send))

    assert sent[0]["status"] == 408
    assert json.loads(sent[1]["body"])["code"] == "upload_timeout"
    assert cancelled == ([True] if deadline == "idle" else [])
    assert len(reads) == 2 and not middleware.upload_slot.locked()
    assert_clean(workspace, parsers)
