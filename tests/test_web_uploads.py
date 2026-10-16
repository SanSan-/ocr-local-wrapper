from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, PngImagePlugin
from starlette.datastructures import Headers, UploadFile
from starlette.requests import Request
from starlette.responses import Response

from ocr_local.web import app as web_app
from ocr_local.web import request_limits
from ocr_local.web import uploads as web_uploads
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


def staging_input(workspace):
    content = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(content, format="PNG")
    data = content.getvalue()
    form_path = workspace.new_path("form", ".part")
    form_path.write_bytes(data)
    foreign = [workspace.path / "notes.txt", workspace.path.parent / "original.png",
               workspace.path.parent / "result.txt", workspace.path.parent / "cache.json"]
    for path in foreign:
        path.write_bytes(b"existing user data")
    workspace.limits = replace(workspace.limits, temp_bytes=2 * len(data))
    protected = [workspace.path / ".owner", form_path, *foreign]
    return UploadFile(form_path.open("rb"), filename="scan.PNG"), data, snapshot(protected)


def assert_staging_preserved(workspace, before):
    assert snapshot(before) == before
    expected = {path.name for path in before if path.parent == workspace.path}
    assert {path.name for path in workspace.path.iterdir()} == expected


class StageGate:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def wait(self):
        assert threading.get_ident() != self.loop_thread
        self.loop.call_soon_threadsafe(self.entered.set)
        if not self.release.wait(timeout=5):
            raise TimeoutError("Тест не освободил подготовку изображения")
        self.finished.set()


class GatedWriter:
    def __init__(self, handle, gate):
        self.handle, self.gate = handle, gate

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.handle.__exit__(*args)

    def write(self, chunk):
        self.gate.wait()
        return self.handle.write(chunk)

    def flush(self):
        self.handle.flush()


def install_stage_gate(monkeypatch, phase, gate):
    if phase == "write":
        original_open = Path.open

        def gated_open(path, mode="r", *args, **kwargs):
            handle = original_open(path, mode, *args, **kwargs)
            return GatedWriter(handle, gate) if mode == "xb" else handle

        monkeypatch.setattr(Path, "open", gated_open)
    else:
        original_verify = PngImagePlugin.PngImageFile.verify

        def gated_verify(image):
            gate.wait()
            original_verify(image)

        monkeypatch.setattr(PngImagePlugin.PngImageFile, "verify", gated_verify)


async def loop_checkpoint():
    reached = asyncio.Event()
    asyncio.get_running_loop().call_soon(reached.set)
    await reached.wait()


@pytest.mark.parametrize("worker_fails", [False, True])
def test_worker_wait_reraises_cancellation_after_thread_finishes(worker_fails):
    async def scenario():
        gate = StageGate()
        failure = OSError("Ошибка рабочего потока")

        def worker():
            gate.wait()
            if worker_fails:
                raise failure
            return "готово"

        future = asyncio.get_running_loop().run_in_executor(None, worker)
        task = asyncio.create_task(web_uploads._wait_for_worker(future))
        try:
            await asyncio.wait_for(gate.entered.wait(), timeout=2)
            for number in range(1000):
                task.cancel(f"отмена {number}")
                await loop_checkpoint()
                assert not task.done() and not future.done()
        finally:
            gate.release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(task, timeout=2)
        assert caught.value.args == ("отмена 999",)
        assert future.done() and not future.cancelled() and gate.finished.is_set()
        if worker_fails:
            assert future.exception() is failure and caught.value.__cause__ is failure
        else:
            assert future.result() == "готово"

    asyncio.run(scenario())


@pytest.mark.parametrize("worker_fails", [False, True])
def test_worker_wait_accepts_already_completed_future(worker_fails):
    async def scenario():
        future = asyncio.get_running_loop().create_future()
        if worker_fails:
            failure = OSError("Ошибка завершённого рабочего потока")
            future.set_exception(failure)
            with pytest.raises(OSError) as caught:
                await web_uploads._wait_for_worker(future)
            assert caught.value is failure
        else:
            future.set_result("готово")
            assert await web_uploads._wait_for_worker(future) == "готово"

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["write", "verify"])
def test_slow_staging_keeps_event_loop_responsive(workspace, monkeypatch, phase):
    file, data, before = staging_input(workspace)

    async def scenario():
        gate = StageGate()
        install_stage_gate(monkeypatch, phase, gate)
        task = asyncio.create_task(stage_image(file, workspace))
        try:
            await asyncio.wait_for(gate.entered.wait(), timeout=2)
            await loop_checkpoint()
            assert not task.done() and not gate.finished.is_set()
        finally:
            gate.release.set()
        staged, digest, suffix = await asyncio.wait_for(task, timeout=2)
        assert staged.read_bytes() == data
        assert digest == hashlib.sha256(data).hexdigest() and suffix == ".png"
        assert gate.finished.is_set() and not file.file.closed
        staged.unlink()

    try:
        asyncio.run(scenario())
    finally:
        file.file.close()
    assert_staging_preserved(workspace, before)


@pytest.mark.parametrize("phase", ["write", "verify"])
@pytest.mark.parametrize("cancellations", [1, 2])
def test_staging_cancellation_joins_worker_before_cleanup(workspace, monkeypatch, phase, cancellations):
    file, _, before = staging_input(workspace)
    cleanup_threads = []
    original_unlink = Path.unlink

    async def scenario():
        gate = StageGate()
        install_stage_gate(monkeypatch, phase, gate)

        def checked_unlink(path, *args, **kwargs):
            if path.name.startswith("image-"):
                assert gate.finished.is_set()
                cleanup_threads.append(threading.get_ident())
                assert cleanup_threads[-1] != gate.loop_thread
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", checked_unlink)
        task = asyncio.create_task(stage_image(file, workspace))
        try:
            await asyncio.wait_for(gate.entered.wait(), timeout=2)
            for _ in range(cancellations):
                task.cancel()
                await loop_checkpoint()
                assert not task.done() and not file.file.closed and not cleanup_threads
                assert list(workspace.path.glob("image-*.png"))
        finally:
            gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert gate.finished.is_set() and cleanup_threads

    try:
        asyncio.run(scenario())
    finally:
        file.file.close()
    assert_staging_preserved(workspace, before)


def test_cancel_all_tasks_waits_for_staging_thread_before_cleanup(workspace, monkeypatch):
    file, _, before = staging_input(workspace)
    original_open, original_unlink = Path.open, Path.unlink
    writers, premature_cleanup = [], []

    async def scenario():
        gate = StageGate()

        loop = asyncio.get_running_loop()
        original_executor = loop.run_in_executor
        executor_futures = []

        def observed_executor(executor, function, *args):
            future = original_executor(executor, function, *args)
            executor_futures.append(future)
            return future

        def gated_open(path, mode="r", *args, **kwargs):
            handle = original_open(path, mode, *args, **kwargs)
            if mode != "xb":
                return handle
            writers.append(handle)
            return GatedWriter(handle, gate)

        def checked_unlink(path, *args, **kwargs):
            if path.name.startswith("image-") and any(not writer.closed for writer in writers):
                premature_cleanup.append(path)
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", gated_open)
        monkeypatch.setattr(Path, "unlink", checked_unlink)
        monkeypatch.setattr(loop, "run_in_executor", observed_executor)
        staging_task = asyncio.create_task(stage_image(file, workspace))
        try:
            await asyncio.wait_for(gate.entered.wait(), timeout=2)
            assert len(executor_futures) == 1 and writers and all(not writer.closed for writer in writers)
            worker_future = executor_futures[0]
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()
            assert not worker_future.cancelled() and not worker_future.done()
            await loop_checkpoint()
            await loop_checkpoint()
            assert not staging_task.done() and not premature_cleanup
            assert not worker_future.done() and all(not writer.closed for writer in writers)
        finally:
            gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(staging_task, timeout=2)
        assert worker_future.done() and not worker_future.cancelled()
        assert all(writer.closed for writer in writers) and not premature_cleanup

    try:
        asyncio.run(scenario())
    finally:
        file.file.close()
    assert_staging_preserved(workspace, before)


def test_cancellation_after_worker_success_removes_unclaimed_image(workspace, monkeypatch):
    file, _, before = staging_input(workspace)
    original_run = web_uploads._ImageStaging.run
    original_unlink = Path.unlink
    completed = threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        gate = StageGate()

        def gated_unlink(path, *args, **kwargs):
            if path.name.startswith("image-"):
                assert completed.is_set()
                gate.wait()
            return original_unlink(path, *args, **kwargs)

        def cancel_after_success(staging):
            result = original_run(staging)
            completed.set()
            loop.call_soon_threadsafe(task.cancel)
            return result

        monkeypatch.setattr(web_uploads._ImageStaging, "run", cancel_after_success)
        monkeypatch.setattr(Path, "unlink", gated_unlink)
        task = asyncio.create_task(stage_image(file, workspace))
        try:
            await asyncio.wait_for(gate.entered.wait(), timeout=2)
            task.cancel()
            await loop_checkpoint()
            assert not task.done() and list(workspace.path.glob("image-*.png"))
        finally:
            gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert completed.is_set()

    try:
        asyncio.run(scenario())
    finally:
        file.file.close()
    assert_staging_preserved(workspace, before)


@pytest.mark.parametrize("phase", ["write", "verify"])
def test_staging_failure_cleans_only_its_copy(workspace, monkeypatch, phase):
    file, _, before = staging_input(workspace)

    if phase == "write":
        original_open = Path.open

        class FailingWriter(GatedWriter):
            def write(self, chunk):
                self.handle.write(chunk[:1])
                self.handle.flush()
                raise OSError("Ошибка записи")

        def failing_open(path, mode="r", *args, **kwargs):
            handle = original_open(path, mode, *args, **kwargs)
            return FailingWriter(handle, None) if mode == "xb" else handle

        monkeypatch.setattr(Path, "open", failing_open)
        expected = OSError
    else:
        def fail_verify(image):
            raise ValueError("Повреждённое изображение")

        monkeypatch.setattr(PngImagePlugin.PngImageFile, "verify", fail_verify)
        expected = WebError

    async def scenario():
        with pytest.raises(expected) as caught:
            await stage_image(file, workspace)
        if phase == "verify":
            assert caught.value.code == "invalid_image" and caught.value.status_code == 400

    try:
        asyncio.run(scenario())
    finally:
        file.file.close()
    assert_staging_preserved(workspace, before)


def test_staging_open_failure_preserves_existing_candidate(workspace, monkeypatch):
    file, _, before = staging_input(workspace)
    collision = workspace.path / ("image-" + "a" * 32 + ".png")
    collision.write_bytes(b"existing candidate")
    before.update(snapshot([collision]))
    monkeypatch.setattr(workspace, "new_path", lambda kind, suffix: collision)

    async def scenario():
        with pytest.raises(FileExistsError):
            await stage_image(file, workspace)

    try:
        asyncio.run(scenario())
    finally:
        file.file.close()
    assert_staging_preserved(workspace, before)


@pytest.mark.parametrize("phase", ["write", "verify"])
def test_cancel_during_staging_closes_form_only_after_worker(upload_handler, monkeypatch, tmp_path, phase):
    workspace, parsers, middleware = upload_handler
    content = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(content, format="PNG")
    data = content.getvalue()
    workspace.limits = replace(workspace.limits, temp_bytes=2 * len(data))
    original = tmp_path / "user-result.txt"
    original.write_text("Существующий результат", encoding="utf-8")
    before = snapshot([original])

    async def scenario():
        gate = StageGate()
        install_stage_gate(monkeypatch, phase, gate)

        async def receive():
            return {"type": "http.request", "body": FILE_HEADER + data + FILE_END, "more_body": False}

        async def send(message):
            pytest.fail("Отменённая загрузка не должна отправлять ответ")

        task = asyncio.create_task(middleware(request_scope(), receive, send))
        try:
            await asyncio.wait_for(gate.entered.wait(), timeout=2)
            task.cancel()
            await loop_checkpoint()
            assert not task.done() and middleware.upload_slot.locked()
            assert all(not handle.closed for parser in parsers for handle in parser._files_to_close_on_error)
        finally:
            gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

    asyncio.run(scenario())
    assert not middleware.upload_slot.locked()
    assert_clean(workspace, parsers)
    assert snapshot([original]) == before


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
