from __future__ import annotations

import io
import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_local.models import OcrResult
from ocr_local.model_profiles import settings_identity
from ocr_local.utils import output_cache
from ocr_local.web import app as web_app

ORIGIN = "http://127.0.0.1:7861"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", tmp_path / "cache.json")
    web_app.logger.setLevel(logging.INFO)
    with TestClient(web_app.create_app(), base_url=ORIGIN, client=("127.0.0.1", 50000),
                    headers={"Origin": ORIGIN}) as client:
        assert client.get("/").status_code == 200
        yield client


def image_bytes():
    data = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(data, format="PNG")
    return data.getvalue()


def upload(client, model_id="glm-ocr"):
    response = client.post("/api/upload", files={"file": ("image.png", image_bytes(), "image/png")},
                           data={"settings": json.dumps({"model_id": model_id})})
    assert response.status_code == 200, response.text
    return response.json()


def options_for(payload, **settings):
    path = web_app.UPLOADS_DIR / payload["upload_id"]
    return web_app._make_options(path, web_app.ModelSettings(**settings))


def events(client, job_id):
    response = client.get(f"/api/stream/{job_id}")
    assert response.status_code == 200, response.text
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def fake_result(options, text="Готовый текст"):
    return OcrResult(options.input_path, options.output_path, text, "cpu", False,
                     model_id=options.model_id, settings=settings_identity(options))


def test_startup_and_catalog_do_not_load_model(client, monkeypatch):
    monkeypatch.setattr(web_app._model_pool, "get", lambda *args: pytest.fail("Не нужны веса"))
    assert client.get("/api/health").json() == {"status": "ok"}
    catalog = client.get("/api/models").json()
    assert len(catalog["models"]) == 2
    assert all("path" not in item for item in catalog["models"])
    assert "clipboardData" in client.get("/static/app.js").text
    assert client.get("/docs").status_code == 404


def test_upload_and_model_change_return_only_selected_cache(client):
    first = upload(client)
    options = options_for(first, model_id="glm-ocr")
    output_cache.update_output_cache(options.input_path, options.output_path, "GLM result", web_app.logger,
                                     settings=settings_identity(options))
    second = upload(client)
    assert second["cached"] and second["text"] == "GLM result"
    paddle = client.post("/api/cached-result", json={
        "upload_id": first["upload_id"], "model_id": "paddleocr-vl-1.6",
    }).json()
    assert not paddle["cached"] and paddle["text"] == ""
    assert paddle["result_url"] != first["result_url"]
    assert paddle["settings"]["prompt"] == "OCR:"
    for payload in (first, paddle):
        assert not {"image_path", "output_path", "model_path"} & payload.keys()
        assert "model_path" not in payload["settings"]
    assert client.get(first["image_url"]).content == image_bytes()


def test_recognize_streams_selected_settings_text_and_metrics(client, monkeypatch):
    payload = upload(client, "paddleocr-vl-1.6")

    def fake_run(options, *, loader, storage_policy):
        assert options.model_id == "paddleocr-vl-1.6"
        assert (options.prompt, options.max_new_tokens, options.max_pixels) == ("Custom:", 400, 500000)
        assert options.allow_cpu_fallback is True
        result = fake_result(options, "Распознанный текст")
        storage_policy.save_result(result)
        return result

    monkeypatch.setattr(web_app, "run_ocr", fake_run)
    response = client.post("/api/recognize", json={
        "upload_id": payload["upload_id"], "model_id": "paddleocr-vl-1.6",
        "prompt": "Custom:", "max_new_tokens": 400, "max_pixels": 500000,
    })
    done = events(client, response.json()["job_id"])[-1]
    assert done["status"] == "ok"
    assert done["result"]["text"] == "Распознанный текст"
    assert "metrics" in done["result"] and done["elapsed_seconds"] >= 0
    assert web_app._registry.active_id is None
    assert client.get(done["result"]["result_url"]).text == "Распознанный текст"


@pytest.mark.parametrize("settings", [
    {"model_id": "unknown"}, {"max_new_tokens": 0}, {"max_new_tokens": 1.5},
    {"max_pixels": 1}, {"prompt": ""}, {"prompt": " \t\n\u001c"},
])
def test_bad_settings_rejected_before_job(client, settings):
    uploaded = upload(client)
    response = client.post("/api/recognize", json={"upload_id": uploaded["upload_id"], **settings})
    assert response.status_code in (400, 422)
    assert not web_app._registry.jobs and web_app._registry.active_id is None


def test_thread_start_failure_releases_slot_and_reservation(client, monkeypatch):
    uploaded = upload(client)
    original_start = web_app.threading.Thread.start

    def fail(self):
        if self._target is web_app._run_recognition_job:
            raise RuntimeError("start failure")
        return original_start(self)

    monkeypatch.setattr(web_app.threading.Thread, "start", fail)
    response = client.post("/api/recognize", json={"upload_id": uploaded["upload_id"]})
    assert response.status_code == 500
    assert web_app._registry.active_id is None and not web_app._registry.jobs
    assert not web_app._storage._reservations


def test_error_done_releases_slot_and_keeps_output_absent(client, monkeypatch, caplog):
    uploaded = upload(client)
    private = r"C:\private\weights\model.safetensors"

    def fail(*args, **kwargs):
        raise RuntimeError(private)

    monkeypatch.setattr(web_app, "run_ocr", fail)
    response = client.post("/api/recognize", json={"upload_id": uploaded["upload_id"]})
    done = events(client, response.json()["job_id"])[-1]
    assert done["status"] == "error" and done["code"] == "internal_error"
    assert "private" not in json.dumps(done)
    assert done["diagnostic_id"] in caplog.text and private in caplog.text
    assert web_app._registry.active_id is None
    assert not options_for(uploaded).output_path.exists()


def test_rejects_outside_uploads_and_active_job(client, tmp_path):
    outside = tmp_path / "outside.png"
    assert client.post("/api/recognize", json={"upload_id": str(outside)}).status_code == 400
    uploaded = upload(client)
    web_app._registry.active_id = "other-job"
    try:
        assert client.post("/api/recognize", json={"upload_id": uploaded["upload_id"]}).status_code == 409
    finally:
        web_app._registry.active_id = None


@pytest.mark.parametrize("outcome", ["ok", "error"])
def test_completed_stream_replays_terminal_event_without_new_ocr(client, monkeypatch, outcome):
    uploaded = upload(client)
    calls = []

    def fake_run(options, *, loader, storage_policy):
        calls.append(options)
        if outcome == "error":
            raise RuntimeError("Тестовый отказ OCR")
        return fake_result(options)

    monkeypatch.setattr(web_app, "run_ocr", fake_run)
    response = client.post("/api/recognize", json={"upload_id": uploaded["upload_id"]})
    job_id = response.json()["job_id"]
    first = events(client, job_id)[-1]
    assert first["type"] == "done" and first["status"] == outcome
    job = web_app._registry.jobs[job_id]
    assert not job.events and job.options is None and job.thread is None
    assert events(client, job_id) == [first]
    assert events(client, job_id) == [first]
    assert len(calls) == 1


@pytest.mark.parametrize("outcome", ["ok", "error"])
def test_terminal_event_is_published_after_job_release(client, monkeypatch, outcome):
    uploaded = upload(client)
    publication_states = []
    original_emit = web_app._emit_job_event

    def capture_publication(job, event):
        if event.get("type") == "done":
            handler_present = any(
                isinstance(handler, web_app.QueueLogHandler) and handler._job is job
                for handler in logging.getLogger("ocr_local").handlers
            )
            publication_states.append((web_app._registry.active_id, handler_present))
        original_emit(job, event)

    def fake_run(options, *, loader, storage_policy):
        if outcome == "error":
            raise RuntimeError("Тестовый отказ OCR")
        return fake_result(options)

    monkeypatch.setattr(web_app, "_emit_job_event", capture_publication)
    monkeypatch.setattr(web_app, "run_ocr", fake_run)
    response = client.post("/api/recognize", json={"upload_id": uploaded["upload_id"]})
    assert events(client, response.json()["job_id"])[-1]["status"] == outcome
    assert publication_states == [(None, False)]


def test_upload_is_rejected_before_reading_when_job_active(client):
    reads = []

    def body():
        reads.append(True)
        yield image_bytes()

    web_app._registry.active_id = "active-job"
    try:
        response = client.post("/api/upload", content=body(), headers={"Content-Type": "multipart/form-data"})
        assert response.status_code == 409 and reads == []
        assert not list(web_app.UPLOADS_DIR.glob("*.png"))
    finally:
        web_app._registry.active_id = None


def test_upload_is_rejected_if_job_starts_during_read(client, monkeypatch):
    original_stage = web_app.stage_image

    async def stage(file, workspace):
        staged = await original_stage(file, workspace)
        web_app._registry.active_id = "started-during-read"
        return staged

    monkeypatch.setattr(web_app, "stage_image", stage)
    try:
        response = client.post("/api/upload", files={"file": ("image.png", image_bytes())})
        assert response.status_code == 409
        assert not list(web_app.UPLOADS_DIR.glob("*.png"))
        assert list(web_app._workspace.path.iterdir()) == [web_app._workspace.path / ".owner"]
    finally:
        web_app._registry.active_id = None


@pytest.mark.parametrize("filename,content,status", [
    ("empty.png", b"", 400), ("bad.png", b"not an image", 400), ("file.pdf", b"%PDF", 400),
])
def test_rejected_upload_cleans_all_temporary_files(client, filename, content, status):
    response = client.post("/api/upload", files={"file": (filename, content)})
    assert response.status_code == status
    assert not list(web_app.UPLOADS_DIR.glob("*.png"))
    assert [p.name for p in web_app._workspace.path.iterdir()] == [".owner"]


def test_excess_form_parts_and_settings_are_rejected(client):
    cases = [
        {"files": [("file", ("a.png", image_bytes())), ("file", ("b.png", image_bytes()))]},
        {"files": {"file": ("a.png", image_bytes())}, "data": {"settings": "{}", "extra": "x"}},
        {"files": {"file": ("a.png", image_bytes())}, "data": {"settings": " " * (32 * 1024 + 1)}},
    ]
    for kwargs in cases:
        assert client.post("/api/upload", **kwargs).status_code in (400, 413)
        assert [p.name for p in web_app._workspace.path.iterdir()] == [".owner"]
    assert not list(web_app.UPLOADS_DIR.glob("*.png"))


def test_validation_does_not_reflect_input(client, caplog):
    marker = r"C:\secret\private\path"
    response = client.post("/api/recognize", json={"upload_id": "x", "max_pixels": marker, "secret": marker})
    assert response.status_code == 422
    assert marker not in response.text and "input" not in response.json()
    assert response.json()["code"] == "invalid_request"
    assert response.json()["diagnostic_id"] in caplog.text
    assert "RequestValidationError" in caplog.text and "private" in caplog.text


def test_saved_text_remains_readable_when_cache_is_too_large(client):
    uploaded = upload(client)
    options = options_for(uploaded)
    options.output_path.write_text("Пользовательский текст", encoding="utf-8")
    web_app._storage.limits = replace(web_app.LIMITS, cache_bytes=4)
    output_cache.OCR_CACHE_FILE.write_text('{"too": "large"}', encoding="utf-8")
    assert client.post("/api/cached-result", json={"upload_id": uploaded["upload_id"]}).status_code == 507
    assert client.get(uploaded["result_url"]).text == "Пользовательский текст"


def test_size_failure_before_saving_result_becomes_safe_done(client, monkeypatch):
    uploaded = upload(client)

    def large_result(options, *, loader, storage_policy):
        result = fake_result(options, "x" * (web_app.LIMITS.text_bytes + 1))
        storage_policy.save_result(result)
        return result

    monkeypatch.setattr(web_app, "run_ocr", large_result)
    done = events(client, client.post("/api/recognize", json={"upload_id": uploaded["upload_id"]}).json()["job_id"])[-1]
    assert done["status"] == "error" and done["code"] == "result_too_large"
    assert not options_for(uploaded).output_path.exists()
    assert not output_cache.OCR_CACHE_FILE.exists()


def test_only_safe_stage_logs_enter_stream(client):
    job = web_app._registry.add(None, None)
    handler = web_app.QueueLogHandler(job)
    record = logging.LogRecord("ocr_local", logging.ERROR, "private.py", 1, r"C:\private\file", (), None)
    handler.emit(record)
    assert not job.events
    record.ocr_stage = "started"
    handler.emit(record)
    assert len(job.events) == 1
    assert b"private" not in job.events[0]
    web_app._registry.discard_start(job)


@pytest.mark.parametrize("extra,status", [(0, 200), (1, 200), (2, 413)])
def test_file_size_boundary_cleans_temporary_files(client, extra, status):
    content = image_bytes()
    web_app._workspace.limits = replace(web_app.LIMITS, image_bytes=len(content) + 1)
    response = client.post("/api/upload", files={"file": ("a.png", content + b" " * extra)})
    assert response.status_code == status
    assert [path.name for path in web_app._workspace.path.iterdir()] == [".owner"]


@pytest.mark.parametrize("size,status", [(32767, 200), (32768, 200), (32769, 413)])
def test_settings_field_byte_boundary(client, size, status):
    response = client.post("/api/upload", files={"file": ("a.png", image_bytes())},
                           data={"settings": "{}" + " " * (size - 2)})
    assert response.status_code == status
    assert [path.name for path in web_app._workspace.path.iterdir()] == [".owner"]


@pytest.mark.parametrize("prompt,status", [("я" * 4095 + "x", 200), ("я" * 4096, 200),
                                          ("я" * 4096 + "x", 413)])
def test_prompt_utf8_byte_boundary(client, prompt, status):
    uploaded = upload(client)
    response = client.post("/api/cached-result", json={"upload_id": uploaded["upload_id"], "prompt": prompt})
    assert response.status_code == status


@pytest.mark.parametrize("pixels,status", [(767, 413), (768, 200), (769, 200)])
def test_source_pixel_boundary_is_checked_before_publish(client, pixels, status):
    web_app._workspace.limits = replace(web_app.LIMITS, image_pixels=pixels)
    response = client.post("/api/upload", files={"file": ("a.png", image_bytes())})
    assert response.status_code == status
    assert len(list(web_app.UPLOADS_DIR.glob("*.png"))) == (status == 200)


def test_publication_write_failure_closes_and_removes_temporary_files(client, monkeypatch):
    import ocr_local.web.storage as storage

    def fail(*args, **kwargs):
        raise OSError(r"C:\private\disk")

    monkeypatch.setattr(storage.os, "link", fail)
    response = client.post("/api/upload", files={"file": ("a.png", image_bytes())})
    assert response.status_code == 507
    assert "private" not in response.text
    assert not list(web_app.UPLOADS_DIR.glob("*.png"))
    assert [path.name for path in web_app._workspace.path.iterdir()] == [".owner"]


def test_partial_multipart_disconnect_cleans_open_files(client, monkeypatch):
    from starlette.requests import ClientDisconnect

    parsers = []
    original_parser = web_app.UploadParser

    class CapturedParser(original_parser):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            parsers.append(self)

    monkeypatch.setattr(web_app, "UploadParser", CapturedParser)

    def partial_body():
        yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\n\r\npartial'
        raise ClientDisconnect()

    response = client.post("/api/upload", content=partial_body(),
                           headers={"Content-Type": "multipart/form-data; boundary=boundary"})
    assert response.status_code == 400
    assert parsers and all(handle.closed for parser in parsers for handle in parser._files_to_close_on_error)
    assert [path.name for path in web_app._workspace.path.iterdir()] == [".owner"]
    assert not list(web_app.UPLOADS_DIR.glob("*.png"))


def test_thread_construction_failure_releases_reserved_resources(client, monkeypatch):
    uploaded = upload(client)
    original_thread = web_app.threading.Thread

    def make_thread(*args, **kwargs):
        if kwargs.get("target") is web_app._run_recognition_job:
            raise RuntimeError("Создание потока невозможно")
        return original_thread(*args, **kwargs)

    monkeypatch.setattr(web_app.threading, "Thread", make_thread)
    response = client.post("/api/recognize", json={"upload_id": uploaded["upload_id"]})
    assert response.status_code == 500
    assert not web_app._registry.jobs and not web_app._storage._reservations
