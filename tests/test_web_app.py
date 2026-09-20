from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_local.models import OcrResult
from ocr_local.utils import output_cache
from ocr_local.web import app as web_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", tmp_path / "cache.json")
    web_app._jobs.clear()
    web_app._active_job_id = None
    web_app.logger.setLevel(logging.INFO)
    with TestClient(web_app.app) as client:
        yield client
    for job in web_app._jobs.values():
        if job.thread:
            job.thread.join(timeout=3)


def upload(client, tmp_path, model_id="glm-ocr"):
    path = tmp_path / "source.png"
    Image.new("RGB", (32, 24), "white").save(path)
    response = client.post("/api/upload", files={"file": ("image.png", path.read_bytes(), "image/png")},
                           data={"settings": json.dumps({"model_id": model_id})})
    assert response.status_code == 200
    return response.json()


def events(client, job_id):
    response = client.get(f"/api/stream/{job_id}")
    assert response.status_code == 200
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_startup_and_catalog_do_not_load_model(client, monkeypatch):
    monkeypatch.setattr(web_app._model_pool, "get", lambda *args: pytest.fail("Не нужны веса"))
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json()["status"] == "ok"
    assert len(client.get("/api/models").json()["models"]) == 2
    assert "clipboardData" in client.get("/static/app.js").text


def test_upload_and_model_change_return_only_selected_cache(client, tmp_path):
    first = upload(client, tmp_path)
    image_path = Path(first["image_path"])
    output_cache.update_output_cache(image_path, Path(first["output_path"]), "GLM result", web_app.logger,
                                     settings=first["settings"])
    second = upload(client, tmp_path)
    assert second["cached"] and second["text"] == "GLM result"
    paddle = client.post("/api/cached-result", json={
        "image_path": first["image_path"], "model_id": "paddleocr-vl-1.6",
    }).json()
    assert not paddle["cached"] and paddle["text"] == ""
    assert paddle["output_path"] != first["output_path"]
    assert paddle["settings"]["prompt"] == "OCR:"


def test_recognize_streams_selected_settings_text_metrics_and_logs(client, tmp_path, monkeypatch):
    payload = upload(client, tmp_path, "paddleocr-vl-1.6")

    def fake_run(options, *, loader):
        assert options.model_id == "paddleocr-vl-1.6"
        assert (options.prompt, options.max_new_tokens, options.max_pixels) == ("Custom:", 400, 500000)
        assert options.allow_cpu_fallback is True
        web_app.logger.info("Тестовый лог OCR.")
        return OcrResult(options.input_path, options.output_path, "Распознанный текст", "cuda", False,
                         model_id=options.model_id)
    monkeypatch.setattr(web_app, "run_ocr", fake_run)
    response = client.post("/api/recognize", json={
        "image_path": payload["image_path"], "model_id": "paddleocr-vl-1.6",
        "prompt": "Custom:", "max_new_tokens": 400, "max_pixels": 500000,
    })
    all_events = events(client, response.json()["job_id"])
    assert any("Тестовый лог" in e.get("message", "") for e in all_events)
    done = all_events[-1]
    assert done["status"] == "ok"
    assert done["result"]["text"] == "Распознанный текст"
    assert "metrics" in done["result"] and done["elapsed_seconds"] >= 0
    assert web_app._active_job_id is None


@pytest.mark.parametrize("settings", [
    {"model_id": "unknown"}, {"max_new_tokens": 0}, {"max_new_tokens": 1.5},
    {"max_pixels": 1}, {"prompt": ""},
])
def test_bad_settings_rejected_before_job(client, tmp_path, settings):
    uploaded = upload(client, tmp_path)
    response = client.post("/api/recognize", json={"image_path": uploaded["image_path"], **settings})
    assert response.status_code in (400, 422)
    assert not web_app._jobs and web_app._active_job_id is None


def test_thread_start_failure_releases_slot(client, tmp_path, monkeypatch):
    uploaded = upload(client, tmp_path)

    original_start = web_app.threading.Thread.start
    def fail(self):
        if self._target is web_app._run_recognition_job:
            raise RuntimeError("start failure")
        return original_start(self)
    monkeypatch.setattr(web_app.threading.Thread, "start", fail)
    # ASGI TestClient уже запущен контекстным менеджером.
    response = client.post("/api/recognize", json={"image_path": uploaded["image_path"]})
    assert response.status_code == 500
    assert web_app._active_job_id is None and not web_app._jobs


def test_error_done_releases_slot_and_keeps_output_absent(client, tmp_path, monkeypatch):
    uploaded = upload(client, tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("test GPU failure")
    monkeypatch.setattr(web_app, "run_ocr", fail)
    response = client.post("/api/recognize", json={"image_path": uploaded["image_path"]})
    done = events(client, response.json()["job_id"])[-1]
    assert done["status"] == "error" and "test GPU failure" in done["error"]
    assert web_app._active_job_id is None
    assert not Path(uploaded["output_path"]).exists()


def test_rejects_outside_uploads_and_active_job(client, tmp_path):
    outside = tmp_path / "outside.png"
    Image.new("RGB", (20, 20)).save(outside)
    assert client.post("/api/recognize", json={"image_path": str(outside)}).status_code == 400
    uploaded = upload(client, tmp_path)
    web_app._active_job_id = "other-job"
    assert client.post("/api/recognize", json={"image_path": uploaded["image_path"]}).status_code == 409
    web_app._active_job_id = None
