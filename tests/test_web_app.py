from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from ocr_local.models import OcrResult
from ocr_local.utils import output_cache
from ocr_local.web import app as web_app


def test_web_index_loads() -> None:
    client = TestClient(web_app.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "OCR Local" in response.text


def test_web_app_js_supports_clipboard_paste() -> None:
    client = TestClient(web_app.app)

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert 'document.addEventListener("paste"' in response.text
    assert "clipboardData" in response.text


def test_upload_returns_cached_text_on_reupload(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", tmp_path / "ocr_cache.json")
    client = TestClient(web_app.app)
    image_bytes = _make_image_bytes(tmp_path / "sample.jpg")

    first = client.post(
        "/api/upload",
        files={"file": ("sample.jpg", image_bytes, "image/jpeg")},
    )
    assert first.status_code == 200
    image_path = Path(first.json()["image_path"])
    output_cache.update_output_cache(
        image_path,
        image_path.with_suffix(".txt"),
        "Текст из кеша",
        web_app.logger,
    )

    second = client.post(
        "/api/upload",
        files={"file": ("sample.jpg", image_bytes, "image/jpeg")},
    )

    assert second.status_code == 200
    payload = second.json()
    assert payload["cached"] is True
    assert payload["text"] == "Текст из кеша"


def test_recognize_streams_text_and_logs(tmp_path, monkeypatch):
    _reset_web_jobs()
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(web_app, "UPLOADS_DIR", upload_dir)
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", tmp_path / "ocr_cache.json")
    image_path = upload_dir / "image.jpg"
    upload_dir.mkdir(parents=True)
    image_path.write_bytes(_make_image_bytes(tmp_path / "source.jpg"))

    def fake_run_ocr(options):
        web_app.logger.info("Тестовый лог OCR.")
        return OcrResult(
            input_path=Path(options.input_path),
            output_path=Path(options.output_path),
            text="Распознанный текст",
            device="cuda",
            quantized=True,
            cached=False,
        )

    monkeypatch.setattr(web_app, "run_ocr", fake_run_ocr)
    client = TestClient(web_app.app)

    response = client.post(
        "/api/recognize",
        json={"image_path": str(image_path), "force": False},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    stream_response = client.get(f"/api/stream/{job_id}")
    assert stream_response.status_code == 200
    events = _parse_sse_events(stream_response.text)
    done_event = events[-1]
    result = done_event["result"]

    assert any(
        event.get("type") == "log" and "Тестовый лог OCR." in event.get("message", "")
        for event in events
    )
    assert done_event["type"] == "done"
    assert done_event["status"] == "ok"
    assert result["text"] == "Распознанный текст"
    assert result["cached"] is False
    assert result["device"] == "cuda"


def _make_image_bytes(path: Path) -> bytes:
    image = Image.new("RGB", (32, 24), "white")
    image.save(path, format="JPEG")
    return path.read_bytes()


def _parse_sse_events(body: str) -> list[dict[str, object]]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line.removeprefix("data: ")))
    return events


def _reset_web_jobs() -> None:
    web_app._jobs.clear()
    web_app._active_job_id = None
