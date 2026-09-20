from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from ocr_local.constants import SUPPORTED_IMAGE_EXTENSIONS, UPLOADS_DIR
from ocr_local.exceptions import OcrError, OcrValidationError
from ocr_local.model_profiles import (
    MAX_NEW_TOKENS, MAX_PIXELS, model_catalog, resolve_options,
    settings_fingerprint, settings_identity,
)
from ocr_local.models import OcrOptions, OcrResult
from ocr_local.service import run_ocr
from ocr_local.utils.env_utils import load_environment
from ocr_local.utils.io_utils import validate_image_file
from ocr_local.utils.logging_utils import setup_logging
from ocr_local.utils.model_utils import ModelPool
from ocr_local.utils.output_cache import build_cache_key, get_cached_text, load_cache_snapshot

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
load_environment()
logger = logging.getLogger(__name__)
_model_pool = ModelPool()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(verbose=False)
    # Модель нужна только при первом промахе кеша, страница доступна сразу.
    try:
        yield
    finally:
        _model_pool.clear()


app = FastAPI(title="OCR Local", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ModelSettings(BaseModel):
    model_id: str | None = None
    prompt: str | None = None
    max_new_tokens: int | None = Field(default=None, ge=1, le=MAX_NEW_TOKENS, strict=True)
    max_pixels: int | None = Field(default=None, ge=1, le=MAX_PIXELS, strict=True)
    quantization_enabled: bool = False


class RecognizeRequest(ModelSettings):
    image_path: str
    force: bool = False
    allow_cpu_fallback: bool = True


@dataclass
class OcrWebJob:
    """Снимок настроек одной фоновой задачи."""

    job_id: str
    image_path: Path
    output_path: Path
    options: OcrOptions
    events: queue.Queue[dict[str, Any]]
    thread: threading.Thread | None = None
    completed: bool = False
    log_lines: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class QueueLogHandler(logging.Handler):
    def __init__(self, job: OcrWebJob) -> None:
        super().__init__()
        self._job = job

    def emit(self, record: logging.LogRecord) -> None:
        _emit_job_event(self._job, {"type": "log", "message": self.format(record)})


_jobs: dict[str, OcrWebJob] = {}
_active_job_id: str | None = None
_active_job_lock = threading.Lock()
LOG_HISTORY_LIMIT = 250


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models() -> dict[str, object]:
    try:
        return model_catalog()
    except OcrError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...), settings: str = Form("{}")) -> dict[str, Any]:
    try:
        selected = ModelSettings.model_validate_json(settings)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="Неверные настройки модели.") from exc
    # Профиль проверяется до сохранения входа.
    _make_options(Path("upload.png"), selected)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат изображения.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Файл изображения пуст.")
    image_path = _store_uploaded_image(content, suffix)
    try:
        validate_image_file(image_path)
    except OcrValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _cached_payload(image_path, _make_options(image_path, selected))


@app.post("/api/cached-result")
def cached_result(payload: RecognizeRequest) -> dict[str, Any]:
    image_path = _resolve_uploaded_path(Path(payload.image_path))
    return _cached_payload(image_path, _make_options(image_path, payload))


def _make_options(image_path: Path, settings: ModelSettings) -> OcrOptions:
    try:
        options = resolve_options(OcrOptions(
            input_path=image_path, model_id=settings.model_id, prompt=settings.prompt,
            max_new_tokens=settings.max_new_tokens, max_pixels=settings.max_pixels,
            quantization_enabled=settings.quantization_enabled,
            force=getattr(settings, "force", False),
            allow_cpu_fallback=getattr(settings, "allow_cpu_fallback", True),
        ))
    except OcrError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    name = f"{image_path.stem}.{options.model_id}.{settings_fingerprint(options)[:16]}.txt"
    return replace(options, output_path=image_path.with_name(name))


def _cached_payload(image_path: Path, options: OcrOptions) -> dict[str, Any]:
    settings = settings_identity(options)
    cache = load_cache_snapshot(logger)
    text = get_cached_text(cache, image_path, settings=settings)
    entry = cache.get(build_cache_key(image_path, settings), {})
    return {
        "image_path": str(image_path), "image_url": f"/api/image/{image_path.name}",
        "output_path": str(options.output_path), "model_id": options.model_id,
        "settings": settings, "cached": text is not None, "text": text or "",
        "limit_reached": bool(text is not None and entry.get("limit_reached", False)),
    }


@app.get("/api/image/{name}")
def image(name: str) -> FileResponse:
    image_path = _resolve_upload_name(name)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Изображение не найдено.")
    return FileResponse(image_path)


@app.post("/api/recognize")
def recognize(payload: RecognizeRequest) -> dict[str, str]:
    image_path = _resolve_uploaded_path(Path(payload.image_path))
    options = _make_options(image_path, payload)
    return _start_recognition_job(options)


@app.get("/api/stream/{job_id}")
def stream(job_id: str) -> StreamingResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена.")

    def _event_stream() -> Iterable[str]:
        while True:
            try:
                event = job.events.get(timeout=1.0)
            except queue.Empty:
                if job.completed:
                    break
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") == "done":
                break

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


def _start_recognition_job(options: OcrOptions) -> dict[str, str]:
    global _active_job_id
    with _active_job_lock:
        if _active_job_id is not None:
            raise HTTPException(status_code=409, detail="Распознавание уже выполняется.")
        job_id = uuid.uuid4().hex
        job = OcrWebJob(job_id, options.input_path, options.output_path, options, queue.Queue())
        thread = threading.Thread(target=_run_recognition_job, args=(job,), daemon=True)
        job.thread = thread
        _jobs[job_id], _active_job_id = job, job_id
        try:
            thread.start()
        except Exception as exc:
            _jobs.pop(job_id, None)
            _active_job_id = None
            raise HTTPException(status_code=500, detail="Не удалось запустить OCR-поток.") from exc
    return {"job_id": job_id}


def _run_recognition_job(job: OcrWebJob) -> None:
    global _active_job_id
    queue_handler = QueueLogHandler(job)
    queue_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ocr_logger = logging.getLogger("ocr_local")
    ocr_logger.addHandler(queue_handler)
    try:
        logger.info("Запущено OCR: %s, модель %s.", job.image_path.name, job.options.model_id)
        started_at = time.perf_counter()
        result = run_ocr(job.options, loader=_model_pool.get)
        elapsed_seconds = time.perf_counter() - started_at
        logger.info("OCR завершён за %.2f с.", elapsed_seconds)
        _emit_job_event(job, {
            "type": "done", "status": "ok", "result": _build_result_payload(result),
            "elapsed_seconds": elapsed_seconds,
        })
    except Exception as exc:
        logger.exception("Ошибка OCR: %s", exc)
        _emit_job_event(job, {"type": "done", "status": "error", "error": str(exc)})
    finally:
        ocr_logger.removeHandler(queue_handler)
        job.completed = True
        with _active_job_lock:
            if _active_job_id == job.job_id:
                _active_job_id = None


def _build_result_payload(result: OcrResult) -> dict[str, Any]:
    return {
        "image_path": str(result.input_path), "image_url": f"/api/image/{result.input_path.name}",
        "output_path": str(result.output_path), "text": result.text, "cached": result.cached,
        "device": result.device, "quantized": result.quantized,
        "model_id": result.model_id, "settings": result.settings, "metrics": asdict(result.metrics),
    }


def _emit_job_event(job: OcrWebJob, event: dict[str, Any]) -> None:
    if event.get("type") == "log":
        message = str(event.get("message", "")).rstrip()
        if message:
            with job.lock:
                job.log_lines.append(message)
                job.log_lines = job.log_lines[-LOG_HISTORY_LIMIT:]
    job.events.put(event)


def _store_uploaded_image(content: bytes, suffix: str) -> Path:
    digest = hashlib.sha256(content).hexdigest()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = UPLOADS_DIR / f"{digest}{suffix}"
    if not image_path.exists():
        image_path.write_bytes(content)
    return image_path


def _resolve_upload_name(name: str) -> Path | None:
    candidate = (UPLOADS_DIR / Path(name).name).resolve()
    if UPLOADS_DIR.resolve() not in candidate.parents or not candidate.is_file():
        return None
    return candidate


def _resolve_uploaded_path(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if UPLOADS_DIR.resolve() not in candidate.parents:
        raise HTTPException(status_code=400, detail="Изображение находится вне каталога загрузок.")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Изображение не найдено.")
    return candidate
