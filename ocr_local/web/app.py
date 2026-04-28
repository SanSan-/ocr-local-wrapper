from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ocr_local.constants import SUPPORTED_IMAGE_EXTENSIONS, UPLOADS_DIR
from ocr_local.exceptions import OcrError, OcrValidationError
from ocr_local.models import OcrOptions, OcrResult
from ocr_local.service import run_ocr
from ocr_local.utils.env_utils import get_default_model_path, load_environment
from ocr_local.utils.io_utils import validate_image_file
from ocr_local.utils.logging_utils import setup_logging
from ocr_local.utils.output_cache import get_cached_text, load_cache_snapshot

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
load_environment()
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR Local")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RecognizeRequest(BaseModel):
    image_path: str
    force: bool = False
    allow_cpu_fallback: bool = True


@dataclass
class OcrWebJob:
    """Состояние фонового OCR-запуска."""

    job_id: str
    image_path: Path
    output_path: Path
    request: RecognizeRequest
    events: queue.Queue[dict[str, Any]]
    thread: threading.Thread | None
    completed: bool = False
    log_lines: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class QueueLogHandler(logging.Handler):
    """Отправляет строки логов в очередь SSE-событий."""

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


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)) -> dict[str, Any]:
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
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_path = _build_output_path(image_path)
    cached_text = get_cached_text(load_cache_snapshot(logger), image_path)
    if cached_text is None and output_path.exists():
        cached_text = output_path.read_text(encoding="utf-8")

    return {
        "image_path": str(image_path),
        "image_url": f"/api/image/{image_path.name}",
        "output_path": str(output_path),
        "cached": cached_text is not None,
        "text": cached_text or "",
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
    output_path = _build_output_path(image_path)
    return _start_recognition_job(image_path, output_path, payload)


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
            payload = json.dumps(event, ensure_ascii=False)
            yield f"data: {payload}\n\n"
            if event.get("type") == "done":
                break

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


def _start_recognition_job(
    image_path: Path,
    output_path: Path,
    request: RecognizeRequest,
) -> dict[str, str]:
    global _active_job_id
    with _active_job_lock:
        if _active_job_id is not None:
            raise HTTPException(status_code=409, detail="Распознавание уже выполняется.")

        job_id = uuid.uuid4().hex
        job = OcrWebJob(
            job_id=job_id,
            image_path=image_path,
            output_path=output_path,
            request=request,
            events=queue.Queue(),
            thread=None,
        )
        thread = threading.Thread(target=_run_recognition_job, args=(job,), daemon=True)
        job.thread = thread
        _jobs[job_id] = job
        _active_job_id = job_id

    thread.start()
    return {"job_id": job_id}


def _run_recognition_job(job: OcrWebJob) -> None:
    global _active_job_id
    setup_logging(verbose=False)
    queue_handler = QueueLogHandler(job)
    queue_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ocr_logger = logging.getLogger("ocr_local")
    ocr_logger.addHandler(queue_handler)
    try:
        logger.info("Запущено OCR-распознавание: %s", job.image_path.name)
        result = run_ocr(
            OcrOptions(
                input_path=job.image_path,
                output_path=job.output_path,
                model_path=get_default_model_path(),
                allow_cpu_fallback=job.request.allow_cpu_fallback,
                force=job.request.force,
            )
        )
        _emit_job_event(
            job,
            {
                "type": "done",
                "status": "ok",
                "result": _build_result_payload(result),
            },
        )
    except OcrError as exc:
        logger.error("Ошибка OCR: %s", exc)
        _emit_job_event(job, {"type": "done", "status": "error", "error": str(exc)})
    except Exception as exc:
        logger.exception("Непредвиденная ошибка OCR: %s", exc)
        _emit_job_event(job, {"type": "done", "status": "error", "error": str(exc)})
    finally:
        ocr_logger.removeHandler(queue_handler)
        job.completed = True
        with _active_job_lock:
            if _active_job_id == job.job_id:
                _active_job_id = None


def _build_result_payload(result: OcrResult) -> dict[str, Any]:
    return {
        "image_path": str(result.input_path),
        "image_url": f"/api/image/{result.input_path.name}",
        "output_path": str(result.output_path),
        "text": result.text,
        "cached": result.cached,
        "device": result.device,
        "quantized": result.quantized,
    }


def _emit_job_event(job: OcrWebJob, event: dict[str, Any]) -> None:
    _record_job_event(job, event)
    job.events.put(event)


def _record_job_event(job: OcrWebJob, event: dict[str, Any]) -> None:
    if event.get("type") != "log":
        return
    message = str(event.get("message", "")).rstrip()
    if not message:
        return
    with job.lock:
        job.log_lines.append(message)
        if len(job.log_lines) > LOG_HISTORY_LIMIT:
            job.log_lines = job.log_lines[-LOG_HISTORY_LIMIT:]


def _store_uploaded_image(content: bytes, suffix: str) -> Path:
    digest = hashlib.sha256(content).hexdigest()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = UPLOADS_DIR / f"{digest}{suffix}"
    if not image_path.exists():
        image_path.write_bytes(content)
    return image_path


def _build_output_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def _resolve_upload_name(name: str) -> Path | None:
    candidate = (UPLOADS_DIR / Path(name).name).resolve()
    upload_root = UPLOADS_DIR.resolve()
    if upload_root not in candidate.parents or not candidate.exists():
        return None
    return candidate


def _resolve_uploaded_path(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    upload_root = UPLOADS_DIR.resolve()
    if upload_root not in candidate.parents:
        raise HTTPException(status_code=400, detail="Изображение находится вне каталога загрузок.")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Изображение не найдено.")
    return candidate
