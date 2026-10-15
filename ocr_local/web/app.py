from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException
from starlette.requests import ClientDisconnect

from ocr_local import __version__
from ocr_local.constants import UPLOADS_DIR
from ocr_local.exceptions import OcrError
from ocr_local.model_profiles import (
    MAX_NEW_TOKENS, MAX_PIXELS, model_catalog, resolve_options, settings_fingerprint, settings_identity,
)
from ocr_local.models import OcrOptions, OcrResult
from ocr_local.service import run_ocr
from ocr_local.utils import output_cache
from ocr_local.utils.env_utils import load_environment
from ocr_local.utils.logging_utils import setup_logging
from ocr_local.utils.model_utils import ModelPool
from ocr_local.web.errors import WebError, public_error
from ocr_local.web.jobs import JobRegistry, JobStreamResponse, OcrWebJob, encode_event
from ocr_local.web.limits import DEFAULT_LIMITS
from ocr_local.web.request_limits import RequestLimits
from ocr_local.web.security import LocalWebSecurity, install_secret_log_redaction
from ocr_local.web.storage import StorageBudget
from ocr_local.web.uploads import TemporaryUploads, UploadParser, stage_image

STATIC_DIR = Path(__file__).resolve().parent / "static"
LIMITS = DEFAULT_LIMITS
load_environment()
logger = logging.getLogger(__name__)
_model_pool = ModelPool()
_registry = JobRegistry(LIMITS)
_storage = None
_workspace = None
router = APIRouter()
UPLOAD_ID_PATTERN = re.compile(r"[0-9a-f]{64}\.(?:jpg|jpeg|png|webp|bmp)")
RESULT_ID_PATTERN = re.compile(r"[0-9a-f]{64}\.(?:glm-ocr|paddleocr-vl-1\.6)\.[0-9a-f]{16}\.txt")
PUBLIC_SETTINGS = ("version", "model_id", "prompt", "max_new_tokens", "max_pixels", "quantization_enabled")
MODEL_FIELDS = ("id", "name", "model_type", "prompt", "max_new_tokens", "max_pixels", "min_pixels", "available")


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _registry, _storage, _workspace
    setup_logging(verbose=False)
    install_secret_log_redaction()
    _registry = JobRegistry(LIMITS)
    _storage = StorageBudget(UPLOADS_DIR, output_cache.OCR_CACHE_FILE, LIMITS)
    _storage.usage()
    _workspace = TemporaryUploads(UPLOADS_DIR, LIMITS)
    _workspace.cleanup()
    cleanup_task = asyncio.create_task(_clean_jobs())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        with _registry.lock:
            running = [job.thread for job in _registry.jobs.values() if job.thread is not None]
        for thread in running:
            await asyncio.to_thread(thread.join)
        _model_pool.clear()


async def _clean_jobs() -> None:
    while True:
        await asyncio.sleep(LIMITS.cleanup_seconds)
        _registry.cleanup()


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str | None = None
    prompt: str | None = None
    max_new_tokens: int | None = Field(default=None, ge=1, le=MAX_NEW_TOKENS, strict=True)
    max_pixels: int | None = Field(default=None, ge=1, le=MAX_PIXELS, strict=True)
    quantization_enabled: bool = False

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value):
        if value is not None:
            if not value.strip():
                raise ValueError("Промпт не должен быть пустым.")
            if len(value.encode("utf-8")) > LIMITS.prompt_bytes:
                raise WebError("request_too_large", 413, "Промпт превышает 8 КиБ UTF-8.")
        return value


class RecognizeRequest(ModelSettings):
    upload_id: str
    force: bool = False
    allow_cpu_fallback: bool = True


class QueueLogHandler(logging.Handler):
    messages = {
        "started": "Распознавание началось.",
        "finished": "Распознавание завершено.",
        "limit": "Достигнут предел новых токенов; текст может быть неполным.",
    }

    def __init__(self, job: OcrWebJob) -> None:
        super().__init__()
        self._job = job

    def emit(self, record: logging.LogRecord) -> None:
        message = self.messages.get(getattr(record, "ocr_stage", None))
        if message:
            _emit_job_event(self._job, {"type": "log", "message": message})


@router.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@router.get("/api/health")
def health():
    return {"status": "ok"}


@router.get("/api/models")
def models():
    catalog = model_catalog()
    return {
        "models": [{key: item[key] for key in MODEL_FIELDS if key in item} for item in catalog["models"]],
        "default_model_id": catalog["default_model_id"],
        "max_new_tokens_limit": catalog["max_new_tokens_limit"],
        "max_pixels_limit": catalog["max_pixels_limit"],
        "upload_bytes_limit": LIMITS.image_bytes,
        "prompt_bytes_limit": LIMITS.prompt_bytes,
        "result_retention_seconds": LIMITS.job_ttl_seconds,
    }


@router.post("/api/upload")
async def upload_image(request: Request):
    if _registry.is_busy():
        raise WebError("ocr_busy", 409, "Распознавание уже выполняется.")
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise WebError("invalid_request", 400, "Ожидается форма с изображением.")
    parser = UploadParser(request.headers, request.stream(), _workspace)
    staged = None
    try:
        form = await parser.parse()
        if not parser.finished or parser._current_files != 1:
            raise MultiPartException("Изображение отсутствует или загрузка не завершена.")
        selected = ModelSettings.model_validate_json(form.get("settings", "{}"))
        _make_options(Path("upload.png"), selected)
        staged, digest, suffix = await stage_image(form["file"], _workspace)
        with _registry.lock:
            if _registry.is_busy():
                raise WebError("ocr_busy", 409, "Распознавание уже выполняется.")
            image_path = UPLOADS_DIR / (digest + suffix)
            payload = _cached_payload(image_path, _make_options(image_path, selected))
            _storage.publish_upload(staged, digest, suffix)
        return payload
    except ValidationError as exc:
        raise WebError("invalid_settings", 422, "Неверные настройки модели.") from exc
    except (MultiPartException, ValueError, KeyError, TypeError) as exc:
        raise WebError("invalid_request", 400, "Неверная форма загрузки изображения.") from exc
    except ClientDisconnect as exc:
        raise WebError("upload_interrupted", 400, "Загрузка изображения прервана.") from exc
    finally:
        parser.close()
        if staged is not None:
            staged.unlink(missing_ok=True)


@router.post("/api/cached-result")
def cached_result(payload: RecognizeRequest):
    image_path = _resolve_upload_id(payload.upload_id)
    return _cached_payload(image_path, _make_options(image_path, payload))


def _make_options(image_path: Path, settings: ModelSettings) -> OcrOptions:
    options = resolve_options(OcrOptions(
        input_path=image_path, model_id=settings.model_id, prompt=settings.prompt,
        max_new_tokens=settings.max_new_tokens, max_pixels=settings.max_pixels,
        quantization_enabled=settings.quantization_enabled, force=getattr(settings, "force", False),
        allow_cpu_fallback=getattr(settings, "allow_cpu_fallback", True),
    ))
    name = f"{image_path.stem}.{options.model_id}.{settings_fingerprint(options)[:16]}.txt"
    return replace(options, output_path=image_path.with_name(name))


def _public_settings(settings: dict) -> dict:
    return {key: settings[key] for key in PUBLIC_SETTINGS if key in settings}


def _cached_payload(image_path: Path, options: OcrOptions):
    settings = settings_identity(options)
    cache = _storage.load_cache()
    text = output_cache.get_cached_text(cache, image_path, settings=settings)
    if text is not None and len(text.encode("utf-8")) > LIMITS.text_bytes:
        raise WebError("stored_data_too_large", 413, "Сохранённый результат превышает допустимый размер.")
    entry = cache.get(output_cache.build_cache_key(image_path, settings), {})
    return {
        "upload_id": image_path.name, "image_url": f"/api/image/{image_path.name}",
        "result_url": f"/api/result/{options.output_path.name}",
        "model_id": options.model_id, "settings": _public_settings(settings),
        "cached": text is not None, "text": text or "",
        "limit_reached": bool(text is not None and entry.get("limit_reached", False)),
    }


@router.get("/api/image/{upload_id}")
def image(upload_id: str):
    return FileResponse(_resolve_upload_id(upload_id))


@router.get("/api/result/{result_id}")
def saved_result(result_id: str):
    if not RESULT_ID_PATTERN.fullmatch(result_id):
        raise WebError("invalid_result_id", 400, "Неверный идентификатор результата.")
    candidate = (UPLOADS_DIR / result_id).resolve()
    if candidate.parent != UPLOADS_DIR.resolve() or not candidate.is_file():
        raise WebError("result_not_found", 404, "Сохранённый результат не найден.")
    # Чтение отдельно от кеша сохраняет доступ к TXT, не объявляя его совпавшим профильным результатом.
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(_storage.read_text(candidate), headers={"Cache-Control": "no-store"})


@router.post("/api/recognize")
def recognize(payload: RecognizeRequest):
    image_path = _resolve_upload_id(payload.upload_id)
    options = _make_options(image_path, payload)
    return _start_recognition_job(options)


@router.get("/api/stream/{job_id}")
def stream(job_id: str):
    return JobStreamResponse(_registry, _registry.subscribe(job_id))


def _start_recognition_job(options: OcrOptions):
    with _registry.lock:
        _registry.check_admission()
        reservation = _storage.reserve(options, validate_result=_validate_result)
        job = None
        try:
            job = _registry.add(options, reservation)
            thread = threading.Thread(target=_run_recognition_job, args=(job,), daemon=True)
            job.thread = thread
            thread.start()
        except Exception as exc:
            if job is not None:
                _registry.discard_start(job)
            reservation.release()
            raise WebError("start_failed", 500, "Не удалось запустить OCR. Повторите попытку.") from exc
    return {"job_id": job.job_id}


def _run_recognition_job(job: OcrWebJob) -> None:
    queue_handler = QueueLogHandler(job)
    ocr_logger = logging.getLogger("ocr_local")
    ocr_logger.addHandler(queue_handler)
    reservation = job.reservation
    try:
        logger.info("Запущено OCR модели %s.", job.options.model_id, extra={"ocr_stage": "started"})
        started = time.perf_counter()
        result = run_ocr(job.options, loader=_model_pool.get, storage_policy=reservation)
        _validate_result(result)
        logger.info("OCR завершён.", extra={"ocr_stage": "finished"})
        terminal = {"type": "done", "status": "ok", "result": _build_result_payload(result),
                    "elapsed_seconds": time.perf_counter() - started}
    except Exception as exc:
        _, payload = public_error(exc, logger)
        terminal = {"type": "done", "status": "error", "error": payload.pop("detail"), **payload}
    finally:
        ocr_logger.removeHandler(queue_handler)
        reservation.release()
    # Освобождение слота и публикация под одним замком исключают повторный старт между ними.
    with _registry.lock:
        if _registry.active_id == job.job_id:
            _registry.active_id = None
        _emit_job_event(job, terminal)


def _build_result_payload(result: OcrResult):
    return {
        "upload_id": result.input_path.name, "image_url": f"/api/image/{result.input_path.name}",
        "result_url": f"/api/result/{result.output_path.name}",
        "text": result.text, "cached": result.cached, "device": result.device, "quantized": result.quantized,
        "model_id": result.model_id, "settings": _public_settings(result.settings), "metrics": asdict(result.metrics),
    }


def _validate_result(result: OcrResult):
    # Максимальное представление времени оставляет место для фактического измерения после записи.
    event = {"type": "done", "status": "ok", "result": _build_result_payload(result),
             "elapsed_seconds": 1.7976931348623157e308}
    if len(result.text.encode("utf-8")) > LIMITS.text_bytes or len(encode_event(event)) > LIMITS.terminal_bytes:
        raise WebError("result_too_large", 413, "Результат превышает допустимый размер.")


def _emit_job_event(job, event):
    _registry.emit(job, event)


def _resolve_upload_id(upload_id: str) -> Path:
    if not UPLOAD_ID_PATTERN.fullmatch(upload_id):
        raise WebError("invalid_upload_id", 400, "Неверный идентификатор изображения.")
    candidate = (UPLOADS_DIR / upload_id).resolve()
    if candidate.parent != UPLOADS_DIR.resolve() or not candidate.is_file():
        raise WebError("image_not_found", 404, "Изображение не найдено.")
    return candidate


async def _error_response(request: Request, exc: Exception):
    original = exc
    if isinstance(exc, RequestValidationError):
        exc = WebError("invalid_request", 422, "Неверные поля или параметры запроса.")
    elif isinstance(exc, HTTPException) and not isinstance(exc, WebError):
        exc = WebError("request_failed", exc.status_code, "Запрос не может быть выполнен.")
    if exc is not original:
        exc.__cause__ = original
    status, payload = public_error(exc, logger)
    return JSONResponse(payload, status_code=status, headers=getattr(exc, "headers", None))


def create_app(host="127.0.0.1", port=7861):
    application = FastAPI(title="OCR Local", version=__version__, lifespan=lifespan,
                          docs_url=None, redoc_url=None, openapi_url=None)
    application.include_router(router)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    for error_type in (WebError, OcrError, RequestValidationError, HTTPException, Exception):
        application.add_exception_handler(error_type, _error_response)
    bounded = RequestLimits(application, limits=LIMITS, is_busy=lambda: _registry.is_busy(), logger=logger)
    return LocalWebSecurity(bounded, host=host, port=port)


app = create_app()
