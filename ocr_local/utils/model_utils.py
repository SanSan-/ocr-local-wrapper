from __future__ import annotations

import gc
import importlib.util
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from ocr_local.exceptions import OcrModelError
from ocr_local.models import LoadedOcrModel

logger = logging.getLogger(__name__)


def load_model_components(
    model_path: Path, allow_quantization: bool, allow_cpu_fallback: bool,
) -> LoadedOcrModel:
    """Загружает только местные веса сразу на выбранное устройство."""
    torch = _import_torch()
    processor_class, model_class = _import_transformers_components()
    device, quantization_config, quantized = resolve_device_and_quantization(
        allow_quantization=allow_quantization,
    )
    try:
        processor = processor_class.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
        )
    except Exception as exc:
        raise OcrModelError(f"Не удалось загрузить процессор {model_path.name}: {exc}") from exc
    try:
        model = _load_model(model_class, model_path, device, quantization_config)
    except Exception as exc:
        if not allow_cpu_fallback or device.type != "cuda":
            raise OcrModelError(f"Не удалось загрузить OCR-модель: {exc}") from exc
        logger.warning("Загрузка на GPU не удалась (%s); пробую CPU.", exc)
        exc.__traceback__ = None
        clear_model_memory()
        device, quantized = torch.device("cpu"), False
        try:
            model = _load_model(model_class, model_path, device)
        except Exception as cpu_exc:
            raise OcrModelError(f"Не удалось загрузить OCR-модель на CPU: {cpu_exc}") from cpu_exc
    logger.info("Модель %s загружена: %s, dtype=%s, INT8=%s.",
                model_path.name, device, getattr(model, "dtype", "unknown"), quantized)
    return LoadedOcrModel(processor, model, device, quantized, model_path.resolve())


def resolve_device_and_quantization(
    *, allow_quantization: bool = False,
) -> tuple[Any, Any | None, bool]:
    """INT8 включается явно; скрытого CPU-offload нет."""
    torch = _import_torch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    quantization_config = None
    if device.type == "cuda" and allow_quantization:
        config_class = _import_bitsandbytes_config()
        if config_class is None:
            raise OcrModelError("Для INT8 нужен bitsandbytes. Отключите INT8 или установите зависимости.")
        quantization_config = config_class(load_in_8bit=True)
    return device, quantization_config, quantization_config is not None


def _load_model(
    model_class: Any, model_path: Path, device: Any, quantization_config: Any = None,
) -> Any:
    torch = _import_torch()
    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": dtype,
        "device_map": {"": str(device)},
        "attn_implementation": "sdpa",
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    model = model_class.from_pretrained(str(model_path), **kwargs)
    return model.eval()


class ModelPool:
    """Хранит одну модель; веб выполняет её генерацию в единственном worker."""

    def __init__(self, loader: Callable[..., LoadedOcrModel] | None = None) -> None:
        self._loader = loader
        self._lock = threading.Lock()
        self._key: tuple[Path, bool, bool] | None = None
        self._loaded: LoadedOcrModel | None = None

    def get(self, path: Path, quantization: bool, cpu_fallback: bool) -> LoadedOcrModel:
        key = (path.resolve(), quantization, cpu_fallback)
        with self._lock:
            if self._key == key and self._loaded is not None:
                logger.info("Используется уже загруженная модель %s.", path.name)
                return self._loaded
            self._release()
            # Ключ публикуется только после успешной загрузки.
            loaded = (self._loader or load_model_components)(path, quantization, cpu_fallback)
            self._loaded, self._key = loaded, key
            return loaded

    def clear(self) -> None:
        with self._lock:
            self._release()

    def _release(self) -> None:
        had_model = self._loaded is not None
        self._loaded, self._key = None, None
        if had_model:
            clear_model_memory()


@contextmanager
def inference_context() -> Iterator[None]:
    """Отключает градиенты и служебный учёт изменений тензоров."""
    with _import_torch().inference_mode():
        yield


def synchronize_device(device: object) -> None:
    """Дожидается GPU только на границах измеряемых стадий."""
    if str(device).startswith("cuda"):
        _import_torch().cuda.synchronize(device)


def clear_model_memory() -> None:
    """Очищает память после освобождения ссылки на прежнюю модель."""
    gc.collect()
    torch = _import_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise OcrModelError("Для работы OCR требуется torch.") from exc
    return torch


def _import_transformers_components() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise OcrModelError("Установите Transformers из requirements.txt для GLM/PaddleOCR.") from exc
    return AutoProcessor, AutoModelForImageTextToText


def _import_bitsandbytes_config() -> Any | None:
    if importlib.util.find_spec("bitsandbytes") is None:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        return None
    return BitsAndBytesConfig
