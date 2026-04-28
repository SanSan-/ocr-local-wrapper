from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ocr_local.exceptions import OcrModelError
from ocr_local.models import LoadedOcrModel

logger = logging.getLogger(__name__)


def load_model_components(
    model_path: Path,
    allow_quantization: bool,
    allow_cpu_fallback: bool,
) -> LoadedOcrModel:
    """Загружает процессор и GLM-OCR с учетом GPU и квантования."""
    torch = _import_torch()
    processor_class, model_class = _import_transformers_components()
    device, quantization_config, quantized = resolve_device_and_quantization(
        allow_quantization=allow_quantization
    )

    try:
        processor = processor_class.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
    except Exception as exc:
        raise OcrModelError(_processor_error_message(exc)) from exc

    try:
        model = _load_model(model_class, model_path, device, quantization_config)
    except Exception as exc:
        if allow_cpu_fallback and getattr(device, "type", "") == "cuda":
            logger.warning("Загрузка на GPU не удалась, пробую CPU.")
            device = torch.device("cpu")
            quantized = False
            try:
                model = _load_model_on_cpu(model_class, model_path, device)
            except Exception as cpu_exc:
                raise OcrModelError("Не удалось загрузить модель GLM-OCR на CPU.") from cpu_exc
        else:
            raise OcrModelError("Не удалось загрузить модель GLM-OCR.") from exc

    model.eval()
    return LoadedOcrModel(
        processor=processor,
        model=model,
        device=device,
        quantized=quantized,
    )


def resolve_device_and_quantization(
    *,
    allow_quantization: bool = True,
) -> tuple[Any, Any | None, bool]:
    """Выбирает устройство и опциональную 8-битную квантовку."""
    torch = _import_torch()
    device_is_gpu = bool(torch.cuda.is_available())
    quantization_config = None
    quantized = False

    if device_is_gpu and allow_quantization:
        bitsandbytes_config = _import_bitsandbytes_config()
        if bitsandbytes_config is not None:
            try:
                quantization_config = bitsandbytes_config(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
            except TypeError:
                quantization_config = bitsandbytes_config(load_in_8bit=True)
            quantized = True
        else:
            logger.warning(
                "BitsAndBytesConfig недоступен - модель загружается без 8-битной квантовки."
            )

    if device_is_gpu:
        device = torch.device("cuda")
        logger.info("OCR-модель загружается в видеопамять GPU.")
    else:
        device = torch.device("cpu")
        logger.info("OCR-модель загружается в оперативную память CPU.")
    return device, quantization_config, quantized


@contextmanager
def inference_context() -> Iterator[None]:
    """Отключает расчет градиентов на время инференса."""
    try:
        import torch
    except ImportError:
        yield
        return

    with torch.no_grad():
        yield


def clear_model_memory() -> None:
    """Очищает память Python и CUDA после выгрузки модели."""
    gc.collect()
    try:
        torch = _import_torch()
    except OcrModelError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _load_model(
    model_class: Any,
    model_path: Path,
    device: Any,
    quantization_config: Any | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": "auto",
    }
    if getattr(device, "type", "") == "cuda" or quantization_config is not None:
        kwargs["device_map"] = "auto"
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    model = model_class.from_pretrained(str(model_path), **kwargs)
    if getattr(device, "type", "") == "cuda" or quantization_config is not None:
        return model
    move = getattr(model, "to", None)
    if callable(move):
        move(device)
    return model


def _load_model_on_cpu(model_class: Any, model_path: Path, device: Any) -> Any:
    model = model_class.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype="auto",
    )
    move = getattr(model, "to", None)
    if callable(move):
        move(device)
    return model


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
        raise OcrModelError(
            "Текущая версия transformers не поддерживает GLM-OCR. "
            "Установите git-версию из requirements.txt."
        ) from exc
    return AutoProcessor, AutoModelForImageTextToText


def _import_bitsandbytes_config() -> Any | None:
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        return None
    return BitsAndBytesConfig


def _processor_error_message(exc: Exception) -> str:
    message = str(exc)
    if "Unrecognized processing class" in message or "Glm46VProcessor" in message:
        return (
            "Не удалось загрузить процессор GLM-OCR. "
            "Установленная версия transformers не поддерживает Glm46VProcessor; "
            "установите зависимости из requirements.txt."
        )
    return "Не удалось загрузить процессор GLM-OCR."
