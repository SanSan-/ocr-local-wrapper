from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from ocr_local.exceptions import OcrModelError
from ocr_local.model_profiles import get_profile, resolve_options, settings_identity
from ocr_local.models import LoadedOcrModel, OcrMetrics, OcrOptions, OcrResult
from ocr_local.utils.io_utils import (
    build_output_path, ensure_output_can_be_written, validate_image_file,
    validate_model_path, write_text_utf8_no_bom,
)
from ocr_local.utils.model_utils import (
    inference_context, load_model_components, synchronize_device,
)
from ocr_local.utils.output_cache import apply_output_cache, update_output_cache

logger = logging.getLogger(__name__)
ModelLoader = Callable[[Path, bool, bool], LoadedOcrModel]
SPECIAL_TOKENS = (
    "<|endoftext|>", "<|begin_of_image|>", "<|end_of_image|>", "<|image|>",
    "<|begin_of_video|>", "<|end_of_video|>", "<|video|>",
)


def run_ocr(
    options: OcrOptions, *, loaded: LoadedOcrModel | None = None,
    loader: ModelLoader = load_model_components,
) -> OcrResult:
    """Выполняет общий OCR: валидация → профильный кеш → модель → TXT."""
    started = time.perf_counter()
    options = resolve_options(options)
    metrics = OcrMetrics()
    input_path = validate_image_file(options.input_path)
    output_path = build_output_path(input_path, options.output_path)
    settings = settings_identity(options)
    if not options.force:
        text = apply_output_cache(input_path, output_path, logger, settings=settings, metrics=metrics)
        if text is not None:
            metrics.total_seconds = time.perf_counter() - started
            return OcrResult(input_path, output_path, text, "cache", False, True,
                             options.model_id, settings, metrics)
    ensure_output_can_be_written(output_path, options.force)
    load_started = time.perf_counter()
    if loaded is None:
        loaded = loader(validate_model_path(options.model_path),
                        options.quantization_enabled, options.allow_cpu_fallback)
    elif loaded.model_path is not None and loaded.model_path.resolve() != options.model_path:
        raise OcrModelError("Загруженная модель не соответствует выбранному каталогу.")
    synchronize_device(loaded.device)
    metrics.load_seconds = time.perf_counter() - load_started
    text = recognize_image_text(input_path, options, loaded, metrics=metrics)
    write_text_utf8_no_bom(output_path, text)
    update_output_cache(input_path, output_path, text, logger, settings=settings, limit_reached=metrics.limit_reached)
    metrics.total_seconds = time.perf_counter() - started
    logger.info("Время: загрузка %.3f с, подготовка %.3f с, генерация %.3f с; %d токенов, %.1f токен/с.",
                metrics.load_seconds, metrics.prepare_seconds, metrics.generate_seconds,
                metrics.generated_tokens, metrics.tokens_per_second)
    return OcrResult(input_path, output_path, text, _device_label(loaded.device),
                     loaded.quantized, False, options.model_id, settings, metrics)


def recognize_image_text(
    image_path: Path, options: OcrOptions, loaded: LoadedOcrModel, *, metrics: OcrMetrics | None = None,
) -> str:
    """Распознаёт изображение, сохраняя KV-кеш между шагами генерации."""
    options = resolve_options(options)
    metrics = metrics if metrics is not None else OcrMetrics()
    processor, model = loaded.processor, loaded.model
    profile = get_profile(options.model_id)
    started = time.perf_counter()
    inputs = processor.apply_chat_template(
        _build_messages(image_path, options.prompt),
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        processor_kwargs={"images_kwargs": {"size": {"shortest_edge": profile.min_pixels, "longest_edge": options.max_pixels}}},
    )
    inputs = _move_inputs_to_device(inputs, model, loaded.device)
    inputs.pop("token_type_ids", None)
    input_length = _input_token_length(inputs)
    synchronize_device(loaded.device)
    metrics.prepare_seconds = time.perf_counter() - started
    started = time.perf_counter()
    try:
        with inference_context():
            generated_ids = model.generate(
                **inputs, max_new_tokens=options.max_new_tokens,
                use_cache=True, do_sample=False, num_beams=1,
            )
        synchronize_device(loaded.device)
    except Exception as exc:
        raise OcrModelError(
            f"Ошибка генерации: {exc}. При нехватке памяти уменьшите max_pixels/max_new_tokens."
        ) from exc
    metrics.generate_seconds = time.perf_counter() - started
    output_tokens = _slice_generated_tokens(generated_ids, input_length)
    metrics.generated_tokens = len(output_tokens)
    metrics.tokens_per_second = metrics.generated_tokens / max(metrics.generate_seconds, 1e-9)
    metrics.limit_reached = _limit_reached(output_tokens, options.max_new_tokens, model)
    if metrics.limit_reached:
        logger.warning("Достигнут лимит %d новых токенов; текст может быть обрезан.", options.max_new_tokens)
    cleaned = clean_output_text(_decode_output(processor, output_tokens))
    if not cleaned:
        raise OcrModelError("Модель не вернула распознанный текст.")
    return cleaned


def _limit_reached(tokens: object, limit: int, model: object) -> bool:
    if len(tokens) < limit:
        return False
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    last = tokens[-1]
    if hasattr(last, "item"):
        last = last.item()
    return last not in (eos if isinstance(eos, (list, tuple)) else [eos])


def clean_output_text(text: str) -> str:
    """Удаляет служебные токены и крайние пробелы."""
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _build_messages(image_path: Path, prompt: str) -> list[dict[str, object]]:
    return [{"role": "user", "content": [
        {"type": "image", "url": str(image_path)}, {"type": "text", "text": prompt},
    ]}]


def _move_inputs_to_device(inputs: object, model: object, fallback_device: object) -> object:
    target = getattr(model, "device", None) or fallback_device
    move = getattr(inputs, "to", None)
    if callable(move):
        inputs = move(target)
    pixels = inputs.get("pixel_values")
    dtype = getattr(model, "dtype", None)
    if pixels is not None and dtype is not None:
        inputs["pixel_values"] = pixels.to(dtype=dtype)
    return inputs


def _input_token_length(inputs: object) -> int:
    try:
        return int(inputs["input_ids"].shape[1])
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise OcrModelError("Не удалось определить длину входного промпта модели.") from exc


def _slice_generated_tokens(generated_ids: object, input_length: int) -> object:
    try:
        return generated_ids[0][input_length:]
    except (TypeError, IndexError) as exc:
        raise OcrModelError("Не удалось отделить ответ модели от входного промпта.") from exc


def _decode_output(processor: object, tokens: object) -> str:
    decode = getattr(processor, "decode", None)
    if not callable(decode):
        raise OcrModelError("Процессор модели не поддерживает decode.")
    return str(decode(tokens, skip_special_tokens=True))


def _device_label(device: object) -> str:
    return str(getattr(device, "type", None) or device)
