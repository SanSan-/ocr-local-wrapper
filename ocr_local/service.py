from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from ocr_local.exceptions import OcrModelError
from ocr_local.models import LoadedOcrModel, OcrOptions, OcrResult
from ocr_local.utils.io_utils import (
    build_output_path,
    ensure_output_can_be_written,
    validate_image_file,
    validate_model_path,
    write_text_utf8_no_bom,
)
from ocr_local.utils.model_utils import inference_context, load_model_components
from ocr_local.utils.output_cache import apply_output_cache, update_output_cache

logger = logging.getLogger(__name__)

ModelLoader = Callable[[Path, bool, bool], LoadedOcrModel]

SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|begin_of_image|>",
    "<|end_of_image|>",
    "<|image|>",
    "<|begin_of_video|>",
    "<|end_of_video|>",
    "<|video|>",
)


def run_ocr(
    options: OcrOptions,
    *,
    loader: ModelLoader = load_model_components,
) -> OcrResult:
    """Запускает полный OCR-пайплайн и записывает результат в файл."""
    input_path = validate_image_file(options.input_path)
    output_path = build_output_path(input_path, options.output_path)

    if not options.force:
        cached_text = apply_output_cache(input_path, output_path, logger)
        if cached_text is not None:
            return OcrResult(
                input_path=input_path,
                output_path=output_path,
                text=cached_text,
                device="cache",
                quantized=False,
                cached=True,
            )

    model_path = validate_model_path(options.model_path)
    ensure_output_can_be_written(output_path, options.force)

    loaded = loader(
        model_path,
        options.quantization_enabled,
        options.allow_cpu_fallback,
    )
    text = recognize_image_text(input_path, options, loaded)
    write_text_utf8_no_bom(output_path, text)
    update_output_cache(input_path, output_path, text, logger)

    return OcrResult(
        input_path=input_path,
        output_path=output_path,
        text=text,
        device=_device_label(loaded.device),
        quantized=loaded.quantized,
        cached=False,
    )


def recognize_image_text(
    image_path: Path,
    options: OcrOptions,
    loaded: LoadedOcrModel,
) -> str:
    """Распознает текст на изображении через уже загруженную модель."""
    processor = loaded.processor
    model = loaded.model
    messages = _build_messages(image_path, options.prompt)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = _move_inputs_to_device(inputs, model, loaded.device)
    if isinstance(inputs, dict):
        inputs.pop("token_type_ids", None)

    input_length = _input_token_length(inputs)
    with inference_context():
        generated_ids = model.generate(**inputs, max_new_tokens=options.max_new_tokens)

    output_tokens = _slice_generated_tokens(generated_ids, input_length)
    output_text = _decode_output(processor, output_tokens)
    cleaned = clean_output_text(output_text)
    if not cleaned:
        raise OcrModelError("Модель не вернула распознанный текст.")
    return cleaned


def clean_output_text(text: str) -> str:
    """Удаляет служебные токены модели и лишние пробелы по краям."""
    cleaned = text
    for token in SPECIAL_TOKENS:
        cleaned = cleaned.replace(token, "")
    return cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()


def _build_messages(image_path: Path, prompt: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _move_inputs_to_device(inputs: object, model: object, fallback_device: object) -> object:
    target_device = getattr(model, "device", None) or fallback_device
    move = getattr(inputs, "to", None)
    if callable(move):
        return move(target_device)
    return inputs


def _input_token_length(inputs: object) -> int:
    try:
        return int(inputs["input_ids"].shape[1])  # type: ignore[index]
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise OcrModelError("Не удалось определить длину входного промпта модели.") from exc


def _slice_generated_tokens(generated_ids: object, input_length: int) -> object:
    try:
        return generated_ids[0][input_length:]  # type: ignore[index]
    except (TypeError, IndexError) as exc:
        raise OcrModelError("Не удалось отделить ответ модели от входного промпта.") from exc


def _decode_output(processor: object, output_tokens: object) -> str:
    decode = getattr(processor, "decode", None)
    if not callable(decode):
        raise OcrModelError("Процессор модели не поддерживает decode.")
    try:
        return str(decode(output_tokens, skip_special_tokens=True))
    except TypeError:
        return str(decode(output_tokens))


def _device_label(device: object) -> str:
    device_type = getattr(device, "type", None)
    if device_type:
        return str(device_type)
    return str(device)
