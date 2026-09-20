from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ocr_local.exceptions import OcrValidationError
from ocr_local.models import OcrOptions
from ocr_local.utils.env_utils import get_default_model_path

MAX_NEW_TOKENS = 32768
MAX_PIXELS = 16777216


@dataclass(frozen=True)
class ModelProfile:
    """Общие настройки модели для CLI, веба и сервиса."""

    id: str
    name: str
    model_type: str
    path_env: str
    prompt: str
    max_new_tokens: int
    max_pixels: int
    min_pixels: int


PROFILES = {
    "glm-ocr": ModelProfile(
        "glm-ocr", "GLM-OCR", "glm_ocr", "OCR_GLM_MODEL_PATH",
        "Text Recognition:", 8192, 2007040, 12544,
    ),
    "paddleocr-vl-1.6": ModelProfile(
        "paddleocr-vl-1.6", "PaddleOCR-VL-1.6", "paddleocr_vl", "OCR_PADDLE_MODEL_PATH",
        "OCR:", 2048, 1003520, 112896,
    ),
}


def get_profile(model_id: str) -> ModelProfile:
    try:
        return PROFILES[model_id]
    except KeyError as exc:
        raise OcrValidationError(f"Неизвестная OCR-модель: {model_id}") from exc


def detect_model_id(path: Path) -> str:
    """Определяет семейство без загрузки весов, в том числе по имени отсутствующего каталога."""
    config_path = path / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model_type = config.get("model_type")
        except (OSError, ValueError, AttributeError) as exc:
            raise OcrValidationError(f"Не удалось прочитать config.json модели: {path}") from exc
        if model_type:
            for profile in PROFILES.values():
                if profile.model_type == model_type:
                    return profile.id
            raise OcrValidationError(f"Неподдерживаемый model_type: {model_type}")
    return "paddleocr-vl-1.6" if "paddleocr" in path.name.lower() else "glm-ocr"


def get_model_path(model_id: str) -> Path:
    profile = get_profile(model_id)
    explicit = os.getenv(profile.path_env)
    if explicit:
        return Path(explicit)
    configured = get_default_model_path()
    if detect_model_id(configured) == model_id:
        return configured
    if os.getenv("OCR_MODEL_PATH"):
        return configured.parent / profile.name
    return Path("models") / profile.name


def resolve_options(options: OcrOptions) -> OcrOptions:
    """Подставляет профиль и проверяет параметры до кеша и загрузки модели."""
    model_id = options.model_id or detect_model_id(options.model_path or get_default_model_path())
    profile = get_profile(model_id)
    path = (options.model_path or get_model_path(model_id)).expanduser().resolve()
    if (path / "config.json").is_file() and detect_model_id(path) != model_id:
        raise OcrValidationError(f"Каталог {path} не соответствует модели {profile.name}.")
    prompt = profile.prompt if options.prompt is None else options.prompt
    tokens = profile.max_new_tokens if options.max_new_tokens is None else options.max_new_tokens
    pixels = profile.max_pixels if options.max_pixels is None else options.max_pixels
    if not isinstance(prompt, str) or not prompt.strip():
        raise OcrValidationError("Промпт не должен быть пустым.")
    if type(tokens) is not int or not 1 <= tokens <= MAX_NEW_TOKENS:
        raise OcrValidationError(f"max-new-tokens должен быть от 1 до {MAX_NEW_TOKENS}.")
    if type(pixels) is not int or not profile.min_pixels <= pixels <= MAX_PIXELS:
        raise OcrValidationError(f"max-pixels должен быть от {profile.min_pixels} до {MAX_PIXELS}.")
    return replace(options, model_id=model_id, model_path=path, prompt=prompt,
                   max_new_tokens=tokens, max_pixels=pixels)


def settings_identity(options: OcrOptions) -> dict[str, object]:
    """Идентичность результата; options уже разрешены через resolve_options."""
    return {
        "version": 2,
        "model_id": options.model_id,
        "model_path": str(options.model_path).casefold(),
        "prompt": options.prompt,
        "max_new_tokens": options.max_new_tokens,
        "max_pixels": options.max_pixels,
        "quantization_enabled": options.quantization_enabled,
    }


def settings_fingerprint(options: OcrOptions) -> str:
    payload = json.dumps(settings_identity(options), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_catalog() -> dict[str, object]:
    """Возвращает доступность и defaults без импорта Torch и загрузки весов."""
    items = []
    for profile in PROFILES.values():
        path = get_model_path(profile.id).expanduser().resolve()
        required = ("config.json", "model.safetensors", "tokenizer.json")
        available = all((path / name).is_file() for name in required)
        item = asdict(profile)
        item.update(path=str(path), available=available)
        items.append(item)
    return {
        "default_model_id": detect_model_id(get_default_model_path()),
        "models": items, "max_new_tokens_limit": MAX_NEW_TOKENS, "max_pixels_limit": MAX_PIXELS,
    }
