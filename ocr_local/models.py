from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocr_local.constants import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_PATH, DEFAULT_PROMPT


@dataclass(frozen=True)
class OcrOptions:
    """Настройки запуска OCR."""

    input_path: Path
    output_path: Path | None = None
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    prompt: str = DEFAULT_PROMPT
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    quantization_enabled: bool = True
    allow_cpu_fallback: bool = False
    force: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class LoadedOcrModel:
    """Загруженные компоненты OCR-модели."""

    processor: Any
    model: Any
    device: Any
    quantized: bool


@dataclass(frozen=True)
class OcrResult:
    """Результат OCR."""

    input_path: Path
    output_path: Path
    text: str
    device: str
    quantized: bool
    cached: bool = False
