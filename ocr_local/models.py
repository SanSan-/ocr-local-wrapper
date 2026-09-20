from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OcrOptions:
    """Настройки запуска; пропущенные параметры берутся из профиля."""

    input_path: Path
    output_path: Path | None = None
    model_path: Path | None = None
    model_id: str | None = None
    prompt: str | None = None
    max_new_tokens: int | None = None
    max_pixels: int | None = None
    quantization_enabled: bool = False
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
    model_path: Path | None = None


@dataclass
class OcrMetrics:
    """Измеренные стадии OCR в секундах."""

    load_seconds: float = 0.0
    prepare_seconds: float = 0.0
    generate_seconds: float = 0.0
    total_seconds: float = 0.0
    generated_tokens: int = 0
    tokens_per_second: float = 0.0
    limit_reached: bool = False


@dataclass(frozen=True)
class OcrResult:
    """Результат OCR с настройками и измерениями."""

    input_path: Path
    output_path: Path
    text: str
    device: str
    quantized: bool
    cached: bool = False
    model_id: str = ""
    settings: dict[str, object] = field(default_factory=dict)
    metrics: OcrMetrics = field(default_factory=OcrMetrics)
