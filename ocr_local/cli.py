from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ocr_local import __version__
from ocr_local.exceptions import OcrError
from ocr_local.model_profiles import PROFILES, resolve_options
from ocr_local.models import OcrOptions
from ocr_local.service import run_ocr
from ocr_local.utils.env_utils import load_environment
from ocr_local.utils.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер; defaults берутся из выбранного профиля."""
    parser = argparse.ArgumentParser(prog="ocr-local", description="Локальное OCR через GLM-OCR или PaddleOCR-VL-1.6.")
    parser.add_argument("--input", required=True, type=Path, help="Путь к входному изображению.")
    parser.add_argument("--output", type=Path, help="Выходной .txt; по умолчанию рядом с изображением.")
    parser.add_argument("--model", choices=tuple(PROFILES), help="Профиль модели; иначе определяется по пути.")
    parser.add_argument("--model-path", type=Path, help="Явный локальный каталог модели.")
    parser.add_argument("--prompt", help="Промпт; по умолчанию из профиля модели.")
    parser.add_argument("--max-new-tokens", type=int, help="Предел новых токенов; по умолчанию из профиля.")
    parser.add_argument("--max-pixels", type=int, help="Предел площади подготовленного изображения.")
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--quantization", action="store_true", help="Включить INT8 на GPU (экономит память).")
    quantization.add_argument("--no-quantization", action="store_true", help="BF16/FP16 на GPU (режим по умолчанию).")
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="Разрешить CPU после ошибки загрузки GPU.")
    parser.add_argument("--force", action="store_true", help="Обойти кеш и перезаписать выбранный TXT.")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод.")
    parser.add_argument("--version", action="version", version=f"ocr-local {__version__}")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> OcrOptions:
    load_environment()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return resolve_options(OcrOptions(
            input_path=args.input, output_path=args.output, model_path=args.model_path,
            model_id=args.model, prompt=args.prompt, max_new_tokens=args.max_new_tokens,
            max_pixels=args.max_pixels, quantization_enabled=args.quantization,
            allow_cpu_fallback=args.allow_cpu_fallback, force=args.force, verbose=args.verbose,
        ))
    except OcrError as exc:
        parser.error(str(exc))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        options = parse_args(argv)
        setup_logging(options.verbose)
        result = run_ocr(options)
    except OcrError as exc:
        print(f"Ошибка OCR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("OCR прерван пользователем.", file=sys.stderr)
        return 130
    status = "OCR взят из кеша" if result.cached else "OCR завершён"
    print(f"{status} ({result.model_id}): {result.output_path}")
    if result.metrics.limit_reached:
        print("Достигнут лимит токенов; увеличьте --max-new-tokens.", file=sys.stderr)
    return 0
