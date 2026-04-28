from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ocr_local import __version__
from ocr_local.constants import DEFAULT_MAX_NEW_TOKENS, DEFAULT_PROMPT
from ocr_local.exceptions import OcrError
from ocr_local.models import OcrOptions
from ocr_local.service import run_ocr
from ocr_local.utils.env_utils import get_default_model_path, load_environment
from ocr_local.utils.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер."""
    parser = argparse.ArgumentParser(
        prog="ocr-local",
        description="Локальное OCR изображения через GLM-OCR.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Путь к входному изображению.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Путь к выходному .txt. По умолчанию создается рядом с изображением.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=get_default_model_path(),
        help="Путь к локальной модели GLM-OCR.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Промпт для OCR-модели.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Лимит новых токенов генерации.",
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Отключить 8-битную квантовку.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Разрешить переход на CPU при ошибке загрузки на GPU.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующий выходной файл.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ocr-local {__version__}",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> OcrOptions:
    """Разбирает аргументы командной строки."""
    load_environment()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens должен быть положительным числом.")
    return OcrOptions(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        quantization_enabled=not args.no_quantization,
        allow_cpu_fallback=args.allow_cpu_fallback,
        force=args.force,
        verbose=args.verbose,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI."""
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

    if result.cached:
        print(f"OCR взят из кеша: {result.output_path}")
    else:
        print(f"OCR завершен: {result.output_path}")
    return 0
