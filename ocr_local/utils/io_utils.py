from __future__ import annotations

from pathlib import Path

from ocr_local.constants import SUPPORTED_IMAGE_EXTENSIONS
from ocr_local.exceptions import OcrValidationError


def validate_image_file(path: Path) -> Path:
    """Проверяет, что путь указывает на поддерживаемое изображение."""
    resolved = _resolve_existing_path(path, "Входное изображение")
    if not resolved.is_file():
        raise OcrValidationError(f"Входной путь не является файлом: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise OcrValidationError(
            f"Неподдерживаемое расширение изображения '{resolved.suffix}'. "
            f"Поддерживаются: {supported}."
        )
    _verify_image_content(resolved)
    return resolved


def validate_model_path(path: Path) -> Path:
    """Проверяет локальный каталог модели."""
    resolved = _resolve_existing_path(path, "Каталог модели")
    if not resolved.is_dir():
        raise OcrValidationError(f"Путь модели не является каталогом: {resolved}")
    required_files = ("config.json", "model.safetensors", "tokenizer.json")
    missing = [name for name in required_files if not (resolved / name).exists()]
    if missing:
        raise OcrValidationError(
            f"В каталоге модели отсутствуют обязательные файлы: {', '.join(missing)}"
        )
    return resolved


def build_output_path(input_path: Path, output_path: Path | None) -> Path:
    """Формирует путь выходного .txt."""
    if output_path is None:
        return input_path.with_suffix(".txt")
    resolved = output_path.expanduser().resolve()
    if resolved.suffix.lower() != ".txt":
        raise OcrValidationError("Выходной файл должен иметь расширение .txt.")
    return resolved


def ensure_output_can_be_written(output_path: Path, force: bool) -> None:
    """Проверяет, можно ли записать выходной файл."""
    if output_path.exists() and output_path.is_dir():
        raise OcrValidationError(f"Путь вывода указывает на каталог: {output_path}")
    if output_path.exists() and not force:
        raise OcrValidationError(
            f"Выходной файл уже существует: {output_path}. Используйте --force."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)


def write_text_utf8_no_bom(path: Path, text: str) -> None:
    """Записывает текст в UTF-8 без BOM."""
    path.write_text(text, encoding="utf-8", newline="")


def read_text_utf8(path: Path) -> str:
    """Читает текст как UTF-8."""
    return path.read_text(encoding="utf-8")


def has_utf8_bom(path: Path) -> bool:
    """Проверяет наличие UTF-8 BOM."""
    with path.open("rb") as file:
        return file.read(3) == b"\xef\xbb\xbf"


def _resolve_existing_path(path: Path, label: str) -> Path:
    try:
        return path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise OcrValidationError(f"{label} не найден: {path}") from exc


def _verify_image_content(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OcrValidationError("Для проверки изображений требуется pillow.") from exc

    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise OcrValidationError(f"Файл не читается как изображение: {path}") from exc
