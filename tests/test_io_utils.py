from __future__ import annotations

import pytest

from ocr_local.exceptions import OcrValidationError
from ocr_local.utils.io_utils import (
    build_output_path,
    ensure_output_can_be_written,
    has_utf8_bom,
    validate_image_file,
    write_text_utf8_no_bom,
)


def test_build_output_path_defaults_to_txt(tmp_path):
    image_path = tmp_path / "sample.jpg"

    output_path = build_output_path(image_path, None)

    assert output_path == tmp_path / "sample.txt"


def test_build_output_path_rejects_non_txt(tmp_path):
    image_path = tmp_path / "sample.jpg"

    with pytest.raises(OcrValidationError):
        build_output_path(image_path, tmp_path / "sample.md")


def test_ensure_output_rejects_existing_without_force(tmp_path):
    output_path = tmp_path / "sample.txt"
    output_path.write_text("старый текст", encoding="utf-8")

    with pytest.raises(OcrValidationError):
        ensure_output_can_be_written(output_path, force=False)


def test_write_text_utf8_without_bom(tmp_path):
    output_path = tmp_path / "result.txt"

    write_text_utf8_no_bom(output_path, "ПОРОШОК\nНЕ ВХОДИ")

    assert output_path.read_text(encoding="utf-8") == "ПОРОШОК\nНЕ ВХОДИ"
    assert has_utf8_bom(output_path) is False


def test_validate_image_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("не изображение", encoding="utf-8")

    with pytest.raises(OcrValidationError):
        validate_image_file(path)

