from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocr_local.cli import parse_args
from ocr_local.exceptions import OcrValidationError
from ocr_local.model_profiles import model_catalog, resolve_options
from ocr_local.models import OcrOptions


@pytest.mark.parametrize("model_id,prompt,tokens,pixels", [
    ("glm-ocr", "Text Recognition:", 8192, 2007040),
    ("paddleocr-vl-1.6", "OCR:", 2048, 1003520),
])
def test_cli_and_service_share_profile(model_id, prompt, tokens, pixels):
    options = parse_args(["--input", "image.png", "--model", model_id])
    direct = resolve_options(OcrOptions(input_path=Path("image.png"), model_id=model_id))
    assert options == direct
    assert (options.prompt, options.max_new_tokens, options.max_pixels) == (prompt, tokens, pixels)
    assert options.quantization_enabled is False


def test_catalog_uses_configured_model_and_sibling(tmp_path, monkeypatch):
    paddle = tmp_path / "PaddleOCR-VL-1.6"
    paddle.mkdir()
    (paddle / "config.json").write_text('{"model_type":"paddleocr_vl"}', encoding="utf-8")
    monkeypatch.setenv("OCR_MODEL_PATH", str(paddle))
    catalog = model_catalog()
    assert catalog["default_model_id"] == "paddleocr-vl-1.6"
    assert Path(catalog["models"][0]["path"]) == tmp_path / "GLM-OCR"
    assert Path(catalog["models"][1]["path"]) == paddle
    assert not catalog["models"][1]["available"]


def test_explicit_path_and_parameters_win(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_PADDLE_MODEL_PATH", str(tmp_path / "other"))
    options = parse_args([
        "--input", "a.png", "--model", "paddleocr-vl-1.6", "--model-path", str(tmp_path),
        "--prompt", "Custom OCR:", "--max-new-tokens", "300", "--max-pixels", "400000",
    ])
    assert options.model_path == tmp_path
    assert (options.prompt, options.max_new_tokens, options.max_pixels) == ("Custom OCR:", 300, 400000)


def test_rejects_model_path_mismatch(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "paddleocr_vl"}), encoding="utf-8")
    with pytest.raises(OcrValidationError, match="не соответствует"):
        resolve_options(OcrOptions(Path("a.png"), model_id="glm-ocr", model_path=tmp_path))


@pytest.mark.parametrize("args", [
    ["--max-new-tokens", "0"], ["--max-new-tokens", "32769"],
    ["--max-pixels", "1"], ["--max-pixels", "16777217"], ["--prompt", " "],
])
def test_cli_rejects_invalid_parameters(args):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--input", "a.png", *args])
    assert exc.value.code == 2
