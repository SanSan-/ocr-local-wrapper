from __future__ import annotations

from pathlib import Path

from ocr_local.cli import parse_args


def test_parse_args_uses_env_model_path(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    monkeypatch.setenv("OCR_MODEL_PATH", str(model_path))

    options = parse_args(
        [
            "--input",
            "image.jpg",
            "--max-new-tokens",
            "128",
            "--no-quantization",
            "--allow-cpu-fallback",
        ]
    )

    assert options.input_path == Path("image.jpg")
    assert options.model_path == model_path
    assert options.max_new_tokens == 128
    assert options.quantization_enabled is False
    assert options.allow_cpu_fallback is True

