from __future__ import annotations

import os
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from ocr_local.constants import DEFAULT_MODEL_PATH
from ocr_local.models import LoadedOcrModel, OcrOptions
from ocr_local.service import run_ocr
from ocr_local.utils import output_cache
from ocr_local.utils.io_utils import has_utf8_bom

ATTACHED_TEXT = "ПОРОШОК\nНЕ ВХОДИ"


class FakeInputIds:
    shape = (1, 3)


class FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=FakeInputIds(), token_type_ids=object())
        self.device = None

    def to(self, device: object) -> "FakeInputs":
        self.device = device
        return self


class FakeGeneratedRow:
    def __getitem__(self, key: object) -> list[str]:
        if isinstance(key, slice):
            return ["decoded"]
        raise IndexError(key)


class FakeProcessor:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] | None = None

    def apply_chat_template(self, messages: list[dict[str, object]], **_: object) -> FakeInputs:
        self.messages = messages
        return FakeInputs()

    def decode(self, tokens: object, skip_special_tokens: bool = True) -> str:
        assert tokens == ["decoded"]
        assert skip_special_tokens is True
        return f"{ATTACHED_TEXT}<|endoftext|>"


class FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.generate_kwargs: dict[str, object] | None = None

    def generate(self, **kwargs: object) -> list[FakeGeneratedRow]:
        self.generate_kwargs = kwargs
        assert "token_type_ids" not in kwargs
        assert kwargs["max_new_tokens"] == 128
        return [FakeGeneratedRow()]


def test_run_ocr_processes_attached_text_image(tmp_path, monkeypatch):
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", tmp_path / "ocr_cache.json")
    image_path = _create_attached_like_image(tmp_path / "attached_text.jpg")
    model_path = _create_fake_model_dir(tmp_path / "model")
    output_path = tmp_path / "result.txt"
    processor = FakeProcessor()
    model = FakeModel()

    def fake_loader(path: Path, allow_quantization: bool, allow_cpu_fallback: bool) -> LoadedOcrModel:
        assert path == model_path
        assert allow_quantization is True
        assert allow_cpu_fallback is False
        return LoadedOcrModel(processor=processor, model=model, device="cpu", quantized=False)

    result = run_ocr(
        OcrOptions(
            input_path=image_path,
            output_path=output_path,
            model_path=model_path,
            max_new_tokens=128,
        ),
        loader=fake_loader,
    )

    assert result.text == ATTACHED_TEXT
    assert output_path.read_text(encoding="utf-8") == ATTACHED_TEXT
    assert has_utf8_bom(output_path) is False
    assert processor.messages is not None
    content = processor.messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["url"] == str(image_path)


@pytest.mark.integration
def test_real_glm_ocr_reads_attached_text_image(tmp_path, monkeypatch):
    if os.getenv("RUN_GLM_OCR_TEST") != "1":
        pytest.skip("Реальный GLM-OCR-тест запускается только при RUN_GLM_OCR_TEST=1.")
    model_path = Path(os.getenv("OCR_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.exists():
        pytest.skip(f"Локальная модель не найдена: {model_path}")
    _skip_if_transformers_lacks_glm_ocr_processor(model_path)
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", tmp_path / "ocr_cache.json")

    image_path = _create_attached_like_image(tmp_path / "attached_text.jpg")
    output_path = tmp_path / "attached_text.txt"

    result = run_ocr(
        OcrOptions(
            input_path=image_path,
            output_path=output_path,
            model_path=model_path,
            allow_cpu_fallback=True,
            force=True,
        )
    )

    normalized = " ".join(result.text.upper().split())
    assert "ПОРОШ" in normalized
    assert "ВХОД" in normalized
    assert has_utf8_bom(output_path) is False


def _create_attached_like_image(path: Path) -> Path:
    image = Image.new("RGB", (800, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 106, 799, 372), fill=(255, 0, 0))
    font = _load_cyrillic_font(92)
    lines = ATTACHED_TEXT.splitlines()
    y = 146
    for line in lines:
        draw.text((52, y), line, fill="white", font=font)
        y += 104
    image.save(path, format="JPEG", quality=95)
    return path


def _load_cyrillic_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\ARIALUNI.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    pytest.skip("Не найден шрифт с поддержкой кириллицы для тестовой картинки.")


def _create_fake_model_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"")
    return path


def _skip_if_transformers_lacks_glm_ocr_processor(model_path: Path) -> None:
    preprocessor_path = model_path / "preprocessor_config.json"
    if not preprocessor_path.exists():
        return
    data = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    processor_class = data.get("processor_class")
    if not processor_class:
        return
    import transformers

    if not hasattr(transformers, str(processor_class)):
        pytest.skip(
            f"Установленный transformers {transformers.__version__} "
            f"не содержит {processor_class}; установите requirements.txt."
        )
