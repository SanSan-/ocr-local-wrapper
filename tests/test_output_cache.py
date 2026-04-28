from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from ocr_local.models import LoadedOcrModel, OcrOptions
from ocr_local.service import run_ocr
from ocr_local.utils import output_cache
from ocr_local.utils.io_utils import has_utf8_bom


class FakeInputIds:
    shape = (1, 2)


class FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=FakeInputIds(), token_type_ids=object())

    def to(self, _: object) -> "FakeInputs":
        return self


class FakeGeneratedRow:
    def __getitem__(self, key: object) -> list[str]:
        if isinstance(key, slice):
            return ["token"]
        raise IndexError(key)


class FakeProcessor:
    def __init__(self, text: str) -> None:
        self._text = text

    def apply_chat_template(self, *_: object, **__: object) -> FakeInputs:
        return FakeInputs()

    def decode(self, _: object, skip_special_tokens: bool = True) -> str:
        return self._text


class FakeModel:
    device = "cpu"

    def generate(self, **_: object) -> list[FakeGeneratedRow]:
        return [FakeGeneratedRow()]


def test_run_ocr_restores_from_cache_without_model_load(tmp_path, monkeypatch):
    cache_path = tmp_path / "ocr_cache.json"
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", cache_path)
    image_path = _create_image(tmp_path / "input.jpg")
    output_path = tmp_path / "nested" / "result.txt"
    cached_text = "Текст из кеша"
    output_cache.update_output_cache(
        image_path,
        output_path,
        cached_text,
        logging.getLogger(__name__),
    )

    def fail_loader(*_: object) -> LoadedOcrModel:
        raise AssertionError("Модель не должна загружаться при наличии кеша.")

    result = run_ocr(
        OcrOptions(
            input_path=image_path,
            output_path=output_path,
            model_path=tmp_path / "missing-model",
        ),
        loader=fail_loader,
    )

    assert result.cached is True
    assert result.device == "cache"
    assert result.text == cached_text
    assert output_path.read_text(encoding="utf-8") == cached_text
    assert has_utf8_bom(output_path) is False


def test_force_ignores_cache_and_updates_it(tmp_path, monkeypatch):
    cache_path = tmp_path / "ocr_cache.json"
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", cache_path)
    image_path = _create_image(tmp_path / "input.jpg")
    output_path = tmp_path / "result.txt"
    model_path = _create_fake_model_dir(tmp_path / "model")
    output_cache.update_output_cache(
        image_path,
        output_path,
        "Старый кеш",
        logging.getLogger(__name__),
    )

    def loader(*_: object) -> LoadedOcrModel:
        return LoadedOcrModel(
            processor=FakeProcessor("Новый OCR"),
            model=FakeModel(),
            device="cpu",
            quantized=False,
        )

    result = run_ocr(
        OcrOptions(
            input_path=image_path,
            output_path=output_path,
            model_path=model_path,
            force=True,
        ),
        loader=loader,
    )

    cache = output_cache.load_cache_snapshot(logging.getLogger(__name__))
    assert result.cached is False
    assert result.text == "Новый OCR"
    assert output_cache.get_cached_text(cache, image_path) == "Новый OCR"
    assert output_path.read_text(encoding="utf-8") == "Новый OCR"


def test_existing_output_seeds_cache_without_model_load(tmp_path, monkeypatch):
    cache_path = tmp_path / "ocr_cache.json"
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", cache_path)
    image_path = _create_image(tmp_path / "input.jpg")
    output_path = tmp_path / "result.txt"
    output_path.write_text("Уже готовый OCR", encoding="utf-8")

    def fail_loader(*_: object) -> LoadedOcrModel:
        raise AssertionError("Модель не должна загружаться при готовом результате.")

    result = run_ocr(
        OcrOptions(
            input_path=image_path,
            output_path=output_path,
            model_path=tmp_path / "missing-model",
        ),
        loader=fail_loader,
    )
    cache = output_cache.load_cache_snapshot(logging.getLogger(__name__))

    assert result.cached is True
    assert result.text == "Уже готовый OCR"
    assert output_cache.get_cached_text(cache, image_path) == "Уже готовый OCR"


def test_cache_ignores_changed_input_file(tmp_path, monkeypatch):
    cache_path = tmp_path / "ocr_cache.json"
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", cache_path)
    image_path = _create_image(tmp_path / "input.jpg", color="white")
    output_cache.update_output_cache(
        image_path,
        tmp_path / "result.txt",
        "Старый текст",
        logging.getLogger(__name__),
    )

    _create_image(image_path, color="black", size=(32, 32))
    cache = output_cache.load_cache_snapshot(logging.getLogger(__name__))

    assert output_cache.get_cached_text(cache, image_path) is None


def _create_image(path: Path, *, color: str = "white", size: tuple[int, int] = (24, 24)) -> Path:
    image = Image.new("RGB", size, color)
    image.save(path, format="JPEG")
    return path


def _create_fake_model_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"")
    return path
