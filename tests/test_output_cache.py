from __future__ import annotations

import logging
from dataclasses import replace

import pytest
from PIL import Image

from ocr_local.exceptions import OcrValidationError
from ocr_local.model_profiles import resolve_options, settings_identity
from ocr_local.models import LoadedOcrModel, OcrOptions
from ocr_local.service import run_ocr
from ocr_local.utils import output_cache
from ocr_local.utils.io_utils import has_utf8_bom

LOG = logging.getLogger(__name__)


class FakeInputs(dict):
    def __init__(self):
        from types import SimpleNamespace
        super().__init__(input_ids=SimpleNamespace(shape=(1, 2)), token_type_ids=object())

    def to(self, device):
        return self


class FakeProcessor:
    def apply_chat_template(self, *args, **kwargs):
        return FakeInputs()

    def decode(self, tokens, **kwargs):
        return "Новый OCR"


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return [[10, 11, 12]]


@pytest.fixture
def options(tmp_path, monkeypatch):
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", tmp_path / "cache.json")
    image = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "white").save(image)
    return resolve_options(OcrOptions(image, model_path=tmp_path / "missing-model"))


def fail_loader(*args):
    raise AssertionError("Модель не должна загружаться.")


def seed(options, text="Сохранённый текст", **kwargs):
    output_cache.update_output_cache(
        options.input_path, options.input_path.with_suffix(".txt"), text, LOG,
        settings=settings_identity(options), **kwargs,
    )


def test_cache_hit_before_model_validation_and_restores_utf8(options):
    seed(options)
    result = run_ocr(options, loader=fail_loader)
    assert result.cached and result.device == "cache"
    assert result.text == "Сохранённый текст"
    assert result.output_path.read_text(encoding="utf-8") == result.text
    assert not has_utf8_bom(result.output_path)


def test_existing_txt_is_preserved_and_not_imported(options):
    output = options.input_path.with_suffix(".txt")
    output.write_text("Пользовательский результат", encoding="utf-8")
    with pytest.raises(OcrValidationError, match="force"):
        run_ocr(options, loader=fail_loader)
    assert output.read_text(encoding="utf-8") == "Пользовательский результат"
    assert not output_cache.OCR_CACHE_FILE.exists()


@pytest.mark.parametrize("changed", [
    {"model_id": "paddleocr-vl-1.6"},
    {"model_path": "other-model"},
    {"prompt": "Other:"}, {"max_new_tokens": 123}, {"max_pixels": 400000},
    {"quantization_enabled": True},
])
def test_cache_does_not_cross_model_or_parameters(options, changed):
    from pathlib import Path
    seed(options)
    if "model_path" in changed:
        changed["model_path"] = Path(changed["model_path"])
    other = resolve_options(replace(options, **changed))
    cache = output_cache.load_cache_snapshot(LOG)
    assert output_cache.get_cached_text(cache, options.input_path, settings=settings_identity(other)) is None
    assert output_cache.get_cached_text(cache, options.input_path, settings=settings_identity(options))


def test_old_cache_preserved_but_not_assumed_selected_model(options):
    output_cache.update_output_cache(options.input_path, options.input_path.with_suffix(".txt"), "Старый кеш", LOG)
    cache = output_cache.load_cache_snapshot(LOG)
    assert output_cache.get_cached_text(cache, options.input_path, settings=settings_identity(options)) is None
    seed(options)
    cache = output_cache.load_cache_snapshot(LOG)
    assert output_cache.get_cached_text(cache, options.input_path) == "Старый кеш"


def test_force_runs_model_and_updates_only_selected_record(options):
    seed(options, "Прежний результат")
    loaded = LoadedOcrModel(FakeProcessor(), FakeModel(), "cpu", False)
    result = run_ocr(replace(options, force=True), loaded=loaded)
    assert not result.cached and result.text == "Новый OCR"
    cache = output_cache.load_cache_snapshot(LOG)
    assert output_cache.get_cached_text(cache, options.input_path, settings=settings_identity(options)) == result.text


def test_changed_image_does_not_hit_cache(options):
    seed(options)
    Image.new("RGB", (48, 48), "black").save(options.input_path)
    cache = output_cache.load_cache_snapshot(LOG)
    assert output_cache.get_cached_text(cache, options.input_path, settings=settings_identity(options)) is None


def test_cache_keeps_truncation_warning_and_does_not_overwrite_txt(options):
    seed(options, limit_reached=True)
    output = options.input_path.with_suffix(".txt")
    output.write_text("Моя правка", encoding="utf-8")
    result = run_ocr(options, loader=fail_loader)
    assert result.metrics.limit_reached
    assert output.read_text(encoding="utf-8") == "Моя правка"
