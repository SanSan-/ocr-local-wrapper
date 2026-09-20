from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ocr_local.exceptions import OcrModelError
from ocr_local.models import LoadedOcrModel, OcrMetrics, OcrOptions
from ocr_local.service import recognize_image_text, run_ocr
from ocr_local.utils import output_cache


class Processor:
    # У native PaddleOCRVLImageProcessor нет min_pixels.
    image_processor = object()

    def __init__(self, text="Пример"):
        self.text, self.kwargs = text, {}

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return {"input_ids": SimpleNamespace(shape=(1, 2)), "token_type_ids": [0, 0]}

    def decode(self, tokens, **kwargs):
        return self.text


class Model:
    device = "cpu"
    generation_config = SimpleNamespace(eos_token_id=9)

    def __init__(self, tokens):
        self.tokens, self.kwargs = tokens, {}

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[1, 2, *self.tokens]]


@pytest.mark.parametrize("model_id,prompt,min_pixels,max_pixels", [
    ("glm-ocr", "Text Recognition:", 12544, 2007040),
    ("paddleocr-vl-1.6", "OCR:", 112896, 1003520),
])
def test_native_processor_contract_and_cached_greedy_generation(model_id, prompt, min_pixels, max_pixels):
    processor, model = Processor(), Model([8, 9])
    options = OcrOptions(Path("unused.png"), model_id=model_id)
    metrics = OcrMetrics()
    result = recognize_image_text(options.input_path, options,
                                  LoadedOcrModel(processor, model, "cpu", False), metrics=metrics)
    assert result == "Пример"
    assert processor.messages[0]["content"][1]["text"] == prompt
    assert processor.kwargs["processor_kwargs"]["images_kwargs"]["size"] == {
        "shortest_edge": min_pixels, "longest_edge": max_pixels,
    }
    assert model.kwargs["use_cache"] is True
    assert model.kwargs["do_sample"] is False and model.kwargs["num_beams"] == 1
    assert "token_type_ids" not in model.kwargs
    assert metrics.generated_tokens == 2 and metrics.generate_seconds >= 0


@pytest.mark.parametrize("tokens,reached", [([3, 4], True), ([3, 9], False), ([9], False)])
def test_token_limit_distinguishes_eos(tokens, reached):
    metrics = OcrMetrics()
    recognize_image_text(Path("a.png"), OcrOptions(Path("a.png"), max_new_tokens=2),
                         LoadedOcrModel(Processor(), Model(tokens), "cpu", False), metrics=metrics)
    assert metrics.limit_reached is reached


def test_empty_response_does_not_publish_txt_or_cache(tmp_path, monkeypatch):
    from PIL import Image
    image = tmp_path / "input.png"
    Image.new("RGB", (24, 24)).save(image)
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", cache)
    with pytest.raises(OcrModelError, match="не вернула"):
        run_ocr(OcrOptions(image), loaded=LoadedOcrModel(Processor("<|endoftext|>"),
                                                       Model([9]), "cpu", False))
    assert not image.with_suffix(".txt").exists() and not cache.exists()
