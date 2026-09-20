from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ocr_local.exceptions import OcrModelError
from ocr_local.models import LoadedOcrModel
from ocr_local.utils import model_utils


class FakeDevice:
    def __init__(self, value):
        self.value, self.type = value, value.split(":")[0]

    def __str__(self):
        return self.value


class FakeTorch:
    bfloat16, float16, float32 = "bf16", "fp16", "fp32"

    def __init__(self, cuda=True, bf16=True):
        self.cuda = SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=lambda: bf16)
        self.device = FakeDevice


@pytest.mark.parametrize("cuda,allow_quantization,quantized", [
    (True, True, True), (True, False, False), (False, True, False),
])
def test_device_and_explicit_quantization(monkeypatch, cuda, allow_quantization, quantized):
    monkeypatch.setattr(model_utils, "_import_torch", lambda: FakeTorch(cuda))
    monkeypatch.setattr(model_utils, "_import_bitsandbytes_config", lambda: lambda **kw: kw)
    device, config, actual = model_utils.resolve_device_and_quantization(allow_quantization=allow_quantization)
    assert actual is quantized
    assert device.type == ("cuda" if cuda else "cpu")
    assert config == ({"load_in_8bit": True} if quantized else None)


def test_missing_int8_dependency_is_not_silent(monkeypatch):
    monkeypatch.setattr(model_utils, "_import_torch", lambda: FakeTorch())
    monkeypatch.setattr(model_utils, "_import_bitsandbytes_config", lambda: None)
    with pytest.raises(OcrModelError, match="Отключите INT8"):
        model_utils.resolve_device_and_quantization(allow_quantization=True)


@pytest.mark.parametrize("device,bf16,dtype", [
    ("cuda:0", True, "bf16"), ("cuda:0", False, "fp16"), ("cpu", True, "fp32"),
])
def test_loads_directly_on_target_without_cpu_copy(monkeypatch, device, bf16, dtype):
    calls = []
    model = SimpleNamespace(eval=lambda: model)
    cls = SimpleNamespace(from_pretrained=lambda *args, **kw: calls.append(kw) or model)
    monkeypatch.setattr(model_utils, "_import_torch", lambda: FakeTorch(bf16=bf16))
    assert model_utils._load_model(cls, Path("model"), FakeDevice(device)) is model
    assert calls[0]["device_map"] == {"": device}
    assert calls[0]["dtype"] == dtype
    assert calls[0]["attn_implementation"] == "sdpa"
    assert calls[0]["local_files_only"] and not calls[0]["trust_remote_code"]


@pytest.mark.parametrize("fallback", [False, True])
def test_gpu_failure_obeys_cpu_policy(monkeypatch, fallback):
    calls = []
    monkeypatch.setattr(model_utils, "_import_torch", lambda: FakeTorch())
    processor = SimpleNamespace(from_pretrained=lambda *a, **kw: object())
    monkeypatch.setattr(model_utils, "_import_transformers_components", lambda: (processor, object()))
    monkeypatch.setattr(model_utils, "clear_model_memory", lambda: None)

    def load(cls, path, device, config=None):
        calls.append(device.type)
        if device.type == "cuda":
            raise RuntimeError("test GPU OOM")
        return SimpleNamespace(dtype="fp32")
    monkeypatch.setattr(model_utils, "_load_model", load)
    if fallback:
        loaded = model_utils.load_model_components(Path("model"), False, True)
        assert loaded.device.type == "cpu"
        assert loaded.quantized is False
        assert calls == ["cuda", "cpu"]
    else:
        with pytest.raises(OcrModelError, match="test GPU OOM"):
            model_utils.load_model_components(Path("model"), False, False)
        assert calls == ["cuda"]


def test_pool_reuses_then_releases_before_switch_and_retries_failure(monkeypatch, tmp_path):
    events = []
    fail = True

    def load(path, quant, cpu):
        nonlocal fail
        events.append(path.name)
        if path.name == "b" and fail:
            fail = False
            raise OcrModelError("test load failed")
        return LoadedOcrModel(object(), object(), "cpu", quant)
    monkeypatch.setattr(model_utils, "clear_model_memory", lambda: events.append("release"))
    pool = model_utils.ModelPool(load)
    first = pool.get(tmp_path / "a", False, False)
    assert pool.get(tmp_path / "a", False, False) is first
    with pytest.raises(OcrModelError):
        pool.get(tmp_path / "b", False, False)
    second = pool.get(tmp_path / "b", False, False)
    assert second is not first
    assert events == ["a", "release", "b", "b"]
    pool.clear()
    assert events[-1] == "release"
