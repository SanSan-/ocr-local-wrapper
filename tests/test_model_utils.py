from __future__ import annotations

from ocr_local.utils import model_utils


class FakeDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type

    def __str__(self) -> str:
        return self.type


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeTorch:
    def __init__(self, cuda_available: bool) -> None:
        self.cuda = FakeCuda(cuda_available)

    def device(self, device_type: str) -> FakeDevice:
        return FakeDevice(device_type)


class FakeBitsAndBytesConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_resolve_device_enables_gpu_quantization(monkeypatch):
    monkeypatch.setattr(model_utils, "_import_torch", lambda: FakeTorch(True))
    monkeypatch.setattr(
        model_utils,
        "_import_bitsandbytes_config",
        lambda: FakeBitsAndBytesConfig,
    )

    device, quantization_config, quantized = model_utils.resolve_device_and_quantization()

    assert device.type == "cuda"
    assert quantized is True
    assert quantization_config.kwargs["load_in_8bit"] is True


def test_resolve_device_uses_cpu_without_quantization(monkeypatch):
    monkeypatch.setattr(model_utils, "_import_torch", lambda: FakeTorch(False))
    monkeypatch.setattr(
        model_utils,
        "_import_bitsandbytes_config",
        lambda: FakeBitsAndBytesConfig,
    )

    device, quantization_config, quantized = model_utils.resolve_device_and_quantization()

    assert device.type == "cpu"
    assert quantization_config is None
    assert quantized is False


def test_resolve_device_respects_no_quantization(monkeypatch):
    monkeypatch.setattr(model_utils, "_import_torch", lambda: FakeTorch(True))
    monkeypatch.setattr(
        model_utils,
        "_import_bitsandbytes_config",
        lambda: FakeBitsAndBytesConfig,
    )

    device, quantization_config, quantized = model_utils.resolve_device_and_quantization(
        allow_quantization=False
    )

    assert device.type == "cuda"
    assert quantization_config is None
    assert quantized is False

