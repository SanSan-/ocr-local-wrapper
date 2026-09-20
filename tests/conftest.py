from __future__ import annotations

import os

import pytest

from ocr_local import cli
from ocr_local.web import app as web_app


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Тесты не читают рабочие настройки и не дописывают пользовательские логи."""
    for name in ("OCR_MODEL_PATH", "OCR_GLM_MODEL_PATH", "OCR_PADDLE_MODEL_PATH"):
        if name == "OCR_MODEL_PATH" and os.getenv("RUN_GLM_OCR_TEST") == "1":
            continue
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "load_environment", lambda: False)
    monkeypatch.setattr(web_app, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(web_app._model_pool, "clear", lambda: None)
