from __future__ import annotations


class OcrError(RuntimeError):
    """Базовая ошибка OCR-пайплайна."""


class OcrValidationError(OcrError):
    """Ошибка проверки входных или выходных данных."""


class OcrModelError(OcrError):
    """Ошибка загрузки или выполнения OCR-модели."""

