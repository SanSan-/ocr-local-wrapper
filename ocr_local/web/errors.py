from __future__ import annotations

import logging
import uuid

from starlette.exceptions import HTTPException

from ocr_local.exceptions import OcrModelError, OcrValidationError


class WebError(HTTPException):
    """Ошибка с заранее определённым безопасным сообщением для браузера."""

    def __init__(self, code: str, status_code: int, message: str, *, headers=None) -> None:
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.code = code
        self.status_code = status_code
        self.message = message


def public_error(exc: Exception, logger: logging.Logger) -> tuple[int, dict[str, str]]:
    """Связывает безопасный ответ с полной локальной диагностикой."""
    diagnostic_id = uuid.uuid4().hex
    if isinstance(exc, WebError):
        status, code, message = exc.status_code, exc.code, exc.message
    elif isinstance(exc, OcrModelError):
        status, code = 400, "model_error"
        message = "Не удалось выполнить OCR. Проверьте локальную модель и параметры распознавания."
    elif isinstance(exc, OcrValidationError):
        status, code = 400, "invalid_input"
        message = "Проверьте изображение, параметры и локальную настройку выбранной модели."
    else:
        status, code, message = 500, "internal_error", "Не удалось завершить операцию. Подробности в локальном журнале."
    logger.error("Диагностика %s: %s", diagnostic_id, code, exc_info=(type(exc), exc, exc.__traceback__))
    return status, {"detail": message, "code": code, "diagnostic_id": diagnostic_id}
