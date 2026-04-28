from __future__ import annotations

import uvicorn


def main() -> None:
    """Запускает веб-интерфейс OCR-local."""
    uvicorn.run("ocr_local.web.app:app", host="127.0.0.1", port=7861, reload=False)


if __name__ == "__main__":
    main()

