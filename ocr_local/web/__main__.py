from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from ocr_local.web.security import validate_listener_config


def main(argv: Sequence[str] | None = None) -> None:
    """Запускает веб-интерфейс OCR-local."""
    parser = argparse.ArgumentParser(description="Локальный веб-интерфейс OCR-local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args(argv)
    try:
        validate_listener_config(args.host, args.port)
    except ValueError as exc:
        parser.error(str(exc))

    from ocr_local.web.app import create_app

    uvicorn.run(
        create_app(host=args.host, port=args.port), host=args.host, port=args.port,
        reload=False, workers=1, proxy_headers=False,
    )


if __name__ == "__main__":
    main()

