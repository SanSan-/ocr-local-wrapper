"""Изолированный стенд браузерной проверки; реальные модели не загружаются."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path


class BrowserCounters:
    """Сохраняет наблюдения локально, не добавляя маршрутов к приложению."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = threading.RLock()
        self.data = {
            "ocr_started": 0, "ocr_succeeded": 0, "ocr_failed": 0,
            "ocr_active": False, "cache_hits": 0, "request_counts": {},
            "response_counts": {}, "stream_attempts": {}, "stream_cuts": 0,
        }
        self.save()

    def save(self) -> None:
        with self.lock:
            temporary = self.root / "browser-state.json.tmp"
            temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.root / "browser-state.json")

    def increment(self, key: str, member: str | None = None) -> None:
        with self.lock:
            if member is None:
                self.data[key] += 1
            else:
                self.data[key][member] = self.data[key].get(member, 0) + 1
            self.save()


class DisconnectFirstStream:
    """Завершает первый допущенный SSE каждой задачи до done, сохраняя защитные проверки."""

    def __init__(self, application, counters: BrowserCounters) -> None:
        self.application = application
        self.counters = counters
        self.cut_jobs: set[str] = set()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.application(scope, receive, send)
            return
        path = scope["path"]
        self.counters.increment("request_counts", scope["method"] + " " + path)
        is_stream = scope["method"] == "GET" and path.startswith("/api/stream/")
        if is_stream:
            self.counters.increment("stream_attempts", path)
        cut = False

        async def observed_send(message):
            nonlocal cut
            if message["type"] == "http.response.start":
                self.counters.increment("response_counts", str(message["status"]) + " " + path)
                headers = dict(message.get("headers", []))
                if (is_stream and message["status"] == 200 and path not in self.cut_jobs
                        and headers.get(b"content-type", b"").startswith(b"text/event-stream")):
                    # Решение принимается только после реальной проверки Host/Origin/cookie и подписки.
                    self.cut_jobs.add(path)
                    cut = True
                    self.counters.increment("stream_cuts")
                    await send(message)
                    await send({"type": "http.response.body", "body": b"retry: 500\n\n: keep-alive\n\n",
                                "more_body": True})
                    await asyncio.sleep(0.05)
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
                    return
            if not cut:
                await send(message)

        # Штатный ответ сам освобождает подписчика при disconnect или завершении итератора.
        # После искусственного EOF его данные не дописываются в уже закрытый HTTP-ответ.
        await self.application(scope, receive, observed_send)


def prepare_root(requested: Path) -> Path:
    """Разрешает только отдельный временный каталог без чужого окружения."""
    root = requested.expanduser().resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if root == temp_root or not root.is_relative_to(temp_root):
        raise ValueError("--root должен указывать на отдельный подкаталог системного TEMP.")
    marker = root / ".browser-hardening-harness"
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError("Непустой временный каталог не принадлежит этому стенду.")
    if (root / ".env").exists():
        raise ValueError("В каталоге стенда не должно быть .env.")
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text("Изолированный браузерный стенд ocr-local.\n", encoding="utf-8")
    for name in ("uploads", "cache", "logs", "models"):
        target = root / name
        if not target.resolve().is_relative_to(root):
            raise ValueError("Каталог стенда не должен выходить за пределы --root.")
        target.mkdir(exist_ok=True)
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Отдельный каталог внутри системного TEMP.")
    parser.add_argument("--port", type=int, required=True, help="Свободный порт 127.0.0.1; от 1 до 65535.")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Порт должен быть от 1 до 65535.")
    root = prepare_root(args.root)
    repo = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))

    for name, folder in (("OCR_MODEL_PATH", "GLM-OCR"), ("OCR_GLM_MODEL_PATH", "GLM-OCR"),
                         ("OCR_PADDLE_MODEL_PATH", "PaddleOCR-VL-1.6")):
        os.environ[name] = str(root / "models" / folder)
    for folder, model_type in (("GLM-OCR", "glm_ocr"), ("PaddleOCR-VL-1.6", "paddleocr_vl")):
        model_dir = root / "models" / folder
        model_dir.mkdir(exist_ok=True)
        # Это заглушки доступности каталога, а не веса модели.
        (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
        (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"")

    # Импорт приложения выполняется только после перехода в изолированный cwd.
    from PIL import Image, ImageDraw
    import uvicorn
    from ocr_local.model_profiles import settings_identity
    from ocr_local.models import OcrMetrics, OcrResult
    from ocr_local.utils import output_cache
    from ocr_local.utils.io_utils import ensure_output_can_be_written
    from ocr_local.web import app as web

    def isolated_logging(verbose=False):
        handler = logging.FileHandler(root / "logs" / "server.log", encoding="utf-8")
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=[handler], force=True,
                            format="%(asctime)s %(levelname)s %(name)s %(message)s")
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            target = logging.getLogger(name)
            target.handlers.clear()
            target.propagate = True

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("Стенд запрещает загрузку реальной модели.")

    web.UPLOADS_DIR = root / "uploads"
    output_cache.OCR_CACHE_FILE = root / "cache" / "ocr_output_cache.json"
    web.setup_logging = isolated_logging
    web._model_pool.get = forbidden_loader
    web._model_pool.clear = lambda: None
    isolated_logging()
    counters = BrowserCounters(root)

    def fake_run_ocr(options, *, loader, storage_policy):
        settings = settings_identity(options)
        if not options.input_path.resolve().is_relative_to(root / "uploads"):
            raise AssertionError("Вход стенда находится вне изолированного uploads.")
        if not options.output_path.resolve().is_relative_to(root / "uploads"):
            raise AssertionError("Выход стенда находится вне изолированного uploads.")
        cached = None if options.force else output_cache.get_cached_text(
            storage_policy.load_cache(), options.input_path, settings=settings)
        if cached is not None:
            result = OcrResult(options.input_path, options.output_path, cached, "cache", False, True,
                               options.model_id, settings)
            storage_policy.restore_result(result)
            counters.increment("cache_hits")
            return result
        ensure_output_can_be_written(options.output_path, options.force)
        with counters.lock:
            counters.data["ocr_active"] = True
            counters.increment("ocr_started")
            run_number = counters.data["ocr_started"]
        try:
            time.sleep(1.5)
            if options.prompt == "FAIL":
                raise RuntimeError("Искусственная внутренняя ошибка: " + str(root / "private" / "model.bin"))
            text = "Тестовое распознавание завершено. Запуск " + str(run_number) + "."
            metrics = OcrMetrics(generate_seconds=1.5, total_seconds=1.5, generated_tokens=8,
                                 tokens_per_second=8 / 1.5)
            result = OcrResult(options.input_path, options.output_path, text, "test", False,
                               model_id=options.model_id, settings=settings, metrics=metrics)
            storage_policy.save_result(result)
            counters.increment("ocr_succeeded")
            return result
        except Exception:
            counters.increment("ocr_failed")
            raise
        finally:
            with counters.lock:
                counters.data["ocr_active"] = False
                counters.save()

    web.run_ocr = fake_run_ocr
    image = Image.new("RGB", (800, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 25, 775, 235), outline="#2463a5", width=6)
    draw.text((55, 75), "WEB-005 / WEB-007 / WEB-008", fill="#17364e", font_size=30)
    draw.text((55, 135), "Isolated browser check: no OCR weights", fill="#17364e", font_size=24)
    image.save(root / "source.png")
    info = {"url": f"http://127.0.0.1:{args.port}", "pid": os.getpid(), "root": str(root),
            "source_image": str(root / "source.png"), "counters": str(root / "browser-state.json"),
            "failure_prompt": "FAIL", "fake_delay_seconds": 1.5,
            "stream_disconnect": "Первая допущенная подписка каждой задачи; повтор через 500 мс."}
    (root / "browser-server.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    application = DisconnectFirstStream(web.create_app(host="127.0.0.1", port=args.port), counters)
    uvicorn.run(application, host="127.0.0.1", port=args.port, proxy_headers=False,
                log_config=None, access_log=False)


if __name__ == "__main__":
    main()
