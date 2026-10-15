from __future__ import annotations

import ipaddress
import json
import logging
import secrets
import uuid
import weakref

from starlette.types import ASGIApp, Message, Receive, Scope, Send


_instances: weakref.WeakSet[LocalWebSecurity] = weakref.WeakSet()


def validate_listener_config(host: str, port: int) -> None:
    """Проверяет адрес до создания приложения и открытия сокета."""
    if host not in ("127.0.0.1", "::1"):
        raise ValueError("Допустим только локальный адрес 127.0.0.1 или ::1.")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Порт должен быть целым числом от 1 до 65535.")


class _SecretRedactingFormatter(logging.Formatter):
    def __init__(self, original: logging.Formatter | None) -> None:
        super().__init__()
        self.original = original or logging.Formatter()

    def format(self, record: logging.LogRecord) -> str:
        # Исключение форматируется раньше замены: секрет не остаётся в traceback.
        rendered = self.original.format(record)
        for instance in list(_instances):
            rendered = rendered.replace(instance._secret, "[СКРЫТО]")
        return rendered


def install_secret_log_redaction() -> None:
    """Закрывает секрет в настроенных обработчиках; вызывать после setup_logging."""
    loggers = [logging.getLogger()]
    loggers.extend(
        item for item in logging.Logger.manager.loggerDict.values()
        if isinstance(item, logging.Logger)
    )
    for logger in loggers:
        for handler in logger.handlers:
            if not isinstance(handler.formatter, _SecretRedactingFormatter):
                handler.setFormatter(_SecretRedactingFormatter(handler.formatter))


class LocalWebSecurity:
    """Проверяет локальную границу до маршрутизации и первого чтения тела."""

    def __init__(self, app: ASGIApp, host: str = "127.0.0.1", port: int = 7861) -> None:
        validate_listener_config(host, port)
        self.app = app
        authority_host = f"[{host}]" if host == "::1" else host
        self.authority = f"{authority_host}:{port}".encode("ascii")
        self.origin = b"http://" + self.authority
        self.cookie_name = f"ocr_local_{port}"
        self._secret = secrets.token_hex(32)
        _instances.add(self)

    @staticmethod
    def _local_peer(scope: Scope) -> bool:
        client = scope.get("client")
        if not client:
            return False
        try:
            address = ipaddress.ip_address(client[0])
        except ValueError:
            return False
        return address.is_loopback

    def _cookie(self, headers: list[tuple[bytes, bytes]]) -> tuple[bool, str | None]:
        values = []
        for name, value in headers:
            if name != b"cookie":
                continue
            for part in value.decode("latin-1").split(";"):
                key, separator, token = part.strip().partition("=")
                if key == self.cookie_name:
                    values.append(token if separator else "")
        return len(values) <= 1, values[0] if values else None

    def _allowed(self, scope: Scope) -> tuple[bool, bool]:
        if not self._local_peer(scope):
            return False, False
        headers = [(key.lower(), value) for key, value in scope.get("headers", [])]
        if any(key == b"forwarded" or key.startswith(b"x-forwarded-") for key, _ in headers):
            return False, False
        hosts = [value for key, value in headers if key == b"host"]
        origins = [value for key, value in headers if key == b"origin"]
        if hosts != [self.authority] or len(origins) > 1:
            return False, False
        if origins and origins[0] != self.origin:
            return False, False
        method = scope["method"]
        if method not in ("GET", "HEAD", "OPTIONS") and not origins:
            return False, False

        metadata = {}
        for key, value in headers:
            if key.startswith(b"sec-fetch-"):
                if key in metadata:
                    return False, False
                metadata[key] = value
        site = metadata.get(b"sec-fetch-site")
        if site is not None and site not in (b"same-origin", b"none"):
            return False, False

        unambiguous, cookie = self._cookie(headers)
        if not unambiguous:
            return False, False
        path = scope.get("path", "")
        bootstrap = method == "GET" and path == "/"
        if bootstrap and metadata:
            if (site not in (b"none", b"same-origin")
                    or metadata.get(b"sec-fetch-mode") != b"navigate"
                    or metadata.get(b"sec-fetch-dest") != b"document"):
                return False, False
        static = path.startswith("/static/") and not any(
            part in (".", "..") for part in path.split("/")
        ) and "\\" not in path
        public = method in ("GET", "HEAD") and (path in ("/", "/api/health") or static)
        if not public:
            if cookie is None or not secrets.compare_digest(cookie.encode("latin-1"), self._secret.encode("ascii")):
                return False, False
        return True, bootstrap

    @staticmethod
    async def _deny(send: Send) -> None:
        diagnostic_id = uuid.uuid4().hex
        logging.getLogger(__name__).warning("Диагностика %s: access_denied", diagnostic_id)
        body = json.dumps({
            "detail": "Доступ к локальному приложению запрещён.",
            "code": "access_denied",
            "diagnostic_id": diagnostic_id,
        }, ensure_ascii=False).encode("utf-8")
        await send({"type": "http.response.start", "status": 403, "headers": [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"cache-control", b"no-store"),
            (b"x-frame-options", b"DENY"),
            (b"content-security-policy", b"frame-ancestors 'none'"),
        ]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        allowed, bootstrap = self._allowed(scope)
        if not allowed:
            await self._deny(send)
            return

        async def secured_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (key, value) for key, value in message.get("headers", [])
                    if key.lower() not in (b"cache-control", b"x-frame-options")
                ]
                headers.extend([
                    (b"cache-control", b"no-store"),
                    (b"x-frame-options", b"DENY"),
                    (b"content-security-policy", b"frame-ancestors 'none'"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                ])
                if bootstrap and message["status"] == 200:
                    headers.append((b"set-cookie", (
                        f"{self.cookie_name}={self._secret}; Path=/; HttpOnly; SameSite=Strict"
                    ).encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, secured_send)
