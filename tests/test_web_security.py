from __future__ import annotations

import asyncio
import io
import json
import logging

import pytest

from ocr_local.web.security import LocalWebSecurity, install_secret_log_redaction, validate_listener_config


class ProbeApp:
    def __init__(self):
        self.calls = 0
        self.reads = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        if scope["method"] == "POST":
            self.reads += 1
            await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"status":"ok"}'})


def request(app, *, path="/api/private", method="GET", headers=None, peer="127.0.0.1"):
    messages = []
    reads = []

    async def receive():
        reads.append(True)
        return {"type": "http.request", "body": b"private image", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "root_path": "", "server": ("127.0.0.1", 7861),
        "client": (peer, 43210) if peer is not None else None,
        "headers": headers if headers is not None else [(b"host", b"127.0.0.1:7861")],
    }
    asyncio.run(app(scope, receive, send))
    return messages, reads


def session(app, **kwargs):
    messages, _ = request(app, path="/", **kwargs)
    assert messages[0]["status"] == 200
    return next(value.split(b";", 1)[0] for key, value in messages[0]["headers"] if key == b"set-cookie")


def authorized_headers(cookie):
    return [(b"host", b"127.0.0.1:7861"), (b"origin", b"http://127.0.0.1:7861"), (b"cookie", cookie)]


@pytest.mark.parametrize("host", ["localhost", "0.0.0.0", "::", "127.1", "127.0.0.2", "192.168.1.2", "[::1]"])
def test_listener_rejects_aliases_and_noncanonical_addresses(host):
    with pytest.raises(ValueError):
        validate_listener_config(host, 7861)


@pytest.mark.parametrize("port", [0, -1, 65536, True, "7861"])
def test_listener_rejects_invalid_ports(port):
    with pytest.raises(ValueError):
        validate_listener_config("127.0.0.1", port)


def test_bootstrap_cookie_headers_and_native_event_source():
    probe = ProbeApp()
    app = LocalWebSecurity(probe)
    navigation = [(b"host", b"127.0.0.1:7861"), (b"sec-fetch-site", b"none"),
                  (b"sec-fetch-mode", b"navigate"), (b"sec-fetch-dest", b"document")]
    messages, reads = request(app, path="/", headers=navigation)
    assert not reads
    headers = dict(messages[0]["headers"])
    cookie = headers[b"set-cookie"]
    assert b"HttpOnly" in cookie and b"SameSite=Strict" in cookie and b"Path=/" in cookie
    assert b"Domain=" not in cookie and b"Secure" not in cookie
    value = cookie.split(b";", 1)[0]
    assert len(bytes.fromhex(value.split(b"=", 1)[1].decode())) == 32
    assert headers[b"cache-control"] == b"no-store"
    assert headers[b"x-frame-options"] == b"DENY"
    assert headers[b"content-security-policy"] == b"frame-ancestors 'none'"
    assert value.split(b"=", 1)[1] not in messages[1]["body"]
    assert session(app) == value
    sse_headers = [(b"host", b"127.0.0.1:7861"), (b"cookie", value),
                   (b"accept", b"text/event-stream"), (b"sec-fetch-site", b"same-origin")]
    for _ in range(2):
        streamed, _ = request(app, path="/api/jobs/id/events", headers=sse_headers)
        assert streamed[0]["status"] == 200
        assert b"set-cookie" not in dict(streamed[0]["headers"])
    accepted, reads = request(app, method="POST", headers=authorized_headers(value))
    assert accepted[0]["status"] == 200 and reads == [True]


@pytest.mark.parametrize("alteration", [
    "host_alias", "host_port", "host_missing", "host_duplicate", "origin_foreign",
    "origin_opaque", "origin_port", "origin_missing", "origin_duplicate", "cross_site",
    "same_site", "site_duplicate", "forwarded", "x_forwarded_host", "x_forwarded_for",
    "cookie_missing", "cookie_wrong", "cookie_duplicate", "cookie_two_headers", "cookie_nonascii",
])
def test_denial_precedes_read_and_application(alteration):
    probe = ProbeApp()
    app = LocalWebSecurity(probe)
    cookie = session(app)
    headers = authorized_headers(cookie)
    replace = {
        "host_alias": (b"host", b"localhost:7861"), "host_port": (b"host", b"127.0.0.1:7862"),
        "origin_foreign": (b"origin", b"https://example.org"), "origin_opaque": (b"origin", b"null"),
        "origin_port": (b"origin", b"http://127.0.0.1:7862"),
        "cookie_wrong": (b"cookie", b"ocr_local_7861=wrong"),
        "cookie_duplicate": (b"cookie", cookie + b"; " + cookie),
        "cookie_nonascii": (b"cookie", b"ocr_local_7861=\xff"),
    }
    additions = {
        "host_duplicate": [(b"host", b"127.0.0.1:7861")],
        "origin_duplicate": [(b"origin", b"http://127.0.0.1:7861")],
        "cross_site": [(b"sec-fetch-site", b"cross-site")],
        "same_site": [(b"sec-fetch-site", b"same-site")],
        "site_duplicate": [(b"sec-fetch-site", b"same-origin")] * 2,
        "forwarded": [(b"forwarded", b"for=127.0.0.1;host=127.0.0.1:7861")],
        "x_forwarded_host": [(b"x-forwarded-host", b"127.0.0.1:7861")],
        "x_forwarded_for": [(b"x-forwarded-for", b"127.0.0.1")],
        "cookie_two_headers": [(b"cookie", cookie)],
    }
    if alteration in replace:
        key, value = replace[alteration]
        headers = [(name, value if name == key else old) for name, old in headers]
    elif alteration.endswith("_missing"):
        removed = alteration.removesuffix("_missing").encode()
        headers = [(name, value) for name, value in headers if name != removed]
    else:
        headers.extend(additions[alteration])
    before = probe.calls
    messages, reads = request(app, method="POST", headers=headers)
    assert messages[0]["status"] == 403
    assert not reads and probe.calls == before and probe.reads == 0
    error = json.loads(messages[1]["body"])
    assert error["code"] == "access_denied" and len(error["diagnostic_id"]) == 32
    assert set(error) == {"detail", "code", "diagnostic_id"}
    assert cookie.split(b"=", 1)[1] not in messages[1]["body"]


@pytest.mark.parametrize("peer", ["192.168.1.2", "203.0.113.1", "invalid", None])
def test_peer_required_even_on_public_health(peer):
    probe = ProbeApp()
    messages, reads = request(LocalWebSecurity(probe), path="/api/health", peer=peer)
    assert messages[0]["status"] == 403 and not reads and probe.calls == 0


@pytest.mark.parametrize("path", ["/", "/api/health", "/static/app.js"])
def test_public_get_still_checks_host(path):
    probe = ProbeApp()
    app = LocalWebSecurity(probe)
    allowed, _ = request(app, path=path)
    assert allowed[0]["status"] == 200
    before = probe.calls
    denied, reads = request(app, path=path, headers=[(b"host", b"evil.example:7861")])
    assert denied[0]["status"] == 403 and not reads and probe.calls == before


@pytest.mark.parametrize("path", ["/api/images/id", "/api/jobs/id/events", "/api/models", "/static/../api/private"])
def test_personal_get_needs_cookie(path):
    probe = ProbeApp()
    messages, reads = request(LocalWebSecurity(probe), path=path)
    assert messages[0]["status"] == 403 and not reads and probe.calls == 0


@pytest.mark.parametrize("site,mode,destination", [
    (b"same-origin", b"navigate", b"iframe"), (b"none", b"no-cors", b"image"),
    (b"cross-site", b"navigate", b"document"), (b"same-site", b"navigate", b"document"),
    (b"same-origin", None, None),
])
def test_bootstrap_rejects_embedded_or_foreign_navigation(site, mode, destination):
    probe = ProbeApp()
    headers = [(b"host", b"127.0.0.1:7861"), (b"sec-fetch-site", site)]
    if mode:
        headers.append((b"sec-fetch-mode", mode))
    if destination:
        headers.append((b"sec-fetch-dest", destination))
    messages, reads = request(LocalWebSecurity(probe), path="/", headers=headers)
    assert messages[0]["status"] == 403 and not reads and probe.calls == 0
    assert b"set-cookie" not in dict(messages[0]["headers"])


def test_new_process_invalidates_old_cookie_and_page_restores_access():
    old = LocalWebSecurity(ProbeApp())
    new = LocalWebSecurity(ProbeApp())
    old_cookie = session(old)
    denied, _ = request(new, headers=authorized_headers(old_cookie))
    assert denied[0]["status"] == 403
    new_cookie = session(new, headers=authorized_headers(old_cookie))
    assert new_cookie != old_cookie
    allowed, _ = request(new, headers=authorized_headers(new_cookie))
    assert allowed[0]["status"] == 200


def test_ipv6_and_configured_port_are_canonical():
    app = LocalWebSecurity(ProbeApp(), host="::1", port=7862)
    cookie = session(app, peer="::1", headers=[(b"host", b"[::1]:7862")])
    assert cookie.startswith(b"ocr_local_7862=")
    allowed, _ = request(app, method="POST", peer="::1", headers=[
        (b"host", b"[::1]:7862"), (b"origin", b"http://[::1]:7862"), (b"cookie", cookie),
    ])
    assert allowed[0]["status"] == 200
    denied, _ = request(app, path="/api/health", peer="::1", headers=[(b"host", b"::1:7862")])
    assert denied[0]["status"] == 403


def test_secret_redaction_in_formatted_arguments_and_exception():
    app = LocalWebSecurity(ProbeApp())
    second_app = LocalWebSecurity(ProbeApp())
    token = session(app).split(b"=", 1)[1].decode()
    second_token = session(second_app).split(b"=", 1)[1].decode()
    output = io.StringIO()
    logger = logging.getLogger("ocr_local.security_test")
    handler = logging.StreamHandler(output)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    original_propagate = logger.propagate
    logger.propagate = False
    all_loggers = [logging.getLogger()] + [
        item for item in logging.Logger.manager.loggerDict.values() if isinstance(item, logging.Logger)
    ]
    originals = {item: item.formatter for log in all_loggers for item in log.handlers}
    try:
        install_secret_log_redaction()
        install_secret_log_redaction()
        logger.warning("Cookies: %s %s", token, second_token)
        try:
            raise ValueError("diagnostic " + token)
        except ValueError:
            logger.exception("Exception with %s", {"cookie": token})
        rendered = output.getvalue()
        assert token not in rendered and second_token not in rendered
        assert "[СКРЫТО]" in rendered and "Traceback" in rendered and "diagnostic" in rendered
    finally:
        for item, formatter in originals.items():
            item.setFormatter(formatter)
        logger.removeHandler(handler)
        logger.propagate = original_propagate


def test_access_denial_diagnostic_can_be_found_in_local_log(caplog):
    messages, _ = request(LocalWebSecurity(ProbeApp()), path="/api/models")
    payload = json.loads(messages[1]["body"])
    assert messages[0]["status"] == 403
    assert payload["diagnostic_id"] in caplog.text and "access_denied" in caplog.text
