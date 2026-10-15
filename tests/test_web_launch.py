from __future__ import annotations

import base64
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ocr_local.web import __main__ as launcher
from ocr_local.web import app as web_app


@pytest.mark.parametrize("host,port", [("127.0.0.1", 7861), ("::1", 7862)])
def test_module_launch_uses_same_canonical_configuration(monkeypatch, host, port):
    created = []
    runs = []
    application = object()

    def create_app(**config):
        created.append(config)
        return application

    monkeypatch.setattr(web_app, "create_app", create_app, raising=False)
    monkeypatch.setattr(launcher.uvicorn, "run", lambda *args, **kwargs: runs.append((args, kwargs)))
    launcher.main(["--host", host, "--port", str(port)])
    assert created == [{"host": host, "port": port}]
    assert runs == [((application,), {
        "host": host, "port": port, "reload": False, "workers": 1, "proxy_headers": False,
    })]


@pytest.mark.parametrize("arguments", [
    ["--host", "0.0.0.0"], ["--host", "localhost"], ["--host", "192.168.1.2"],
    ["--port", "0"], ["--port", "65536"], ["--reload"],
])
def test_invalid_module_launch_rejected_before_create_or_listen(monkeypatch, arguments):
    def unexpected(*args, **kwargs):
        pytest.fail("Некорректный запуск не должен создавать приложение или слушатель")

    monkeypatch.setattr(web_app, "create_app", unexpected, raising=False)
    monkeypatch.setattr(launcher.uvicorn, "run", unexpected)
    with pytest.raises(SystemExit) as error:
        launcher.main(arguments)
    assert error.value.code == 2


POWERSHELL_51 = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
RUN_WEB = Path(__file__).resolve().parents[1] / "run_web.ps1"


def powershell(command):
    if os.name != "nt" or not POWERSHELL_51.is_file():
        pytest.skip("Проверка сценария требует Windows PowerShell 5.1")
    source = (
        "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)\n"
        "Write-Output ('PS_VERSION=' + $PSVersionTable.PSVersion.ToString())\n" + command
    )
    encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        [str(POWERSHELL_51), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True, encoding="utf-8", errors="replace", timeout=30,
    )
    assert "PS_VERSION=5.1." in result.stdout
    return result


@pytest.mark.parametrize("arguments,reason", [
    ("-HostAddress '0.0.0.0'", "loopback"),
    ("-HostAddress 'localhost'", "loopback"),
    ("-HostAddress '192.168.1.2'", "loopback"),
    ("-Port 0", "Port must"),
    ("-Port 65536", "Port must"),
    ("-Reload", "Reload is not supported"),
])
def test_powershell_rejects_invalid_launch_before_resources(arguments, reason):
    script = str(RUN_WEB).replace("'", "''")
    result = powershell(
        "function Test-Path { throw 'UNEXPECTED_RESOURCE_ACCESS' }\n"
        "function Get-NetTCPConnection { throw 'UNEXPECTED_SOCKET_ACCESS' }\n"
        f"& '{script}' {arguments}\n"
    )
    assert result.returncode != 0
    assert reason in result.stderr
    assert "UNEXPECTED_" not in result.stderr


@pytest.mark.parametrize("host,authority", [("127.0.0.1", "127.0.0.1"), ("::1", "[::1]")])
def test_powershell_health_probe_uses_configured_authority_without_network(host, authority):
    script = str(RUN_WEB).replace("'", "''")
    result = powershell(
        "function Test-Path { return $true }\n"
        "function Get-NetTCPConnection { return [pscustomobject]@{ OwningProcess = 123 } }\n"
        "function Invoke-WebRequest {\n"
        "  param([switch]$UseBasicParsing, [string]$Uri, [int]$TimeoutSec)\n"
        "  Write-Host ('HEALTH_URI=' + $Uri)\n"
        "  return [pscustomobject]@{ StatusCode = 200; Content = '{\"status\":\"ok\"}' }\n"
        "}\n"
        f"& '{script}' -HostAddress '{host}' -Port 7862\n"
    )
    assert result.returncode == 0, result.stderr
    assert f"HEALTH_URI=http://{authority}:7862/api/health" in result.stdout
    assert f"already running: http://{authority}:7862" in result.stdout


@pytest.mark.parametrize("host,port", [("127.0.0.1", 7861), ("127.0.0.1", 0), ("::1", 0)])
def test_real_loopback_listener_from_powershell_51(tmp_path, host, port):
    """Штатный Python-запуск с пустым приложением и ограниченным сроком сервера."""
    runner = tmp_path / "bounded_server.py"
    runner.write_text('''from __future__ import annotations
import http.client
import json
import socket
import sys
import threading
import time
import types

sys.path.insert(0, sys.argv.pop(1))
from ocr_local.web.security import LocalWebSecurity
from ocr_local.web import __main__ as launcher

host = sys.argv[1]
port = int(sys.argv[2])
family = socket.AF_INET6 if host == "::1" else socket.AF_INET
with socket.socket(family, socket.SOCK_STREAM) as probe:
    try:
        probe.bind((host, port))
    except OSError:
        print("LISTENER_UNAVAILABLE")
        raise SystemExit(77)
    port = probe.getsockname()[1]

async def empty_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b'{"status":"ok"}'})

module = types.ModuleType("ocr_local.web.app")
module.create_app = lambda **config: LocalWebSecurity(empty_app, **config)
sys.modules["ocr_local.web.app"] = module
report = {}
errors = []

def bounded_run(app, **options):
    server = launcher.uvicorn.Server(launcher.uvicorn.Config(
        app, **options, lifespan="off", log_level="critical"
    ))

    def client():
        try:
            deadline = time.monotonic() + 5
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.02)
            assert server.started, "Server did not start"
            connection = http.client.HTTPConnection(host, port, timeout=2)
            connection.request("GET", "/api/health")
            response = connection.getresponse()
            assert response.status == 200
            response.read()
            connection.request("GET", "/")
            response = connection.getresponse()
            assert response.status == 200
            cookie = response.getheader("Set-Cookie").split(";", 1)[0]
            response.read()
            connection.request("GET", "/api/private", headers={"Cookie": cookie})
            response = connection.getresponse()
            assert response.status == 200
            response.read()
            connection.request("POST", "/api/private", body=b"blocked", headers={
                "Cookie": cookie, "Origin": "https://example.org"
            })
            response = connection.getresponse()
            assert response.status == 403
            assert json.loads(response.read())["code"] == "access_denied"
            connection.close()
            report.update(host=host, port=port, health=200, authorized=200, denied=403)
        except BaseException as exc:
            errors.append(str(exc))
        finally:
            server.should_exit = True

    watchdog = threading.Timer(10, lambda: setattr(server, "should_exit", True))
    watchdog.daemon = True
    worker = threading.Thread(target=client, daemon=True)
    watchdog.start()
    worker.start()
    try:
        server.run()
    finally:
        server.should_exit = True
        worker.join(timeout=3)
        watchdog.cancel()
    assert not errors, errors
    assert report, "No client report"

launcher.uvicorn.run = bounded_run
launcher.main(["--host", host, "--port", str(port)])
print("LISTENER_RESULT=" + json.dumps(report))
''', encoding="utf-8", newline="\n")
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    result = powershell(
        f"& {quote(sys.executable)} -X utf8 -B {quote(runner)} "
        f"{quote(RUN_WEB.parent)} {quote(host)} {port}\nexit $LASTEXITCODE\n"
    )
    if result.returncode == 77:
        pytest.skip(f"Адрес {host}:{port} недоступен для изолированного слушателя")
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"health": 200, "authorized": 200, "denied": 403' in result.stdout
