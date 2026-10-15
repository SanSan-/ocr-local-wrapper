param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7861,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

if ($HostAddress -cne "127.0.0.1" -and $HostAddress -cne "::1") {
    throw "Only the loopback addresses 127.0.0.1 and ::1 are supported."
}
if ($Port -lt 1 -or $Port -gt 65535) {
    throw "Port must be between 1 and 65535."
}
if ($Reload) {
    throw "Reload is not supported: the local session requires one stable process."
}

$ProjectRoot = Split-Path -Parent $PSCommandPath
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$displayHost = $HostAddress
if ($displayHost -eq "::1") {
    $displayHost = "[::1]"
}
$serviceUrl = "http://${displayHost}:$Port"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Virtual environment Python was not found: $PythonPath"
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "$serviceUrl/api/health" -TimeoutSec 2
        if ($response.StatusCode -eq 200 -and $response.Content -match '"status"\s*:\s*"ok"') {
            Write-Host "OCR Local web already running: $serviceUrl"
            exit 0
        }
    } catch {
    }
    throw "Port $Port is busy by PID=$($listener.OwningProcess). Use another port: .\run_web.ps1 -Port 7862"
}

Set-Location -LiteralPath $ProjectRoot

$arguments = @(
    "-B",
    "-m",
    "ocr_local.web",
    "--host",
    $HostAddress,
    "--port",
    $Port.ToString()
)

Write-Host "OCR Local web: $serviceUrl"
& $PythonPath @arguments
exit $LASTEXITCODE
