param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7861,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSCommandPath
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$displayHost = $HostAddress
if ($displayHost -eq "0.0.0.0") {
    $displayHost = "127.0.0.1"
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
    "uvicorn",
    "ocr_local.web.app:app",
    "--host",
    $HostAddress,
    "--port",
    $Port.ToString()
)

if ($Reload) {
    $arguments += "--reload"
}

Write-Host "OCR Local web: $serviceUrl"
& $PythonPath @arguments
exit $LASTEXITCODE
