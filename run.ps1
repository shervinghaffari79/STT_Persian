# Launch the local Persian SOTA ASR app on Windows.
#
# Only the FRONTEND (Vite) is network/internet-exposed, on 0.0.0.0:5000
# (override with $env:FRONTEND_PORT). The FastAPI backend stays on
# 127.0.0.1:8000 (override with $env:BACKEND_PORT) and is reached only
# through the frontend's built-in proxy -- it is never reachable directly
# from outside this machine. There is currently NO authentication on the
# backend's API, so put a firewall/VPN/security-group rule in front of :5000
# before exposing it beyond a trusted network (see README.md's Windows
# Firewall notes).
#
# On Windows the ASR/chat backends auto-select the cross-platform path
# (faster-whisper/CTranslate2 + transformers) instead of MLX, which is
# Apple-Silicon only -- see backend/asr_engine.py and backend/chat.py.
# If an NVIDIA GPU + CUDA are present they are used automatically.
#
# Usage:  .\run.ps1        (Ctrl-C stops both)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $env:BACKEND_PORT) { $env:BACKEND_PORT = "8000" }
if (-not $env:FRONTEND_PORT) { $env:FRONTEND_PORT = "5000" }
$backendPort = $env:BACKEND_PORT
$frontendPort = $env:FRONTEND_PORT

Write-Host "-> starting backend (FastAPI, 127.0.0.1:$backendPort, not exposed) ..."
# Start-Process has no -Environment parameter; child processes inherit the
# parent's environment automatically, so set these here first.
$env:HOST = "127.0.0.1"
$env:PORT = $backendPort
$backend = Start-Process -FilePath "python" -ArgumentList "server.py" `
    -WorkingDirectory (Join-Path $PSScriptRoot "backend") -PassThru -NoNewWindow

function Cleanup {
    Write-Host "`nstopping..."
    if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
}
Register-EngineEvent PowerShell.Exiting -Action { Cleanup } | Out-Null
trap { Cleanup; break }

# wait for backend health
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:$backendPort/api/health" -TimeoutSec 2 | Out-Null
        $ready = $true
        break
    } catch { Start-Sleep -Seconds 1 }
}
if ($ready) { Write-Host "backend ready" } else { Write-Host "backend did not respond in time -- check backend logs" }

Write-Host "-> starting frontend (Vite, 0.0.0.0:$frontendPort, network-exposed) ..."
try {
    npm run dev
} finally {
    Cleanup
}
