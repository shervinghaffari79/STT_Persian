@echo off
REM Launch the local Persian SOTA ASR app on Windows (cmd.exe).
REM
REM Only the FRONTEND (Vite) is network/internet-exposed, on 0.0.0.0:5000
REM (override with FRONTEND_PORT). The FastAPI backend stays on
REM 127.0.0.1:8000 (override with BACKEND_PORT) and is reached only
REM through the frontend's built-in proxy -- it is never reachable directly
REM from outside this machine. There is currently NO authentication on the
REM backend's API, so put a firewall/VPN/security-group rule in front of
REM :5000 before exposing it beyond a trusted network (see README.md's
REM Windows Firewall notes).
REM
REM On Windows the ASR/chat backends auto-select the cross-platform path
REM (faster-whisper/CTranslate2 + transformers) instead of MLX, which is
REM Apple-Silicon only -- see backend\asr_engine.py and backend\chat.py.
REM If an NVIDIA GPU + CUDA are present they are used automatically.
REM
REM Usage:  run.bat        (Ctrl-C stops both; the backend window closes too)

setlocal enabledelayedexpansion
cd /d "%~dp0"

if not defined BACKEND_PORT set "BACKEND_PORT=8000"
if not defined FRONTEND_PORT set "FRONTEND_PORT=5000"

set "BACKEND_TITLE=STT_Backend_%RANDOM%"

echo -^> starting backend (FastAPI, 127.0.0.1:%BACKEND_PORT%, not exposed) ...
REM Set these in THIS process; the started child inherits them. Doing it inside
REM the child's command line instead needs `cmd /c "cd /d "..." && set ..."`,
REM whose nested quotes cmd parses unreliably. Always the quoted `set "X=Y"`
REM form: the bare `set X=Y && ...` form captures the space before `&&` INTO
REM the value, and a HOST of "127.0.0.1 " fails to resolve.
set "HOST=127.0.0.1"
set "PORT=%BACKEND_PORT%"
start "%BACKEND_TITLE%" /MIN /D "%~dp0backend" cmd /c python server.py

REM wait for backend health
set "READY=0"
for /L %%i in (1,1,30) do (
    curl -sf "http://127.0.0.1:%BACKEND_PORT%/api/health" >nul 2>&1
    if not errorlevel 1 (
        set "READY=1"
        goto :ready
    )
    timeout /t 1 /nobreak >nul
)
:ready
if "%READY%"=="1" (
    echo backend ready
) else (
    echo backend did not respond in time -- check backend logs
)

echo -^> starting frontend (Vite, 0.0.0.0:%FRONTEND_PORT%, network-exposed) ...
REM BACKEND_PORT/FRONTEND_PORT are already set in this process and npm inherits
REM them; vite.config.ts reads both.
call npm run dev

echo.
echo stopping...
taskkill /FI "WINDOWTITLE eq %BACKEND_TITLE%*" /T /F >nul 2>&1

endlocal
