@echo off
setlocal

rem Double-click launcher: starts the local server (if it isn't already
rem running) and opens the app in your browser. Self-locating via %~dp0
rem (this script's own folder) so it works wherever this repo is cloned --
rem no machine-specific path to edit. Point a Desktop shortcut at this
rem file rather than moving/copying the .bat itself, so it stays inside
rem the repo (and under version control) for anyone who sets this project
rem up later.

set PORT=8000
set APP_DIR=%~dp0
set URL=http://localhost:%PORT%/

powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if %ERRORLEVEL%==0 (
    echo LocalScribe is already running - opening it in your browser...
    start "" "%URL%"
    goto :eof
)

echo Starting LocalScribe...
start "LocalScribe server" cmd /k "cd /d "%APP_DIR%" && call .venv\Scripts\activate.bat && uvicorn backend.main:app"

echo Waiting for the server to come up...
:waitloop
timeout /t 1 /nobreak >nul
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not %ERRORLEVEL%==0 goto waitloop

start "" "%URL%"

endlocal
