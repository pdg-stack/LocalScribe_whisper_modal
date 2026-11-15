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
rem Runs uvicorn through PowerShell so its console output (request lines,
rem unhandled errors) is shown in the window AND appended to
rem logs\server.log as UTF-8 (Tee-Object in Windows PowerShell 5.1 writes
rem UTF-16, which looks like garbage in editors/grep; an existing UTF-16 log
rem is converted once on startup). cmd /c does the 2>&1 merge so PowerShell only sees
rem plain stdout (avoids its NativeCommandError noise on stderr).
rem uvicorn_log_config.json adds a timestamp to every uvicorn line (incl. its
rem own "Started server process" / "Shutting down" messages); the two
rem ===== lines below mark the launcher starting the server and the server
rem process exiting (with its exit code), so every run is bracketed in the file.
start "LocalScribe server" powershell -NoProfile -NoExit -Command "Set-Location '%APP_DIR%'; New-Item -ItemType Directory -Force logs | Out-Null; $f='%APP_DIR%logs\server.log'; if (Test-Path $f) { $b=[IO.File]::ReadAllBytes($f); if ($b.Length -ge 2 -and $b[0] -eq 255 -and $b[1] -eq 254) { [IO.File]::WriteAllText($f, [Text.Encoding]::Unicode.GetString($b,2,$b.Length-2)) } }; function L { process { $_; [IO.File]::AppendAllText($f, [string]$_ + [Environment]::NewLine) } }; ('{0} ===== LocalScribe server starting =====' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) | L; cmd /c '.venv\Scripts\python.exe -m uvicorn backend.main:app --log-config uvicorn_log_config.json 2>&1' | L; ('{0} ===== LocalScribe server stopped (exit code {1}) =====' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $LASTEXITCODE) | L"

echo Waiting for the server to come up...
:waitloop
timeout /t 1 /nobreak >nul
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not %ERRORLEVEL%==0 goto waitloop

start "" "%URL%"

endlocal
