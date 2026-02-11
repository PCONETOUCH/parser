@echo off
setlocal EnableExtensions
cd /d "%~dp0"

call "%~dp0bootstrap.bat"
if errorlevel 1 exit /b 1

REM Run new launcher (v2)
set "PY=..\.venv\Scripts\python.exe"
"%PY%" launcher.py --config "%~dp0launcher_config.json"

endlocal
