@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Auto-run template parser using suite-wide .venv (created by launcher\bootstrap.bat)
call "%~dp0..\..\launcher\bootstrap.bat"
if errorlevel 1 exit /b 1

set "PY=%~dp0..\..\.venv\Scripts\python.exe"
"%PY%" "%~dp0developer_key_parser.py" --config "%~dp0developer_key_config.json"

endlocal
