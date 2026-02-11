@echo off
setlocal EnableExtensions
cd /d "%~dp0"

call "%~dp0bootstrap.bat"
if errorlevel 1 exit /b 1

set "PY=..\.venv\Scripts\python.exe"
"%PY%" manual_accept.py %*

endlocal
