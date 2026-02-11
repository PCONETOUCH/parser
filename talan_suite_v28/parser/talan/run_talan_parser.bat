@echo off
setlocal
cd /d %~dp0

REM Run TALAN parser directly (not via launcher).

call "%~dp0..\..\launcher\bootstrap.bat"
if errorlevel 1 exit /b 1

echo.
echo Running TALAN parser...
echo.

"%~dp0..\..\.venv\Scripts\python.exe" "%~dp0talan_parser.py"

echo.
echo DONE. Press any key to close.
pause >nul
