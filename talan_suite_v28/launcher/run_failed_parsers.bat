@echo off
setlocal EnableExtensions

REM Re-run only parsers that were NOT OK in last run (launcher/data/last_run.json).

call "%~dp0bootstrap.bat"
if errorlevel 1 exit /b 1

echo.
echo ================================================
echo Running launcher (only_failed)...
echo ================================================
echo.

"%~dp0..\.venv\Scripts\python.exe" "%~dp0launcher.py" --config "%~dp0launcher_config.json" --mode only_failed

echo.
echo DONE. Press any key to close.
pause >nul
