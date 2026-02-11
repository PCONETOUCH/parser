@echo off
setlocal EnableExtensions

REM Run all parsers via launcher.py (v2).

call "%~dp0bootstrap.bat"
if errorlevel 1 exit /b 1

echo.
echo ================================================
echo Running launcher (all parsers)...
echo ================================================
echo.

"%~dp0..\.venv\Scripts\python.exe" "%~dp0launcher.py" --config "%~dp0launcher_config.json" --mode all

echo.
echo DONE. Press any key to close.
pause >nul
