@echo off
setlocal
REM Run orchestrator for TALAN only (publish/quarantine pipeline)
cd /d "%~dp0..\.."
call launcher\bootstrap.bat
if errorlevel 1 exit /b 1
"%CD%\.venv\Scripts\python.exe" "launcher\run_orchestrator.py" --config "launcher\launcher_config.json" --only talan
echo.
echo DONE. Press any key to close.
pause >nul
endlocal
