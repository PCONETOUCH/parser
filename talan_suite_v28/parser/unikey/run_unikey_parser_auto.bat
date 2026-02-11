@echo off
setlocal
cd /d %~dp0

REM Run UNIKEY parser with auto-refresh for signed API URL.
REM If token is invalid, the parser can open a browser and you can manually navigate
REM to the catalog / chessboard page to help it capture a fresh signed URL.

call "%~dp0..\..\launcher\bootstrap.bat"
if errorlevel 1 exit /b 1

if not exist "unikey_config.json" (
  echo Creating unikey_config.json from example...
  copy /y "unikey_config.example.json" "unikey_config.json" >nul
)

echo.
echo Running UNIKEY parser...
echo.

"%~dp0..\..\.venv\Scripts\python.exe" "%~dp0unikey_parser.py" --config "%~dp0unikey_config.json"

echo.
echo DONE. Press any key to close.
pause >nul
