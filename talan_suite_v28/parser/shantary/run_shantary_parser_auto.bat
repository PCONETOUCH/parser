@echo off
setlocal
cd /d %~dp0

REM Run SHANTARY parser (shantary.ru) with auto-refresh for signed API URL.
REM If token is invalid, the parser can open a browser; follow on-screen instructions.

call "%~dp0..\..\launcher\bootstrap.bat"
if errorlevel 1 exit /b 1

if not exist "shantary_config.json" (
  echo Creating shantary_config.json from example...
  copy /y "shantary_config.example.json" "shantary_config.json" >nul
)

echo.
echo Running SHANTARY parser...
echo.

"%~dp0..\..\.venv\Scripts\python.exe" "%~dp0shantary_parser.py" --config "%~dp0shantary_config.json"

echo.
echo DONE. Press any key to close.
pause >nul
