@echo off
setlocal EnableExtensions

REM Bootstraps a shared virtualenv for the whole suite.
REM Venv location: <SUITE_ROOT>\.venv

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "SUITE_ROOT=%%~fI"
set "VENV_DIR=%SUITE_ROOT%\.venv"
set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
set "REQ_FILE=%SCRIPT_DIR%requirements.txt"

echo [bootstrap] SUITE_ROOT=%SUITE_ROOT%
echo [bootstrap] VENV_DIR=%VENV_DIR%

if not exist "%VENV_DIR%" (
  echo [bootstrap] Creating venv...
  python -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [bootstrap] ERROR: failed to create venv. Ensure Python is installed and on PATH.
    exit /b 1
  )
)

echo [bootstrap] Upgrading pip...
"%PY_EXE%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [bootstrap] Installing requirements from %REQ_FILE% ...
"%PY_EXE%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 exit /b 1

echo [bootstrap] OK
endlocal & exit /b 0
