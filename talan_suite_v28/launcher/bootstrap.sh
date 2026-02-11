#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a shared virtualenv for the whole suite.
# Venv location: <SUITE_ROOT>/.venv

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUITE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SUITE_ROOT/.venv"
PY_EXE="$VENV_DIR/bin/python"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

echo "[bootstrap] SUITE_ROOT=$SUITE_ROOT"
echo "[bootstrap] VENV_DIR=$VENV_DIR"

if [ ! -d "$VENV_DIR" ]; then
  echo "[bootstrap] Creating venv..."
  python3 -m venv "$VENV_DIR" || python -m venv "$VENV_DIR"
fi

echo "[bootstrap] Upgrading pip..."
"$PY_EXE" -m pip install --upgrade pip

echo "[bootstrap] Installing requirements from $REQ_FILE ..."
"$PY_EXE" -m pip install -r "$REQ_FILE"

echo "[bootstrap] OK"
