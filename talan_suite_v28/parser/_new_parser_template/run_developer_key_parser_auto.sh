#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Auto-run template parser using suite-wide .venv (created by launcher/bootstrap.bat on Windows).
# On Linux/macOS you can create your own venv and install requirements:
#   python3 -m venv ../..\.venv && ../..\.venv/bin/pip install -r ../../launcher/requirements.txt

PY="../../.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "[run] .venv not found at ../../.venv. Create it first (see comment in this file)." >&2
  exit 1
fi

"$PY" "./developer_key_parser.py" --config "./developer_key_config.json"
