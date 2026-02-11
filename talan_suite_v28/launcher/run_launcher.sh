#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

bash ./bootstrap.sh
PY="../.venv/bin/python"
"$PY" launcher.py --config "./launcher_config.json"
