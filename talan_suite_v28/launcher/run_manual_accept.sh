#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

bash ./bootstrap.sh || true

PY="../.venv/bin/python"
"$PY" manual_accept.py "$@"
