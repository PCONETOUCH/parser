#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Run SHANTARY parser (shantary.ru)
bash "../../launcher/bootstrap.sh"

if [ ! -f "shantary_config.json" ]; then
  cp -f "shantary_config.example.json" "shantary_config.json"
fi

../../.venv/bin/python3 "shantary_parser.py" --config "shantary_config.json"
