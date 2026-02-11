from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parents[0]
DEFAULT_LAUNCHER_DATA_ROOT = REPO_ROOT / "talan_suite_v28" / "launcher" / "data"


@dataclass
class UISettings:
    launcher_data_root: Path
    launcher_root: Path
    parser_root: Path
    ignore_statuses: list[str]
    synonyms_override: dict[str, list[str]]


def _load_ui_config() -> dict:
    cfg_path = BASE_DIR / "config" / "ui_config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_settings() -> UISettings:
    cfg = _load_ui_config()
    launcher_data = Path(
        os.environ.get("LAUNCHER_DATA_ROOT")
        or cfg.get("launcher_data_root")
        or DEFAULT_LAUNCHER_DATA_ROOT
    ).resolve()
    launcher_root = launcher_data.parent if launcher_data.name == "data" else launcher_data
    parser_root = Path(cfg.get("parser_root") or (REPO_ROOT / "talan_suite_v28" / "parser")).resolve()
    ignore_statuses = [
        s.lower().strip()
        for s in cfg.get("ignore_statuses", ["sold", "soon", "reserved", "бронь", "продано", "soldout"])
    ]
    return UISettings(
        launcher_data_root=launcher_data,
        launcher_root=launcher_root,
        parser_root=parser_root,
        ignore_statuses=ignore_statuses,
        synonyms_override=cfg.get("synonyms_override", {}),
    )


SETTINGS = load_settings()
