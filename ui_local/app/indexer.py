from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .settings import SETTINGS

CACHE_PATH = Path(__file__).resolve().parents[1] / ".local" / "index_cache.json"


def _dir_mtime(path: Path) -> float:
    try:
        return max([p.stat().st_mtime for p in path.iterdir()] + [path.stat().st_mtime])
    except Exception:
        return 0.0


def build_runs_index() -> dict[str, Any]:
    root = SETTINGS.launcher_data_root
    pub_root = root / "published_snapshots"
    qua_root = root / "quarantine"
    rep_root = root / "reports"
    runs = sorted({p.name for p in pub_root.glob("*") if p.is_dir()} | {p.name for p in qua_root.glob("*") if p.is_dir()}, reverse=True)
    rows = []
    for run in runs:
        pub_count = len(list((pub_root / run).glob("*.csv")))
        qua_count = len(list((qua_root / run).glob("*.csv")))
        rows.append({"run_id": run, "published_count": pub_count, "quarantine_count": qua_count, "report_dir": str(rep_root / run)})
    payload = {
        "source_mtime": {"published": _dir_mtime(pub_root), "quarantine": _dir_mtime(qua_root), "reports": _dir_mtime(rep_root)},
        "runs": rows,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_runs_index() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return build_runs_index()
    try:
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return build_runs_index()
    fresh = {
        "published": _dir_mtime(SETTINGS.launcher_data_root / "published_snapshots"),
        "quarantine": _dir_mtime(SETTINGS.launcher_data_root / "quarantine"),
        "reports": _dir_mtime(SETTINGS.launcher_data_root / "reports"),
    }
    if cached.get("source_mtime") != fresh:
        return build_runs_index()
    return cached
