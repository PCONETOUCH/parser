from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from .readers import safe_resolve
from .settings import SETTINGS


def manual_accept(run_id: str, snapshot_name: str, reason: str, operator: str) -> dict:
    if not reason.strip():
        raise ValueError("Reason is required")
    q_root = SETTINGS.launcher_data_root / "quarantine"
    p_root = SETTINGS.launcher_data_root / "published_snapshots"
    src = safe_resolve(q_root / run_id, snapshot_name)
    dst_dir = p_root / run_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    method = "fallback"
    script_path = SETTINGS.launcher_root / "manual_accept.py"
    if script_path.exists():
        try:
            subprocess.run(
                ["python", str(script_path), "--run-id", run_id, "--snapshot", src.name, "--reason", reason, "--operator", operator],
                cwd=str(SETTINGS.launcher_root),
                check=True,
                capture_output=True,
                text=True,
            )
            method = "script"
        except Exception:
            method = "fallback"

    if method == "fallback":
        shutil.copy2(src, dst)

    audit_path = SETTINGS.launcher_data_root / "reports" / run_id / "manual_accept.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.utcnow().isoformat(),
        "run_id": run_id,
        "developer_key": src.name.split("__")[0],
        "snapshot_name": src.name,
        "reason": reason,
        "operator": operator,
        "method": method,
    }
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
