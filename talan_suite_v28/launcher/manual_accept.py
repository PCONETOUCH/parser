#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual accept: move snapshot from quarantine/<RUN_ID>/ to published_snapshots/<RUN_ID>/.

Use case:
- Launcher quarantined a snapshot (WARN/FAIL).
- Human проверил snapshot и принимает решение опубликовать.

This script:
- Copies the snapshot file (no subfolders) from quarantine to published for the same RUN_ID.
- Writes acceptance record into launcher/data/reports/<RUN_ID>/manual_accept.jsonl
- Updates launcher/data/published_state.json baseline pointer for this developer_key.

Важно: Скрипт НЕ удаляет исходный файл из quarantine (след может быть полезен).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from typing import Any, Dict, Optional


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_json(path: str, default: Optional[dict] = None) -> dict:
    if not os.path.exists(path):
        return default or {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Manual accept quarantined snapshot into published folder")
    ap.add_argument("--run_id", required=True, help="RUN_ID of launcher run, e.g. 20260210_120501")
    ap.add_argument("--developer_key", required=True, help="Parser key, e.g. talan / unikey / <new>")
    ap.add_argument("--file", default="", help="Optional exact file name to accept (if multiple).")
    ap.add_argument("--reason", required=True, help="Human reason why this snapshot is accepted")
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_id = args.run_id.strip()
    key = args.developer_key.strip()

    qdir = os.path.join(repo_root, "launcher", "data", "quarantine", run_id)
    pdir = os.path.join(repo_root, "launcher", "data", "published_snapshots", run_id)
    ensure_dir(pdir)

    if not os.path.isdir(qdir):
        print(f"Quarantine folder not found: {qdir}")
        return 2

    # find candidates
    cands = [f for f in os.listdir(qdir) if f.lower().endswith(".csv") and f.startswith(f"{key}__")]
    if args.file:
        cands = [f for f in cands if f == args.file]
    if not cands:
        print(f"No snapshot found for key={key} in {qdir}")
        return 3
    if len(cands) > 1:
        print("Multiple candidates found, specify --file:")
        for f in cands:
            print(" -", f)
        return 4

    fn = cands[0]
    src = os.path.join(qdir, fn)
    dst = os.path.join(pdir, fn)

    shutil.copy2(src, dst)
    print(f"Accepted: {dst}")

    # update baseline pointer
    state_path = os.path.join(repo_root, "launcher", "data", "published_state.json")
    state = read_json(state_path, default={})
    if not isinstance(state, dict):
        state = {}
    state[key] = os.path.relpath(dst, repo_root).replace("\\", "/")
    write_json(state_path, state)

    # write acceptance record
    record = {
        "ts": dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "run_id": run_id,
        "developer_key": key,
        "file": fn,
        "from": os.path.relpath(src, repo_root).replace("\\", "/"),
        "to": os.path.relpath(dst, repo_root).replace("\\", "/"),
        "reason": args.reason.strip(),
    }
    reports_dir = os.path.join(repo_root, "launcher", "data", "reports", run_id)
    ensure_dir(reports_dir)
    append_jsonl(os.path.join(reports_dir, "manual_accept.jsonl"), record)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
