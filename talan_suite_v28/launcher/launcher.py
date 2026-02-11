#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launcher (v2+)

Функции:
- Запуск набора парсеров по launcher_config.json (последовательно или параллельно).
- Поддержка 2 контрактов:
  1) result.json mode (для новых парсеров): status OK/WARN/FAIL + output_snapshot_path (+ reason.json)
  2) legacy-glob mode (для существующих): snapshot_glob/errors_glob + exit_code
- Копирование "зелёных" snapshots в:
    launcher/data/published_snapshots/<RUN_ID>/
  и проблемных в:
    launcher/data/quarantine/<RUN_ID>/
  (без подпапок внутри RUN_ID)
- Полное логирование:
  - общий лог launchera
  - stdout/stderr каждого парсера
  - meta.json по каждому запуску
- Support bundle для WARN/FAIL:
  launcher/data/reports/<RUN_ID>/support_bundles/<developer_key>__<RUN_ID>__.zip
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------- fs utils -----------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_run_id() -> str:
    return dt.datetime.now().astimezone().replace(microsecond=0).strftime("%Y%m%d_%H%M%S")


def run_id_to_iso(run_id: str) -> str:
    # run_id format: YYYYMMDD_HHMMSS
    try:
        d = dt.datetime.strptime(run_id, "%Y%m%d_%H%M%S")
        return d.replace(tzinfo=dt.datetime.now().astimezone().tzinfo).isoformat()
    except Exception:
        return dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def read_json(path: str, default: Optional[dict] = None) -> dict:
    if not os.path.exists(path):
        return default or {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def relposix(repo_root: str, p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    try:
        return os.path.relpath(p, repo_root).replace("\\", "/")
    except Exception:
        return p.replace("\\", "/")


def is_windows() -> bool:
    return os.name == "nt"


# ----------------------------- logging -----------------------------


class Log:
    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path))
        self.path = path
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("")

    def line(self, msg: str) -> None:
        ts = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        s = f"{ts} | {msg}"
        print(s)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(s + "\n")


# ----------------------------- runner -----------------------------


def build_cmd(repo_root: str, runner: dict) -> List[str]:
    rtype = runner.get("type", "python")
    if rtype == "python":
        script = runner["script"]
        args = runner.get("args") or []
        return [sys.executable, os.path.join(repo_root, script), *args]

    if rtype in {"cmd", "bat"}:
        cmd_path = os.path.join(repo_root, runner["cmd"]) if not os.path.isabs(runner["cmd"]) else runner["cmd"]
        if is_windows():
            return ["cmd.exe", "/c", cmd_path]
        return ["bash", cmd_path]

    if rtype == "sh":
        sh_path = os.path.join(repo_root, runner["cmd"]) if not os.path.isabs(runner["cmd"]) else runner["cmd"]
        return ["bash", sh_path]

    raise ValueError(f"Unsupported runner.type={rtype}")



def _ps_quote(s: str) -> str:
    # PowerShell single-quote escaping: ' -> ''
    return "'" + str(s).replace("'", "''") + "'"



def _windows_console_transcript_cmd(cmd: List[str], stdout_path: str) -> List[str]:
    """Wrap a command so that it runs in a visible console window AND logs output.

    Why not pipe to Tee-Object:
      piping makes Python think stdout is not a TTY, which breaks tqdm progress bars.

    We use Start-Transcript to capture everything printed in the console while keeping
    stdout a real terminal for the child process.
    """
    parts = " ".join(_ps_quote(c) for c in cmd)
    ps = (
        "$ErrorActionPreference='Continue'; "
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        "$env:PYTHONIOENCODING='utf-8'; "
        "$env:PYTHONUNBUFFERED='1'; "
        f"$out={_ps_quote(stdout_path)}; "
        "try { Start-Transcript -Path $out -Append -Force | Out-Null } catch {} ; "
        f"& {parts}; "
        "$rc=$LASTEXITCODE; "
        "try { Stop-Transcript | Out-Null } catch {} ; "
        "exit $rc"
    )
    return ["powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps]



def run_process(
    cmd: List[str],
    *,
    cwd: str,
    timeout_sec: int,
    stdout_path: str,
    stderr_path: str,
    meta_path: str,
    env: Optional[Dict[str, str]] = None,
    show_window: bool = False,
) -> Tuple[int, float, float]:
    """Run process.

    Modes:
    - show_window=False: capture stdout/stderr into files (default).
    - show_window=True (Windows): open a new console window for the process, while tee'ing its output into stdout_path.
      stderr_path will contain a short note (stderr is merged into stdout via PowerShell tee).
    Returns: (exit_code, started_at_epoch, finished_at_epoch)
    """
    ensure_dir(os.path.dirname(stdout_path))
    ensure_dir(os.path.dirname(stderr_path))
    ensure_dir(os.path.dirname(meta_path))

    started_at = time.time()

    meta: Dict[str, Any] = {
        "cmd": cmd,
        "cwd": cwd,
        "timeout_sec": timeout_sec,
        "show_window": bool(show_window),
        "started_at": dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
    }

    # Visible console mode (Windows only)
    if show_window and is_windows():
        # Prepare log headers
        with open(stdout_path, "w", encoding="utf-8") as f_out:
            f_out.write("CMD: " + " ".join(cmd) + "\n")
            f_out.write("NOTE: output is captured via PowerShell Start-Transcript to keep stdout a real TTY (tqdm works).\n\n")
        with open(stderr_path, "w", encoding="utf-8") as f_err:
            f_err.write("CMD: " + " ".join(cmd) + "\n\n")
            f_err.write("NOTE: stderr is merged into stdout in show_window mode. See .stdout.log\n")

        wrapped = _windows_console_transcript_cmd(cmd, os.path.abspath(stdout_path))
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        p = subprocess.Popen(
            wrapped,
            cwd=cwd,
            text=True,
            env=env,
            creationflags=creationflags,
        )
        try:
            rc = p.wait(timeout=timeout_sec if timeout_sec > 0 else None)
            finished_at = time.time()
            meta["exit_code"] = rc
            meta["finished_at"] = dt.datetime.now().astimezone().replace(microsecond=0).isoformat()
            meta["duration_sec"] = round(finished_at - started_at, 3)
            write_json(meta_path, meta)
            return rc, started_at, finished_at
        except subprocess.TimeoutExpired:
            # Kill process tree
            try:
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            finished_at = time.time()
            meta["exit_code"] = 124
            meta["timeout"] = True
            meta["finished_at"] = dt.datetime.now().astimezone().replace(microsecond=0).isoformat()
            meta["duration_sec"] = round(finished_at - started_at, 3)
            write_json(meta_path, meta)
            with open(stdout_path, "a", encoding="utf-8") as f_out2:
                f_out2.write("\n\nTIMEOUT\n")
            return 124, started_at, finished_at

    # Default: capture to files
    with open(stdout_path, "w", encoding="utf-8") as f_out, open(stderr_path, "w", encoding="utf-8") as f_err:
        f_out.write("CMD: " + " ".join(cmd) + "\n\n")
        f_err.write("CMD: " + " ".join(cmd) + "\n\n")
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=f_out,
            stderr=f_err,
            text=True,
            env=env,
        )
        try:
            rc = p.wait(timeout=timeout_sec if timeout_sec > 0 else None)
            finished_at = time.time()
            meta["exit_code"] = rc
            meta["finished_at"] = dt.datetime.now().astimezone().replace(microsecond=0).isoformat()
            meta["duration_sec"] = round(finished_at - started_at, 3)
            write_json(meta_path, meta)
            return rc, started_at, finished_at
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass
            finished_at = time.time()
            meta["exit_code"] = 124
            meta["timeout"] = True
            meta["finished_at"] = dt.datetime.now().astimezone().replace(microsecond=0).isoformat()
            meta["duration_sec"] = round(finished_at - started_at, 3)
            write_json(meta_path, meta)
            with open(stderr_path, "a", encoding="utf-8") as f_err2:
                f_err2.write("\n\nTIMEOUT\n")
            return 124, started_at, finished_at



# ----------------------------- snapshot normalization -----------------------------


def normalize_and_copy_csv(
    src_csv: str,
    dst_csv: str,
    *,
    developer_key: str,
    source: str,
    captured_at: str,
) -> None:
    """Copy snapshot into dst, ensuring required columns exist.

    We never edit src file. If src already has required columns, we just copy2.
    If not — we rewrite dst with added columns.
    """
    required = {"developer_key", "captured_at", "source"}

    with open(src_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            raise RuntimeError("empty csv")

    header_set = set(header)
    if required.issubset(header_set):
        ensure_dir(os.path.dirname(dst_csv))
        shutil.copy2(src_csv, dst_csv)
        return

    ensure_dir(os.path.dirname(dst_csv))
    with open(src_csv, "r", encoding="utf-8", newline="") as fin, open(dst_csv, "w", encoding="utf-8", newline="") as fout:
        rr = csv.DictReader(fin)
        cols = list(rr.fieldnames or [])
        out_cols = ["developer_key", "captured_at", "source"] + [c for c in cols if c not in required]
        ww = csv.DictWriter(fout, fieldnames=out_cols)
        ww.writeheader()
        for row in rr:
            row = dict(row)
            row.setdefault("developer_key", developer_key)
            row.setdefault("captured_at", captured_at)
            row.setdefault("source", source)
            ww.writerow({c: row.get(c) for c in out_cols})


# ----------------------------- legacy quality (light) -----------------------------


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(" ", "")
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s.replace(",", "."))
    except Exception:
        return None


def compute_csv_basic_stats(
    path: str,
    *,
    price_col: str = "price",
    area_col: str = "area_m2",
    rooms_col: str = "rooms",
    status_cols: Optional[List[str]] = None,
    price_null_ignore_statuses: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Basic CSV stats with an optional 'effective' price-null rate that ignores
    specified statuses (e.g. sold/soon). Keeps backward-compatible keys.

    - price_null_pct_all: price null % over all rows
    - price_null_pct_effective: price null % over rows NOT in ignore statuses (if enabled)
    - price_null_pct: equals effective if enabled and scope>0 else equals all
    """
    rows = 0
    price_null_all = 0
    area_null = 0
    rooms_null = 0
    areas: List[float] = []
    rooms_map: Dict[str, int] = {}

    # effective scope (optional)
    eff_total = 0
    eff_price_null = 0
    ignore = set((s or "").strip().lower() for s in (price_null_ignore_statuses or []) if str(s).strip() != "")

    with open(path, "r", encoding="utf-8", newline="") as f:
        rr = csv.DictReader(f)
        for row in rr:
            rows += 1

            # status (optional)
            st = None
            if status_cols:
                for sc in status_cols:
                    v = row.get(sc)
                    if v is not None and str(v).strip() != "":
                        st = str(v).strip().lower()
                        break

            pv = safe_float(row.get(price_col))
            if pv is None:
                price_null_all += 1

            # effective price scope: exclude ignored statuses
            if ignore and st is not None and st in ignore:
                pass
            else:
                eff_total += 1
                if pv is None:
                    eff_price_null += 1

            a = safe_float(row.get(area_col))
            if a is None:
                area_null += 1
            else:
                areas.append(a)

            rv = row.get(rooms_col)
            if rv is None or str(rv).strip() == "" or str(rv).strip().lower() in {"none", "null", "nan"}:
                rooms_null += 1
            else:
                k = str(rv).strip()
                rooms_map[k] = rooms_map.get(k, 0) + 1

    def pct(n: int, d: int) -> float:
        return round((n / d * 100.0) if d else 0.0, 3)

    price_null_pct_all = pct(price_null_all, rows)
    price_null_pct_effective = pct(eff_price_null, eff_total) if (ignore and eff_total) else None
    price_null_pct = price_null_pct_effective if price_null_pct_effective is not None else price_null_pct_all

    stats: Dict[str, Any] = {
        "rows": rows,
        "price_null_pct": price_null_pct,
        "price_null_pct_all": price_null_pct_all,
        "price_null_pct_effective": price_null_pct_effective,
        "price_effective_scope_rows": eff_total if ignore else None,
        "area_null_pct": pct(area_null, rows),
        "rooms_null_pct": pct(rooms_null, rows),
        "rooms_dist": rooms_map,
    }

    if areas:
        areas_sorted = sorted(areas)

        def q(p: float) -> float:
            idx = int(round((len(areas_sorted) - 1) * p))
            return float(areas_sorted[max(0, min(len(areas_sorted) - 1, idx))])

        stats["area_q10"] = round(q(0.10), 3)
        stats["area_q50"] = round(q(0.50), 3)
        stats["area_q90"] = round(q(0.90), 3)
        # mode share (rounded)
        from collections import Counter

        c = Counter([round(x, 2) for x in areas_sorted])
        v, cnt = c.most_common(1)[0]
        stats["area_mode"] = float(v)
        stats["area_mode_share"] = round(cnt / len(areas_sorted), 4)
    return stats


def pct_delta(new: float, old: float) -> float:
    if old <= 0:
        return 0.0
    return (new - old) / old * 100.0


# ----------------------------- run model -----------------------------


@dataclass
class ParserOutcome:
    key: str
    developer: str
    exit_code: int
    status: str  # OK/WARN/FAIL
    output_snapshot: Optional[str]
    result_json: Optional[str]
    reason_json: Optional[str]
    reason_code: Optional[str]
    message_for_human: str
    suspect_flags: List[str]
    metrics: Dict[str, Any]
    copied_to: Optional[str]
    proc_stdout: Optional[str]
    proc_stderr: Optional[str]
    proc_meta: Optional[str]
    support_bundle: Optional[str]


# ----------------------------- detectors -----------------------------


def newest_by_mtime(pattern: str, *, min_mtime: Optional[float] = None) -> Optional[str]:
    paths = glob.glob(pattern)
    if not paths:
        return None
    if min_mtime is not None:
        paths = [p for p in paths if os.path.getmtime(p) >= min_mtime]
        if not paths:
            return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def detect_result_json(repo_root: str, key: str, pcfg: dict) -> Optional[str]:
    out_cfg = pcfg.get("output") or {}
    cand = out_cfg.get("result_json")
    if cand:
        p = os.path.join(repo_root, cand) if not os.path.isabs(cand) else cand
        if os.path.exists(p):
            return p

    p2 = os.path.join(repo_root, "parser", key, f"output_{key}", "result.json")
    if os.path.exists(p2):
        return p2

    return None


def detect_reason_json(result_path: str) -> Optional[str]:
    if not result_path:
        return None
    base = os.path.dirname(result_path)
    p = os.path.join(base, "reason.json")
    return p if os.path.exists(p) else None


def detect_snapshot(repo_root: str, pcfg: dict, *, min_mtime: float) -> Optional[str]:
    out_cfg = pcfg.get("output") or {}
    snap_glob = out_cfg.get("snapshot_glob")
    if not snap_glob:
        return None
    g = os.path.join(repo_root, snap_glob)
    return newest_by_mtime(g, min_mtime=min_mtime)


def resolve_rel(repo_root: str, p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(repo_root, p))


def guess_parser_output_root(repo_root: str, key: str) -> Optional[str]:
    cand = os.path.join(repo_root, "parser", key, f"output_{key}")
    if os.path.isdir(cand):
        return cand
    return None


def detect_parser_log(output_root: Optional[str], *, run_id: str, started_epoch: float) -> Optional[str]:
    if not output_root:
        return None
    logs_dir = os.path.join(output_root, "logs")
    if not os.path.isdir(logs_dir):
        return None
    # prefer file containing run_id
    cands = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.lower().endswith(".log")]
    if not cands:
        return None
    by_name = [p for p in cands if run_id in os.path.basename(p)]
    if by_name:
        by_name.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return by_name[0]
    # else newest after start
    cands2 = [p for p in cands if os.path.getmtime(p) >= started_epoch - 5]
    if not cands2:
        cands2 = cands
    cands2.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands2[0]


def detect_errors_csv(output_root: Optional[str]) -> Optional[str]:
    if not output_root:
        return None
    p = os.path.join(output_root, "errors", "errors.csv")
    return p if os.path.exists(p) else None


# ----------------------------- support bundle -----------------------------


def take_file_head(src: str, max_lines: int = 60) -> str:
    lines: List[str] = []
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            lines.append(line)
            if i + 1 >= max_lines:
                break
    return "".join(lines)


def filter_errors_by_run_id(errors_csv_path: str, run_id: str) -> str:
    """Return csv text with only rows matching run_id (if column exists)."""
    try:
        with open(errors_csv_path, "r", encoding="utf-8", newline="") as f:
            rr = csv.DictReader(f)
            if not rr.fieldnames or "run_id" not in rr.fieldnames:
                return take_file_head(errors_csv_path, 200)
            out_lines: List[str] = []
            import io
            buf = io.StringIO()
            ww = csv.DictWriter(buf, fieldnames=rr.fieldnames)
            ww.writeheader()
            for row in rr:
                if str(row.get("run_id") or "").strip() == run_id:
                    ww.writerow(row)
            return buf.getvalue()
    except Exception:
        return ""


def write_zip_text(z: zipfile.ZipFile, arcname: str, text: str) -> None:
    z.writestr(arcname, text.encode("utf-8"))


def add_file_if_exists(z: zipfile.ZipFile, src: Optional[str], arcname: str) -> None:
    if src and os.path.exists(src) and os.path.isfile(src):
        z.write(src, arcname=arcname)


def add_dir_recent(z: zipfile.ZipFile, src_dir: str, arcdir: str, *, started_epoch: float, max_files: int = 50) -> int:
    """Add up to max_files from src_dir modified after started_epoch (fallback: newest)."""
    if not os.path.isdir(src_dir):
        return 0
    files: List[str] = []
    for root, _dirs, fs in os.walk(src_dir):
        for fn in fs:
            p = os.path.join(root, fn)
            if os.path.isfile(p):
                files.append(p)
    if not files:
        return 0
    recent = [p for p in files if os.path.getmtime(p) >= started_epoch - 5]
    if not recent:
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        recent = files[:max_files]
    else:
        recent.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        recent = recent[:max_files]
    for p in recent:
        rel = os.path.relpath(p, src_dir).replace("\\", "/")
        z.write(p, arcname=f"{arcdir}/{rel}")
    return len(recent)


def build_support_bundle(
    *,
    repo_root: str,
    reports_dir: str,
    run_id: str,
    outcome: ParserOutcome,
    launcher_log_path: str,
    parser_log_path: Optional[str],
    errors_csv_path: Optional[str],
    started_epoch: float,
    cfg_bundle: dict,
) -> Optional[str]:
    enabled = bool(cfg_bundle.get("enabled", True))
    include_ok = bool(cfg_bundle.get("include_ok", False))
    if not enabled:
        return None
    if outcome.status == "OK" and not include_ok:
        return None

    bundles_dir = os.path.join(reports_dir, "support_bundles")
    ensure_dir(bundles_dir)

    reason_code = outcome.reason_code or "NA"
    zip_name = f"{outcome.key}__{run_id}__{outcome.status}__{reason_code}.zip"
    zip_path = os.path.join(bundles_dir, zip_name)

    snapshot_path = outcome.copied_to or outcome.output_snapshot
    snapshot_base = os.path.basename(snapshot_path) if snapshot_path else None

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "developer_key": outcome.key,
        "status": outcome.status,
        "reason_code": reason_code,
        "created_at": dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "paths": {
            "snapshot": relposix(repo_root, snapshot_path) if snapshot_path else None,
            "result_json": relposix(repo_root, outcome.result_json) if outcome.result_json else None,
            "reason_json": relposix(repo_root, outcome.reason_json) if outcome.reason_json else None,
            "launcher_log": relposix(repo_root, launcher_log_path),
            "proc_stdout": relposix(repo_root, outcome.proc_stdout) if outcome.proc_stdout else None,
            "proc_stderr": relposix(repo_root, outcome.proc_stderr) if outcome.proc_stderr else None,
            "proc_meta": relposix(repo_root, outcome.proc_meta) if outcome.proc_meta else None,
            "parser_log": relposix(repo_root, parser_log_path) if parser_log_path else None,
            "errors_csv": relposix(repo_root, errors_csv_path) if errors_csv_path else None,
        },
        "notes": [
            "Это support bundle для быстрого разбора WARN/FAIL. Отправь архив разработчику парсера.",
            "Внутри есть reason.json (человеческая причина), логи и небольшая выборка snapshot.",
        ],
    }

    # extra stats (computed by launcher)
    stats: Dict[str, Any] = {}
    price_col = (parser_cfg.get('output') or {}).get('price_col')
    status_cols = (parser_cfg.get('output') or {}).get('status_cols')
    ignore_statuses = (parser_cfg.get('output') or {}).get('price_null_ignore_statuses')
    if snapshot_path and os.path.exists(snapshot_path):
        try:
            stats = compute_csv_basic_stats(snapshot_path, price_col=(price_col or 'price'), status_cols=status_cols, price_null_ignore_statuses=ignore_statuses)
        except Exception:
            stats = {}
    manifest["snapshot_stats"] = stats

    max_head_lines = int(cfg_bundle.get("snapshot_head_lines", 60))
    max_errors_lines = int(cfg_bundle.get("errors_head_lines", 250))
    max_recent_files = int(cfg_bundle.get("max_recent_files", 50))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        write_zip_text(z, "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        # key files
        add_file_if_exists(z, outcome.result_json, "result.json")
        add_file_if_exists(z, outcome.reason_json, "reason.json")

        if snapshot_path and os.path.exists(snapshot_path):
            # include full snapshot (zipped)
            add_file_if_exists(z, snapshot_path, f"snapshot/{snapshot_base}")
            # include snapshot head
            head = take_file_head(snapshot_path, max_lines=max_head_lines)
            write_zip_text(z, "snapshot_head.csv", head)
            write_zip_text(z, "snapshot_stats.json", json.dumps(stats, ensure_ascii=False, indent=2))

        # logs
        add_file_if_exists(z, launcher_log_path, "logs/launcher.log")
        add_file_if_exists(z, outcome.proc_stdout, "logs/parser_stdout.log")
        add_file_if_exists(z, outcome.proc_stderr, "logs/parser_stderr.log")
        add_file_if_exists(z, outcome.proc_meta, "logs/parser_meta.json")
        if parser_log_path:
            add_file_if_exists(z, parser_log_path, "logs/parser_internal.log")

        # errors
        if errors_csv_path and os.path.exists(errors_csv_path):
            filtered = filter_errors_by_run_id(errors_csv_path, run_id)
            if filtered:
                write_zip_text(z, "errors_filtered.csv", filtered[: 5_000_000])  # safety cap
            # head of full errors
            err_head = take_file_head(errors_csv_path, max_lines=max_errors_lines)
            write_zip_text(z, "errors_head.csv", err_head[: 5_000_000])

        # raw dumps / recent files
        output_root = guess_parser_output_root(repo_root, outcome.key)
        if output_root:
            added = add_dir_recent(z, os.path.join(output_root, "errors"), "artifacts/errors_recent", started_epoch=started_epoch, max_files=max_recent_files)
            _ = added
            added2 = add_dir_recent(z, os.path.join(output_root, "quarantine"), "artifacts/quarantine_recent", started_epoch=started_epoch, max_files=max_recent_files)
            _ = added2

    return zip_path


# ----------------------------- main runner -----------------------------


def run_one_parser(
    *,
    repo_root: str,
    key: str,
    pcfg: dict,
    run_id: str,
    launcher_run_logs_dir: str,
    baseline_state: Dict[str, str],
    cfg_quality: dict,
    city: str,
    captured_at_iso: str,
    log: Log,
    reports_dir: str,
    launcher_log_path: str,
    cfg_console: dict,
    cfg_bundle: dict,
) -> ParserOutcome:
    developer = pcfg.get("developer", key)
    runner = pcfg.get("runner") or {}
    console_cfg = dict(cfg_console or {})
    show_window_default = bool(console_cfg.get("show_window", console_cfg.get("enabled", False)))
    show_window = bool(runner.get("show_window", show_window_default))

    timeout_sec = int(pcfg.get("timeout_sec") or pcfg.get("timeout") or 1800)

    cmd = build_cmd(repo_root, runner)
    parsers_dir = os.path.join(launcher_run_logs_dir, "parsers")
    ensure_dir(parsers_dir)

    stdout_path = os.path.join(parsers_dir, f"{key}.stdout.log")
    stderr_path = os.path.join(parsers_dir, f"{key}.stderr.log")
    meta_path = os.path.join(parsers_dir, f"{key}.meta.json")

    env = dict(os.environ)
    env["RUN_ID"] = run_id
    env["LAUNCHER_RUN_ID"] = run_id
    env["PARSER_KEY"] = key

    log.line(f"RUN {key} | cmd={' '.join(cmd)} | timeout={timeout_sec}s")

    exit_code, started_epoch, finished_epoch = run_process(
        cmd,
        cwd=repo_root,
        timeout_sec=timeout_sec,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        meta_path=meta_path,
        env=env,
        show_window=show_window,
    )

    # Try result.json first
    result_json_path = detect_result_json(repo_root, key, pcfg)
    reason_json_path = detect_reason_json(result_json_path) if result_json_path else None

    status = "FAIL"
    suspect_flags: List[str] = []
    message = ""
    metrics: Dict[str, Any] = {}
    snapshot_path: Optional[str] = None
    reason_code: Optional[str] = None

    if result_json_path:
        rj = read_json(result_json_path)
        status = (rj.get("status") or "FAIL").upper()
        message = str(rj.get("message_for_human") or "")
        suspect_flags = list(rj.get("suspect_flags") or [])
        metrics = dict(rj.get("metrics") or {})
        reason_code = (rj.get("reason_code") or rj.get("reason") or None)
        snapshot_rel = rj.get("output_snapshot_path")
        snapshot_path = resolve_rel(repo_root, snapshot_rel) if snapshot_rel else None

    # Fallback legacy detection
    if not snapshot_path or not os.path.exists(snapshot_path):
        snapshot_path = detect_snapshot(repo_root, pcfg, min_mtime=started_epoch)

    # Legacy status classification if no result.json
    if not result_json_path:
        status = "OK" if (exit_code == 0 and snapshot_path and os.path.exists(snapshot_path)) else "FAIL"
        message = "legacy parser (no result.json)"
        # Lightweight check (optional)
        price_col = (pcfg.get("output") or {}).get("price_col")
        if status == "OK" and price_col:
            try:
                status_cols = ((pcfg.get("output") or {}).get("status_cols") or None)
                ignore_statuses = ((pcfg.get("output") or {}).get("price_null_ignore_statuses") or None)
                cand = compute_csv_basic_stats(snapshot_path, price_col=price_col, status_cols=status_cols, price_null_ignore_statuses=ignore_statuses)
                metrics.update(cand)
                base_rel = baseline_state.get(key)
                base_abs = resolve_rel(repo_root, base_rel)
                if base_abs and os.path.exists(base_abs):
                    b = compute_csv_basic_stats(base_abs, price_col=price_col, status_cols=status_cols, price_null_ignore_statuses=ignore_statuses)
                    metrics["baseline_rows"] = b.get("rows")
                    metrics["row_delta_pct"] = round(pct_delta(float(cand.get("rows") or 0), float(b.get("rows") or 0)), 3)
                    if abs(float(metrics["row_delta_pct"])) >= float(cfg_quality.get("warn_row_delta_pct", 25)):
                        status = "WARN"
                        suspect_flags.append(f"WARN_ROW_DELTA_{metrics['row_delta_pct']:.1f}%")
                # warn on low price fill
                if float(cand.get("price_null_pct") or 0.0) >= float(cfg_quality.get("warn_price_null_pct", 15)):
                    status = "WARN"
                    suspect_flags.append("WARN_PRICE_NULL")
            except Exception:
                pass

    copied_to = None

    # Copy decision: OK -> published; WARN/FAIL -> quarantine
    target = "published" if status == "OK" else "quarantine"

    if snapshot_path and os.path.exists(snapshot_path):
        base_name = os.path.basename(snapshot_path)
        if not base_name.startswith(f"{key}__"):
            base_name = f"{key}__{city}__{run_id}.csv"

        dest_dir = os.path.join(repo_root, "launcher", "data", "published_snapshots" if target == "published" else "quarantine", run_id)
        ensure_dir(dest_dir)
        dest_path = os.path.join(dest_dir, base_name)

        src = snapshot_path
        source = str(pcfg.get("source") or pcfg.get("source_url") or "")
        if result_json_path:
            try:
                rj = read_json(result_json_path)
                source = str(rj.get("source") or source)
            except Exception:
                pass
        normalize_and_copy_csv(src, dest_path, developer_key=key, source=source or "(unknown)", captured_at=captured_at_iso)
        copied_to = dest_path

        if target == "published":
            baseline_state[key] = os.path.relpath(dest_path, repo_root).replace("\\", "/")
    else:
        status = "FAIL"
        message = message or "no snapshot file produced"

    # detect parser internal artifacts
    output_root = guess_parser_output_root(repo_root, key)
    parser_internal_log = detect_parser_log(output_root, run_id=run_id, started_epoch=started_epoch)
    errors_csv_path = detect_errors_csv(output_root)
    # If parser is legacy (no result.json) but we have suspect flags, use the first flag as reason_code for reporting
    if not reason_code and suspect_flags:
        reason_code = str(suspect_flags[0])

    support_bundle_path = build_support_bundle(
        repo_root=repo_root,
        reports_dir=reports_dir,
        run_id=run_id,
        outcome=ParserOutcome(
            key=key,
            developer=developer,
            exit_code=exit_code,
            status=status,
            output_snapshot=snapshot_path,
            result_json=result_json_path,
            reason_json=reason_json_path,
            reason_code=reason_code,
            message_for_human=message,
            suspect_flags=suspect_flags,
            metrics=metrics,
            copied_to=copied_to,
            proc_stdout=stdout_path,
            proc_stderr=stderr_path,
            proc_meta=meta_path,
            support_bundle=None,
        ),
        launcher_log_path=launcher_log_path,
        parser_log_path=parser_internal_log,
        errors_csv_path=errors_csv_path,
        started_epoch=started_epoch,
        cfg_bundle=cfg_bundle,
    )

    if status != "OK" and support_bundle_path:
        log.line(f"SUPPORT_BUNDLE {key}: {relposix(repo_root, support_bundle_path)}")

    return ParserOutcome(
        key=key,
        developer=developer,
        exit_code=exit_code,
        status=status,
        output_snapshot=snapshot_path,
        result_json=result_json_path,
        reason_json=reason_json_path,
        reason_code=reason_code,
        message_for_human=message,
        suspect_flags=suspect_flags,
        metrics=metrics,
        copied_to=copied_to,
        proc_stdout=stdout_path,
        proc_stderr=stderr_path,
        proc_meta=meta_path,
        support_bundle=support_bundle_path,
    )


# ----------------------------- CLI -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Launcher: run parsers and publish/quarantine snapshots")
    ap.add_argument("--config", default=os.path.join("launcher", "launcher_config.json"))
    ap.add_argument("--mode", default="all", choices=["all", "only_failed", "selected"])
    ap.add_argument("--only", default="", help="Comma-separated parser keys for mode=selected")
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(repo_root, args.config) if not os.path.isabs(args.config) else args.config
    cfg = read_json(cfg_path)

    run_id = now_run_id()
    city = str(cfg.get("city") or "city")
    captured_at_iso = run_id_to_iso(run_id)

    # Launcher log folders
    launcher_logs_root = os.path.join(repo_root, "launcher", "logs")
    ensure_dir(launcher_logs_root)
    launcher_run_logs_dir = os.path.join(launcher_logs_root, run_id)
    ensure_dir(launcher_run_logs_dir)

    launcher_log_path = os.path.join(launcher_run_logs_dir, "launcher.log")
    log = Log(launcher_log_path)

    reports_dir = os.path.join(repo_root, "launcher", "data", "reports", run_id)
    ensure_dir(reports_dir)

    # Always create publish/quarantine dirs for this RUN_ID (even if no snapshots are copied)
    published_run_dir = os.path.join(repo_root, "launcher", "data", "published_snapshots", run_id)
    quarantine_run_dir = os.path.join(repo_root, "launcher", "data", "quarantine", run_id)
    ensure_dir(published_run_dir)
    ensure_dir(quarantine_run_dir)


    log.line(f"START launcher run_id={run_id} mode={args.mode} city={city}")
    log.line(f"Config: {relposix(repo_root, cfg_path)}")
    log.line(f"Reports: {relposix(repo_root, reports_dir)}")
    log.line(f"Logs: {relposix(repo_root, launcher_run_logs_dir)}")

    parsers_cfg = cfg.get("parsers") or {}

    enabled_keys = [k for k, v in parsers_cfg.items() if v.get("enabled", True)]
    selected = enabled_keys

    if args.mode == "selected":
        only = [x.strip() for x in args.only.split(",") if x.strip()]
        selected = [k for k in enabled_keys if k in set(only)]

    if args.mode == "only_failed":
        last = read_json(os.path.join(repo_root, "launcher", "data", "last_run.json"), default={})
        last_id = last.get("run_id")
        if not last_id:
            log.line("No last_run.json. Nothing to retry.")
            return 0
        last_sum = read_json(os.path.join(repo_root, "launcher", "data", "reports", str(last_id), "summary.json"), default={})
        failed = []
        for k, info in (last_sum.get("parsers") or {}).items():
            if (info.get("status") or "").upper() != "OK":
                failed.append(k)
        selected = [k for k in enabled_keys if k in set(failed)]

    if not selected:
        log.line("No parsers selected.")
        return 0

    # baseline state
    state_path = os.path.join(repo_root, "launcher", "data", "published_state.json")
    baseline_state = read_json(state_path, default={})
    if not isinstance(baseline_state, dict):
        baseline_state = {}

    cfg_quality = cfg.get("quality") or {}
    cfg_bundle = (cfg.get("support_bundle") or {})
    cfg_console = (cfg.get("console") or {})

    max_parallel = int(cfg.get("max_parallel", 1))
    if max_parallel < 1:
        max_parallel = 1

    outcomes: List[ParserOutcome] = []
    queue = list(selected)

    def run_key(k: str) -> ParserOutcome:
        return run_one_parser(
            repo_root=repo_root,
            key=k,
            pcfg=parsers_cfg[k],
            run_id=run_id,
            launcher_run_logs_dir=launcher_run_logs_dir,
            baseline_state=baseline_state,
            cfg_quality=cfg_quality,
            city=city,
            captured_at_iso=captured_at_iso,
            log=log,
            reports_dir=reports_dir,
            launcher_log_path=launcher_log_path,
            cfg_console=cfg_console,
            cfg_bundle=cfg_bundle,
        )

    if max_parallel == 1:
        for k in queue:
            outcomes.append(run_key(k))
    else:
        import concurrent.futures
        in_flight: Dict[concurrent.futures.Future, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
            while queue or in_flight:
                while queue and len(in_flight) < max_parallel:
                    k = queue.pop(0)
                    if not parsers_cfg[k].get("parallel", True):
                        log.line(f"RUN (sync) {k}")
                        outcomes.append(run_key(k))
                        continue
                    log.line(f"RUN (async) {k}")
                    fut = ex.submit(run_key, k)
                    in_flight[fut] = k

                if not in_flight:
                    continue

                done, _ = concurrent.futures.wait(in_flight.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    k = in_flight.pop(fut)
                    try:
                        outcomes.append(fut.result())
                    except Exception as e:
                        outcomes.append(
                            ParserOutcome(
                                key=k,
                                developer=parsers_cfg[k].get("developer", k),
                                exit_code=1,
                                status="FAIL",
                                output_snapshot=None,
                                result_json=None,
                                reason_json=None,
                                reason_code="LAUNCHER_EXCEPTION",
                                message_for_human=f"launcher exception: {e}",
                                suspect_flags=["LAUNCHER_EXCEPTION"],
                                metrics={},
                                copied_to=None,
                                proc_stdout=None,
                                proc_stderr=None,
                                proc_meta=None,
                                support_bundle=None,
                            )
                        )

    write_json(state_path, baseline_state)

    summary = {
        "run_id": run_id,
        "created_at": dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "parsers": {},
        "published": 0,
        "quarantine": 0,
        "launcher_log": relposix(repo_root, launcher_log_path),
        "launcher_logs_dir": relposix(repo_root, launcher_run_logs_dir),
    }

    for o in outcomes:
        summary["parsers"][o.key] = {
            "developer": o.developer,
            "exit_code": o.exit_code,
            "status": o.status,
            "reason_code": o.reason_code,
            "copied_to": relposix(repo_root, o.copied_to),
            "result_json": relposix(repo_root, o.result_json),
            "reason_json": relposix(repo_root, o.reason_json),
            "suspect_flags": o.suspect_flags,
            "metrics": o.metrics,
            "message_for_human": o.message_for_human,
            "proc_stdout": relposix(repo_root, o.proc_stdout),
            "proc_stderr": relposix(repo_root, o.proc_stderr),
            "proc_meta": relposix(repo_root, o.proc_meta),
            "support_bundle": relposix(repo_root, o.support_bundle),
        }
        if o.status == "OK":
            summary["published"] += 1
        else:
            summary["quarantine"] += 1

    write_json(os.path.join(reports_dir, "summary.json"), summary)

    lines: List[str] = []
    lines.append(f"RUN_ID: {run_id}")
    lines.append(f"Published: {summary['published']}")
    lines.append(f"Quarantine: {summary['quarantine']}")
    lines.append(f"Launcher log: {summary['launcher_log']}")
    lines.append("")
    for o in outcomes:
        lines.append(f"- {o.developer} ({o.key}): {o.status} | copied_to={o.copied_to or '-'}")
        if o.reason_code:
            lines.append(f"    reason_code: {o.reason_code}")
        if o.suspect_flags:
            lines.append(f"    flags: {', '.join(o.suspect_flags)}")
        if o.message_for_human:
            lines.append(f"    note: {o.message_for_human}")
        if o.support_bundle:
            lines.append(f"    support_bundle: {relposix(repo_root, o.support_bundle)}")

    with open(os.path.join(reports_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    write_json(os.path.join(repo_root, "launcher", "data", "last_run.json"), {"run_id": run_id})

    log.line(f"DONE run_id={run_id} published={summary['published']} quarantine={summary['quarantine']}")
    log.line(f"Report: {relposix(repo_root, reports_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())