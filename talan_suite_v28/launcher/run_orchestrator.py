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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


# ----------------------------- helpers -----------------------------


def now_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().replace(microsecond=0).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_path(repo_root: str, p: Optional[str]) -> Optional[str]:
    """Resolve a possibly-relative path stored in state/config into an absolute path."""
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(repo_root, p))


def to_relpath(repo_root: str, p_abs: str) -> str:
    """Store paths in state as repo-root-relative to keep them portable."""
    return os.path.relpath(os.path.normpath(p_abs), repo_root).replace("\\", "/")


class OrchestratorLogger:
    """Very small file+stdout logger.

    We avoid Python logging config here because parsers already manage their own logging.
    """

    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path))
        self.path = path
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("")

    def log(self, msg: str) -> None:
        ts = dt.datetime.now(dt.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} | {msg}"
        print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def relpath(p: str) -> str:
    try:
        return os.path.relpath(p)
    except Exception:
        return p


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


def read_csv_rows_count(path: str) -> int:
    # fast line count (excluding header)
    with open(path, "r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(" ", "")
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None


def pct(a: float) -> str:
    return f"{a:.1f}%"


def ask_choice(prompt: str, choices: Dict[str, str], default_key: Optional[str]) -> str:
    # choices: key -> label
    keys = "/".join(choices.keys())
    if default_key:
        full = f"{prompt} [{keys}] (Enter={default_key}): "
    else:
        full = f"{prompt} [{keys}]: "
    while True:
        ans = input(full).strip().lower()
        if ans == "" and default_key:
            return default_key
        if ans in choices:
            return ans
        print(f"Введите один из вариантов: {', '.join(choices.keys())}")


def tee_process(cmd: List[str], *, prefix: str, log_path: str, cwd: str) -> int:
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            lf.write(line)
            lf.flush()
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()
        return p.wait()


def append_line(path: str, line: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


# ----------------------------- quality report -----------------------------


@dataclass
class QualityResult:
    hard_fail: bool
    warnings: List[str]
    metrics: Dict[str, float]


def compute_quality(
    candidate_csv: str,
    baseline_csv: Optional[str],
    *,
    id_col: str,
    price_col: str,
    cfg_quality: dict,
) -> QualityResult:
    warnings: List[str] = []
    metrics: Dict[str, float] = {}

    # minimal fallback without pandas
    if pd is None:
        # only row count + duplicates + price fill (best-effort)
        rows = []
        with open(candidate_csv, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        n = len(rows)
        metrics["rows"] = float(n)
        ids = [row.get(id_col) for row in rows]
        ids_nonnull = [i for i in ids if i not in (None, "")]
        dup = len(ids_nonnull) - len(set(ids_nonnull))
        dup_pct = (dup / n * 100.0) if n else 0.0
        metrics["dup_id_pct"] = dup_pct
        price_filled = sum(1 for row in rows if safe_float(row.get(price_col)) is not None)
        price_fill_pct = (price_filled / n * 100.0) if n else 0.0
        metrics["price_fill_pct"] = price_fill_pct
        baseline_n = None
        baseline_price_fill_pct = None
        if baseline_csv and os.path.exists(baseline_csv):
            with open(baseline_csv, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                b_rows = list(r)
            baseline_n = len(b_rows)
            baseline_price_filled = sum(1 for row in b_rows if safe_float(row.get(price_col)) is not None)
            baseline_price_fill_pct = (baseline_price_filled / baseline_n * 100.0) if baseline_n else 0.0

        row_delta_pct = 0.0
        if baseline_n:
            row_delta_pct = ((n - baseline_n) / baseline_n) * 100.0
        metrics["row_delta_pct"] = row_delta_pct
        price_fill_drop_pct = 0.0
        if baseline_price_fill_pct is not None:
            price_fill_drop_pct = baseline_price_fill_pct - price_fill_pct
        metrics["price_fill_drop_pct"] = price_fill_drop_pct
    else:
        df = pd.read_csv(candidate_csv)
        n = len(df)
        metrics["rows"] = float(n)

        if id_col not in df.columns:
            return QualityResult(True, [f"HARD: нет колонки id_col='{id_col}'"], metrics)
        if price_col not in df.columns:
            return QualityResult(True, [f"HARD: нет колонки price_col='{price_col}'"], metrics)

        dup = df[id_col].duplicated(keep=False).sum()
        dup_pct = (float(dup) / n * 100.0) if n else 0.0
        metrics["dup_id_pct"] = dup_pct

        price_fill_pct = (df[price_col].notna().mean() * 100.0) if n else 0.0
        metrics["price_fill_pct"] = float(price_fill_pct)

        row_delta_pct = 0.0
        price_fill_drop_pct = 0.0
        if baseline_csv and os.path.exists(baseline_csv):
            b = pd.read_csv(baseline_csv)
            bn = len(b)
            metrics["baseline_rows"] = float(bn)
            if bn:
                row_delta_pct = ((n - bn) / bn) * 100.0
            if price_col in b.columns and bn:
                b_price_fill = b[price_col].notna().mean() * 100.0
                price_fill_drop_pct = float(b_price_fill) - float(price_fill_pct)
        metrics["row_delta_pct"] = float(row_delta_pct)
        metrics["price_fill_drop_pct"] = float(price_fill_drop_pct)

    # classify
    warn_row = float(cfg_quality.get("warn_row_delta_pct", 25))
    hard_row = float(cfg_quality.get("hard_row_delta_pct", 40))
    warn_drop = float(cfg_quality.get("warn_price_fill_drop_pct", 8))
    hard_drop = float(cfg_quality.get("hard_price_fill_drop_pct", 15))
    warn_dup = float(cfg_quality.get("warn_dup_id_pct", 0.2))
    hard_dup = float(cfg_quality.get("hard_dup_id_pct", 1.0))

    # row delta is only meaningful if baseline exists
    if baseline_csv and os.path.exists(baseline_csv):
        if abs(metrics.get("row_delta_pct", 0.0)) >= warn_row:
            warnings.append(f"WARN: изменение количества строк vs prev: {pct(metrics['row_delta_pct'])}")

    if metrics.get("price_fill_drop_pct", 0.0) >= warn_drop:
        warnings.append(f"WARN: просадка заполненности цены vs prev: {pct(metrics['price_fill_drop_pct'])}")

    if metrics.get("dup_id_pct", 0.0) >= warn_dup:
        warnings.append(f"WARN: дубликаты по id: {metrics['dup_id_pct']:.3f}%")

    # HARD only by multi-signal (чтобы не ломать из-за одной метрики)
    hard_fail = False
    hard_reasons: List[str] = []
    if metrics.get("dup_id_pct", 0.0) >= hard_dup:
        hard_fail = True
        hard_reasons.append(f"HARD: много дублей id ({metrics['dup_id_pct']:.3f}%)")

    if baseline_csv and os.path.exists(baseline_csv):
        if abs(metrics.get("row_delta_pct", 0.0)) >= hard_row and metrics.get("price_fill_drop_pct", 0.0) >= warn_drop:
            hard_fail = True
            hard_reasons.append(
                f"HARD: сильное изменение строк ({pct(metrics['row_delta_pct'])}) + просадка цен ({pct(metrics['price_fill_drop_pct'])})"
            )

    if metrics.get("price_fill_drop_pct", 0.0) >= hard_drop:
        hard_fail = True
        hard_reasons.append(f"HARD: сильная просадка заполненности цены ({pct(metrics['price_fill_drop_pct'])})")

    warnings = hard_reasons + warnings
    return QualityResult(hard_fail, warnings, metrics)


def write_report_md(path: str, *, dev_key: str, developer: str, candidate: str, baseline: Optional[str], errors_csv: Optional[str], q: QualityResult) -> None:
    ensure_dir(os.path.dirname(path))
    lines: List[str] = []
    lines.append(f"# Report: {developer} ({dev_key})")
    lines.append("")
    lines.append(f"Candidate: `{relpath(candidate)}`")
    lines.append(f"Baseline : `{relpath(baseline)}`" if baseline else "Baseline : (none)")
    lines.append(f"Errors   : `{relpath(errors_csv)}`" if errors_csv else "Errors   : (none)")
    lines.append("")
    lines.append("## Metrics")
    for k, v in q.metrics.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Flags")
    if not q.warnings:
        lines.append("- (none)")
    else:
        for w in q.warnings:
            lines.append(f"- {w}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------- parser specs -----------------------------


@dataclass
class ParserRun:
    key: str
    developer: str
    exit_code: int
    candidate_snapshot: Optional[str]
    candidate_errors: Optional[str]
    parser_internal_log: Optional[str]
    run_log: str
    quality: Optional[QualityResult]
    decision: str
    published_snapshot: Optional[str]
    quarantine_snapshot: Optional[str]


def build_cmd(repo_root: str, runner: dict) -> List[str]:
    # We always run with the current python interpreter (already inside .venv)
    if runner.get("type") != "python":
        raise ValueError("Only runner.type='python' is supported")
    script = runner["script"]
    args = runner.get("args") or []
    return [sys.executable, os.path.join(repo_root, script), *args]


def detect_candidate_files(repo_root: str, out_cfg: dict, *, min_mtime: float) -> Tuple[Optional[str], Optional[str]]:
    snap_glob = os.path.join(repo_root, out_cfg["snapshot_glob"])
    err_glob = os.path.join(repo_root, out_cfg.get("errors_glob", "")) if out_cfg.get("errors_glob") else ""
    snapshot = newest_by_mtime(snap_glob, min_mtime=min_mtime)
    errors = newest_by_mtime(err_glob, min_mtime=min_mtime) if err_glob else None
    return snapshot, errors


def detect_candidate_log(repo_root: str, out_cfg: dict, *, min_mtime: float) -> Optional[str]:
    """Detect the parser's own log file (if it writes one), not the orchestrator tee log."""
    log_glob_rel = out_cfg.get("log_glob")
    if not log_glob_rel:
        return None
    log_glob = os.path.join(repo_root, log_glob_rel)
    return newest_by_mtime(log_glob, min_mtime=min_mtime)


def unikey_retry_merge(
    *,
    base_snapshot: str,
    retry_snapshot: str,
    out_path: str,
    id_col: str,
    house_id_col: str = "house_id",
) -> None:
    if pd is None:
        raise RuntimeError("pandas is required for Unikey merge")
    a = pd.read_csv(base_snapshot)
    b = pd.read_csv(retry_snapshot)
    if house_id_col in b.columns and house_id_col in a.columns:
        houses = sorted(set(b[house_id_col].dropna().astype(int).tolist()))
        a = a[~a[house_id_col].isin(houses)]
    merged = pd.concat([a, b], ignore_index=True)
    if id_col in merged.columns:
        merged = merged.drop_duplicates(subset=[id_col], keep="last")
    ensure_dir(os.path.dirname(out_path))
    merged.to_csv(out_path, index=False)


def parse_unikey_failed_houses(errors_csv: str) -> List[int]:
    if pd is None:
        houses = []
        with open(errors_csv, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                hid = row.get("house_id")
                try:
                    houses.append(int(hid))
                except Exception:
                    pass
        return sorted(set(houses))
    df = pd.read_csv(errors_csv)
    if "house_id" not in df.columns:
        return []
    vals = []
    for x in df["house_id"].dropna().tolist():
        try:
            vals.append(int(x))
        except Exception:
            pass
    return sorted(set(vals))


def run_retry_for_dev(
    *,
    repo_root: str,
    dev_key: str,
    parser_cfg: dict,
    base_candidate_snapshot: str,
    errors_csv: str,
    run_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Returns (new_candidate_snapshot, new_errors_csv)."""

    runner = parser_cfg["runner"]
    out_cfg = parser_cfg["output"]

    if dev_key == "talan":
        # use retry-only
        cmd = [sys.executable, os.path.join(repo_root, runner["script"]), "--retry-only", errors_csv]
        log_path = os.path.join(run_dir, "logs", f"{dev_key}_retry.log")
        start = time.time()
        code = tee_process(cmd, prefix=f"{dev_key}:retry", log_path=log_path, cwd=repo_root)
        # Find patched_latest produced in parser/retry/*.csv after start
        retry_glob = os.path.join(repo_root, os.path.dirname(runner_cfg["script"]), "retry", "*_patched_latest.csv")
        patched = newest_by_mtime(retry_glob, min_mtime=start)
        retry_err_glob = os.path.join(repo_root, os.path.dirname(runner_cfg["script"]), "retry", "*_retry_errors.csv")
        retry_err = newest_by_mtime(retry_err_glob, min_mtime=start)
        if code != 0:
            return None, retry_err
        return patched, retry_err

    if dev_key == "unikey":
        houses = parse_unikey_failed_houses(errors_csv)
        if not houses:
            return None, errors_csv

        cfg_path = parser_cfg.get("output", {}).get("config_path")
        if not cfg_path:
            # fallback to runner args: ["--config", "..."]
            ra = parser_cfg.get("runner", {}).get("args", [])
            cfg_path = ra[1] if len(ra) >= 2 else "parser/unikey_config.json"
        cmd = [
            sys.executable,
            os.path.join(repo_root, runner["script"]),
            "--config",
            os.path.join(repo_root, cfg_path),
            "--houses",
            ",".join(map(str, houses)),
        ]
        log_path = os.path.join(run_dir, "logs", f"{dev_key}_retry.log")
        start = time.time()
        code = tee_process(cmd, prefix=f"{dev_key}:retry", log_path=log_path, cwd=repo_root)
        retry_snapshot, retry_err = detect_candidate_files(repo_root, out_cfg, min_mtime=start)
        if code != 0 and not retry_snapshot:
            return None, retry_err

        merged_path = os.path.join(run_dir, f"{dev_key}_candidate_merged.csv")
        unikey_retry_merge(
            base_snapshot=base_candidate_snapshot,
            retry_snapshot=retry_snapshot,
            out_path=merged_path,
            id_col=out_cfg["id_col"],
        )
        return merged_path, retry_err

    return None, errors_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Run multiple parsers sequentially with quality gate + publish/quarantine")
    ap.add_argument("--config", default=os.path.join("launcher", "launcher_config.json"))
    ap.add_argument("--only", default="", help="Comma-separated parser keys (e.g. talan,unikey)")
    ap.add_argument("--retry-failed", action="store_true", help="Retry only failed targets from last run (data/last_run.json)")
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(repo_root, args.config) if not os.path.isabs(args.config) else args.config
    cfg = read_json(cfg_path)

    run_id = now_ts()
    data_dir = os.path.join(repo_root, cfg.get("data_dir", "data"))
    runs_dir = os.path.join(repo_root, cfg.get("runs_dir", "data/runs"))
    published_dir = os.path.join(repo_root, cfg.get("published_dir", "data/published_snapshots"))
    quarantine_dir = os.path.join(repo_root, cfg.get("quarantine_dir", "data/quarantine"))
    ensure_dir(data_dir)
    ensure_dir(runs_dir)
    ensure_dir(published_dir)
    ensure_dir(quarantine_dir)

    # Published baseline state (for compare with previous published snapshot per parser)
    published_state_path = os.path.join(data_dir, "published_state.json")
    published_state_raw = read_json(published_state_path, default={})
    published_state: Dict[str, str] = published_state_raw if isinstance(published_state_raw, dict) else {}

    last_run_path = os.path.join(data_dir, "last_run.json")

    only = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else []
    parsers_cfg = cfg.get("parsers", {})
    selected_keys = [k for k in parsers_cfg.keys() if parsers_cfg[k].get("enabled", True)]
    if only:
        selected_keys = [k for k in selected_keys if k in set(only)]
    if args.retry_failed:
        if not os.path.exists(last_run_path):
            print("No data/last_run.json found. Run full orchestrator first.")
            return 2
        last = read_json(last_run_path)
        last_dir = os.path.join(runs_dir, last.get("run_id", ""))
        if not os.path.isdir(last_dir):
            print(f"Last run folder not found: {last_dir}")
            return 2
        decisions_path = os.path.join(last_dir, "decisions.json")
        if not os.path.exists(decisions_path):
            print(f"decisions.json not found in last run: {decisions_path}")
            return 2
        decisions = read_json(decisions_path)
        selected_keys = [k for k, v in decisions.get("parsers", {}).items() if v.get("needs_retry")]
        if not selected_keys:
            print("No parsers marked as needs_retry in last run.")
            return 0

    run_dir = os.path.join(runs_dir, run_id)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "logs"))
    ensure_dir(os.path.join(run_dir, "errors"))
    ensure_dir(os.path.join(run_dir, "parser_logs"))

    olog = OrchestratorLogger(os.path.join(run_dir, "orchestrator.log"))
    olog.log(f"Orchestrator start. run_id={run_id}")

    errors_dir = cfg.get("errors_dir") or os.path.join(data_dir, "errors")
    run_errors_dir = os.path.join(repo_root, errors_dir, run_id)
    ensure_dir(run_errors_dir)

    write_json(last_run_path, {"run_id": run_id, "created_at": dt.datetime.now(dt.timezone.utc).isoformat()})

    decisions: dict = {"run_id": run_id, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "parsers": {}}

    city = cfg.get("city", "khabarovsk")
    city_ru = cfg.get("city_ru", "Хабаровск")
    cfg_quality = cfg.get("quality", {})
    prompt_mode = cfg.get("prompt_mode", "ask")

    results: List[ParserRun] = []

    for key in selected_keys:
        pcfg = parsers_cfg[key]
        developer = pcfg.get("developer", key)
        runner = pcfg.get("runner", {})
        out_cfg = pcfg.get("output", {})

        print("\n" + "=" * 72)
        print(f"RUN: {developer} ({key})")
        print("=" * 72)

        start = time.time()
        cmd = build_cmd(repo_root, runner)
        log_path = os.path.join(run_dir, "logs", f"{key}.log")
        olog.log(f"START {key} | cmd={' '.join(cmd)}")
        exit_code = tee_process(cmd, prefix=key, log_path=log_path, cwd=repo_root)

        olog.log(f"FINISH {key} | exit_code={exit_code}")

        cand_snap, cand_err = detect_candidate_files(repo_root, out_cfg, min_mtime=start)
        cand_plog = detect_candidate_log(repo_root, out_cfg, min_mtime=start)

        olog.log(
            f"ARTIFACTS {key} | snapshot={relpath(cand_snap) if cand_snap else None} | "
            f"errors={relpath(cand_err) if cand_err else None} | parser_log={relpath(cand_plog) if cand_plog else None}"
        )

        # Copy candidate artifacts into run_dir for stability
        dev_run_dir = os.path.join(run_dir, key)
        ensure_dir(dev_run_dir)
        copied_snap = None
        copied_err = None
        copied_plog = None
        if cand_snap and os.path.exists(cand_snap):
            copied_snap = os.path.join(dev_run_dir, os.path.basename(cand_snap))
            shutil.copy2(cand_snap, copied_snap)
        if cand_err and os.path.exists(cand_err):
            ensure_dir(os.path.join(dev_run_dir, "errors"))
            copied_err = os.path.join(dev_run_dir, "errors", os.path.basename(cand_err))
            shutil.copy2(cand_err, copied_err)

            # Also copy errors to a central folder for quick review
            try:
                central_name = f"{run_id}__{key}__{os.path.basename(cand_err)}"
                shutil.copy2(cand_err, os.path.join(run_dir, "errors", central_name))
                shutil.copy2(cand_err, os.path.join(run_errors_dir, central_name))
            except Exception:
                pass

        # Copy parser's own log, if any
        if cand_plog and os.path.exists(cand_plog):
            ensure_dir(os.path.join(dev_run_dir, "parser_logs"))
            copied_plog = os.path.join(dev_run_dir, "parser_logs", os.path.basename(cand_plog))
            shutil.copy2(cand_plog, copied_plog)

            # Central copy (run folder)
            try:
                ensure_dir(os.path.join(run_dir, "parser_logs", key))
                shutil.copy2(cand_plog, os.path.join(run_dir, "parser_logs", key, os.path.basename(cand_plog)))
            except Exception:
                pass

        baseline = resolve_path(repo_root, published_state.get(key))
        if baseline and not os.path.exists(baseline):
            baseline = None

        q: Optional[QualityResult] = None
        if copied_snap:
            q = compute_quality(
                copied_snap,
                baseline,
                id_col=out_cfg["id_col"],
                price_col=out_cfg["price_col"],
                cfg_quality=cfg_quality,
            )

            report_path = os.path.join(run_dir, f"report_{key}.md")
            write_report_md(
                report_path,
                dev_key=key,
                developer=developer,
                candidate=copied_snap,
                baseline=baseline,
                errors_csv=copied_err,
                q=q,
            )

            print(f"Report: {relpath(report_path)}")

        # Determine needs_retry
        needs_retry = False
        err_rows = 0
        if copied_err and os.path.exists(copied_err):
            try:
                err_rows = read_csv_rows_count(copied_err)
            except Exception:
                err_rows = 0
        if err_rows > 0 or exit_code != 0:
            needs_retry = True

        # Decide action
        default = "p"
        if q and q.hard_fail:
            default = "q"
        if needs_retry:
            default = "r" if key in {"talan", "unikey"} else "q"

        decision = "s"
        published_path = None
        quarantine_path = None
        if prompt_mode == "ask" and sys.stdin.isatty():
            choices = {
                "p": "publish",
                "q": "quarantine",
                "r": "retry_failed",
                "s": "skip",
            }
            decision = ask_choice(
                f"Action for {developer} (rows={q.metrics.get('rows') if q else 'n/a'}, errors={err_rows}, hard={q.hard_fail if q else 'n/a'})",
                choices,
                default,
            )
        else:
            # non-interactive: auto publish only if no errors and not hard fail
            if (not needs_retry) and (q is not None) and (not q.hard_fail):
                decision = "p"
            else:
                decision = "q"

        # Retry flow (optional)
        if decision == "r":
            if not copied_err:
                print("No errors file to retry from. Switching to quarantine.")
                decision = "q"
            else:
                print(f"Retrying failed targets for {developer}...")
                new_snap, new_err = run_retry_for_dev(
                    repo_root=repo_root,
                    dev_key=key,
                    parser_cfg=pcfg,
                    base_candidate_snapshot=copied_snap,
                    errors_csv=copied_err,
                    run_dir=dev_run_dir,
                )
                if new_snap and os.path.exists(new_snap):
                    copied_snap = new_snap
                if new_err and os.path.exists(new_err):
                    copied_err = new_err
                # recompute quality after retry
                if copied_snap:
                    q = compute_quality(
                        copied_snap,
                        baseline,
                        id_col=out_cfg["id_col"],
                        price_col=out_cfg["price_col"],
                        cfg_quality=cfg_quality,
                    )
                # re-evaluate retry status
                err_rows = 0
                if copied_err and os.path.exists(copied_err):
                    try:
                        err_rows = read_csv_rows_count(copied_err)
                    except Exception:
                        err_rows = 0
                needs_retry = err_rows > 0

                # ask again publish/quarantine
                default2 = "p" if (not needs_retry) and (q is not None) and (not q.hard_fail) and (not q.warnings) else "q"
                if prompt_mode == "ask" and sys.stdin.isatty():
                    decision = ask_choice(
                        f"After retry: publish/quarantine? (rows={q.metrics.get('rows') if q else 'n/a'}, errors={err_rows}, hard={q.hard_fail if q else 'n/a'})",
                        {"p": "publish", "q": "quarantine"},
                        default2,
                    )
                else:
                    decision = "p" if default2 == "p" else "q"

        if decision == "p":
            if not copied_snap:
                print("No candidate snapshot produced. Nothing to publish.")
                decision = "q"
            else:
                pub_run_dir = os.path.join(published_dir, run_id)
                ensure_dir(pub_run_dir)
                pub_name = f"{key}__{run_id}.csv"
                pub_path = os.path.join(pub_run_dir, pub_name)
                shutil.copy2(copied_snap, pub_path)
                published_path = pub_path
                # update baseline pointer
                published_state[key] = to_relpath(repo_root, pub_path)
                write_json(published_state_path, published_state)
                print(f"Published: {relpath(pub_path)}")

        if decision == "q":
            if copied_snap:
                q_run_dir = os.path.join(quarantine_dir, run_id)
                ensure_dir(q_run_dir)
                q_path = os.path.join(q_run_dir, f"{key}__{run_id}.csv")
                shutil.copy2(copied_snap, q_path)
                quarantine_path = q_path
                print(f"Quarantine: {relpath(q_path)}")

        # record
        olog.log(f"DECISION {key} | decision={decision} | needs_retry={bool(needs_retry)}")
        pr = ParserRun(
            key=key,
            developer=developer,
            exit_code=exit_code,
            candidate_snapshot=copied_snap,
            candidate_errors=copied_err,
            parser_internal_log=copied_plog,
            run_log=log_path,
            quality=q,
            decision=decision,
            published_snapshot=published_path,
            quarantine_snapshot=quarantine_path,
        )
        results.append(pr)

        decisions["parsers"][key] = {
            "developer": developer,
            "exit_code": exit_code,
            "candidate_snapshot": relpath(copied_snap) if copied_snap else None,
            "candidate_errors": relpath(copied_err) if copied_err else None,
            "parser_internal_log": relpath(copied_plog) if copied_plog else None,
            "report": relpath(os.path.join(run_dir, f"report_{key}.md")) if copied_snap else None,
            "decision": decision,
            "published_snapshot": relpath(published_path) if published_path else None,
            "quarantine_snapshot": relpath(quarantine_path) if quarantine_path else None,
            "needs_retry": bool(needs_retry),
            "errors_rows": err_rows,
            "hard_fail": bool(q.hard_fail) if q else None,
            "warnings": q.warnings if q else [],
        }

    # ---- summary (human-friendly) ----
    # Keep it extremely simple: a compact per-parser table + what to do next.
    published_cnt = sum(1 for r in results if r.decision == "p")
    quarantine_cnt = sum(1 for r in results if r.decision == "q")
    retry_cnt = sum(1 for r in results if decisions["parsers"].get(r.key, {}).get("needs_retry"))

    def _rows(r: ParserRun) -> Optional[int]:
        try:
            if r.quality and "rows" in r.quality.metrics:
                return int(r.quality.metrics["rows"])
        except Exception:
            return None
        return None

    def _price_fill(r: ParserRun) -> Optional[float]:
        try:
            if r.quality and "price_fill_pct" in r.quality.metrics:
                return float(r.quality.metrics["price_fill_pct"])
        except Exception:
            return None
        return None

    summary_lines: List[str] = []
    summary_lines.append(f"# Run summary: {run_id}")
    summary_lines.append("")
    summary_lines.append(f"Run folder: `{relpath(run_dir)}`")
    summary_lines.append(f"Main log : `{relpath(os.path.join(run_dir, 'orchestrator.log'))}`")
    summary_lines.append(f"Errors   : `{relpath(run_errors_dir)}`")
    summary_lines.append("")
    summary_lines.append("## Result")
    summary_lines.append(f"- Published: **{published_cnt}**")
    summary_lines.append(f"- Quarantine: **{quarantine_cnt}**")
    summary_lines.append(f"- Needs retry: **{retry_cnt}**")
    summary_lines.append("")
    summary_lines.append("## Parsers")
    summary_lines.append("| Parser | Exit | Rows | Errors | Price fill | Decision | Snapshot | Published/Quarantine |")
    summary_lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for r in results:
        rows = _rows(r)
        price_fill = _price_fill(r)
        d = decisions["parsers"].get(r.key, {})
        err_rows = d.get("errors_rows", 0)
        snap = d.get("candidate_snapshot") or ""
        pub = d.get("published_snapshot") or d.get("quarantine_snapshot") or ""
        summary_lines.append(
            f"| {r.developer} ({r.key}) | {r.exit_code} | {rows if rows is not None else ''} | {err_rows} | "
            f"{(f'{price_fill:.1f}%' if price_fill is not None else '')} | {d.get('decision','')} | `{snap}` | `{pub}` |"
        )

    # show warnings (only if exist)
    any_warn = False
    for r in results:
        d = decisions["parsers"].get(r.key, {})
        ws = d.get("warnings") or []
        if ws:
            any_warn = True
            break
    if any_warn:
        summary_lines.append("")
        summary_lines.append("## Warnings")
        for r in results:
            d = decisions["parsers"].get(r.key, {})
            ws = d.get("warnings") or []
            if not ws:
                continue
            summary_lines.append(f"### {r.developer} ({r.key})")
            for w in ws:
                summary_lines.append(f"- {w}")

    summary_lines.append("")
    summary_lines.append("## Next steps")
    summary_lines.append("- To retry failed parsers from this run: run `run_failed_parsers.bat` (or `python orchestrator/run_orchestrator.py --retry-failed`).")
    summary_lines.append("- Review raw per-parser logs in `data/runs/<RUN_ID>/logs/`.")
    summary_lines.append("- Review errors CSVs in `data/errors/<RUN_ID>/`.")

    summary_md = os.path.join(run_dir, "summary.md")
    ensure_dir(os.path.dirname(summary_md))
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    olog.log(f"SUMMARY written: {relpath(summary_md)}")

    # also write a tiny machine-readable summary
    write_json(os.path.join(run_dir, "summary.json"), {
        "run_id": run_id,
        "run_dir": relpath(run_dir),
        "published": published_cnt,
        "quarantine": quarantine_cnt,
        "needs_retry": retry_cnt,
        "parsers": decisions.get("parsers", {}),
    })

    decisions_path = os.path.join(run_dir, "decisions.json")
    write_json(decisions_path, decisions)

    print("\n" + "=" * 72)
    print("DONE")
    print("Run folder:", relpath(run_dir))
    print("Decisions :", relpath(decisions_path))
    print("Summary   :", relpath(summary_md))
    print("=" * 72)
    olog.log("Orchestrator DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
