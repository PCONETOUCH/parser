#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boilerplate parser for a new developer (dev-friendly reasons + full logging).

Шаблон делает:
- Автономный запуск (python ... --config ...)
- Auto-retry (до max_retries, мягче на каждой попытке)
- Устойчивость к мелким изменениям (get/try/except на уровне записи)
- Snapshot CSV:
    <developer_key>__<city>__<RUN_ID>.csv
  + обязательные колонки: developer_key, captured_at, source
- Артефакты в output_<developer_key>/:
  - logs/ (полный лог)
  - errors/errors.csv (единый append-only лог ошибок/событий)
  - result.json (всегда)
  - reason.json (только для WARN/FAIL; человеческая причина для девелопмента)

ВАЖНО: это шаблон. Реальную логику сети/браузера вставь в collect_lots().
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

SOFT_GREEN = "#7ED957"  # мягкий зелёный для progress-bar


# ----------------------------- helpers: fs/json -----------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def now_iso() -> str:
    return dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def run_id_to_iso(run_id: str) -> str:
    try:
        d = dt.datetime.strptime(run_id, "%Y%m%d_%H%M%S")
        return d.replace(tzinfo=dt.datetime.now().astimezone().tzinfo).isoformat()
    except Exception:
        return now_iso()


# ----------------------------- reason codes -----------------------------


REASON_CODE = {
    # Network / HTTP
    "HTTP_429_RATE_LIMIT": "HTTP_429_RATE_LIMIT",
    "HTTP_403_FORBIDDEN": "HTTP_403_FORBIDDEN",
    "HTTP_5XX_UPSTREAM": "HTTP_5XX_UPSTREAM",
    "TIMEOUT_FETCH": "TIMEOUT_FETCH",
    # Auth / session
    "TOKEN_EXPIRED": "TOKEN_EXPIRED",
    "CAPTCHA_OR_BOT_CHALLENGE": "CAPTCHA_OR_BOT_CHALLENGE",
    # Data / parsing
    "EMPTY_RESULT": "EMPTY_RESULT",
    "SCHEMA_CHANGED": "SCHEMA_CHANGED",
    "FIELD_PARSE_SHIFT": "FIELD_PARSE_SHIFT",
    "DISTRIBUTION_SHIFT": "DISTRIBUTION_SHIFT",
    # Runtime / IO
    "WRITE_SNAPSHOT_FAIL": "WRITE_SNAPSHOT_FAIL",
    "UNHANDLED_RUNTIME_ERROR": "UNHANDLED_RUNTIME_ERROR",
}

REASON_CODE_TO_ERROR_TYPE = {
    "HTTP_429_RATE_LIMIT": "http_429",
    "HTTP_403_FORBIDDEN": "http_403",
    "HTTP_5XX_UPSTREAM": "http_5xx",
    "TIMEOUT_FETCH": "timeout",
    "TOKEN_EXPIRED": "auth",
    "CAPTCHA_OR_BOT_CHALLENGE": "auth",
    "EMPTY_RESULT": "empty_result",
    "SCHEMA_CHANGED": "schema_change",
    "FIELD_PARSE_SHIFT": "parse_error",
    "DISTRIBUTION_SHIFT": "validation",
    "WRITE_SNAPSHOT_FAIL": "io",
    "UNHANDLED_RUNTIME_ERROR": "runtime",
}


# ----------------------------- full logging -----------------------------


class Logger:
    def __init__(self, *, log_path: str, run_id: str, developer_key: str, print_to_console: bool = True):
        ensure_dir(os.path.dirname(log_path))
        self.log_path = log_path
        self.run_id = run_id
        self.developer_key = developer_key
        self.print_to_console = print_to_console
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")

    def _write(self, level: str, stage: str, msg: str) -> None:
        ts = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} | {self.run_id} | {self.developer_key} | {level:<5} | {stage:<10} | {msg}"
        if self.print_to_console:
            if tqdm is not None:
                try:
                    tqdm.write(line)
                except Exception:
                    print(line)
            else:
                print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def debug(self, stage: str, msg: str) -> None:
        self._write("DEBUG", stage, msg)

    def info(self, stage: str, msg: str) -> None:
        self._write("INFO", stage, msg)

    def warn(self, stage: str, msg: str) -> None:
        self._write("WARN", stage, msg)

    def error(self, stage: str, msg: str) -> None:
        self._write("ERROR", stage, msg)


class ErrorSink:
    """Append-only errors/events log as CSV (единый формат)."""

    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path))
        self.path = path
        self.cols = [
            "ts",
            "run_id",
            "developer_key",
            "level",
            "stage",
            "attempt",
            "reason_code",
            "error_type",
            "url",
            "http_status",
            "message",
            "context_json",
        ]
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.cols)
                w.writeheader()

    def emit(
        self,
        *,
        ts: str,
        run_id: str,
        developer_key: str,
        level: str,
        stage: str,
        attempt: int,
        reason_code: str,
        message: str,
        error_type: Optional[str] = None,
        url: Optional[str] = None,
        http_status: Optional[int] = None,
        context: Optional[dict] = None,
    ) -> None:
        row = {
            "ts": ts,
            "run_id": run_id,
            "developer_key": developer_key,
            "level": level,
            "stage": stage,
            "attempt": attempt,
            "reason_code": reason_code,
            "error_type": error_type or REASON_CODE_TO_ERROR_TYPE.get(reason_code, "unknown"),
            "url": url,
            "http_status": http_status,
            "message": message,
            "context_json": json.dumps(context or {}, ensure_ascii=False),
        }
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.cols)
            w.writerow(row)



# ----------------------------- progress (tqdm) -----------------------------


def progress(it, *, desc: str, total: Optional[int] = None):
    """Единый progress-bar (мягкий зелёный). Работает в окне консоли, если stdout=TTY."""
    if tqdm is None:
        return it
    return tqdm(it, desc=desc, total=total, dynamic_ncols=True, colour=SOFT_GREEN, leave=True)


# ----------------------------- result contract -----------------------------


@dataclass
class ParserResult:
    status: str  # OK/WARN/FAIL
    retries_used: int
    suspect_flags: List[str]
    output_snapshot_path: Optional[str]
    message_for_human: str
    reason_code: Optional[str] = None
    reason_summary: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "retries_used": self.retries_used,
            "suspect_flags": self.suspect_flags,
            "output_snapshot_path": self.output_snapshot_path,
            "message_for_human": self.message_for_human,
            "reason_code": self.reason_code,
            "reason_summary": self.reason_summary,
        }


# ----------------------------- stats/validation -----------------------------


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(" ", "")
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


def normalize_rooms(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s == "" or s in {"nan", "none", "null"}:
        return None
    # "евро-2" => 2
    for tok in ["евро", "euro", "евро-", "euro-"]:
        s = s.replace(tok, "")
    # keep digits only
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        return None
    try:
        v = int(digits)
        if v < 0:
            return None
        if v > 20:
            return None
        return v
    except Exception:
        return None


def compute_snapshot_stats(
    csv_path: str,
    *,
    price_col: str = "price",
    area_col: str = "area_m2",
    rooms_col: str = "rooms",
    complex_col: str = "complex",
    house_col: str = "house",
    status_col: str = "status",
    available_values: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Basic + distribution stats.

    Смысл:
    - rows / null% по ключевым полям
    - квантили площади + доля "моды" (слишком большая доля одного значения часто означает слом парсинга)
    - распределение комнат
    - число уникальных ЖК/домов (если есть колонки complex/house)
    - доля status=available (если есть колонка status)
    """
    rows = 0
    price_null = 0
    area_null = 0
    rooms_null = 0
    areas: List[float] = []
    rooms_dist: Dict[str, int] = {}
    rooms_norm_dist: Dict[str, int] = {}

    complexes = set()
    houses = set()

    status_counts: Dict[str, int] = {}
    status_null = 0
    avail_cnt = 0
    avail_set = set([s.strip().lower() for s in (available_values or ["available"])])

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rr = csv.DictReader(f)
        for row in rr:
            rows += 1

            if safe_float(row.get(price_col)) is None:
                price_null += 1

            a = safe_float(row.get(area_col))
            if a is None:
                area_null += 1
            else:
                areas.append(a)

            r_raw = row.get(rooms_col)
            if r_raw is None or str(r_raw).strip() == "" or str(r_raw).strip().lower() in {"none", "null", "nan"}:
                rooms_null += 1
            else:
                k = str(r_raw).strip()
                rooms_dist[k] = rooms_dist.get(k, 0) + 1
                rn = normalize_rooms(r_raw)
                if rn is not None:
                    kk = str(rn)
                    rooms_norm_dist[kk] = rooms_norm_dist.get(kk, 0) + 1

            # complex/house counts (optional)
            c = row.get(complex_col)
            if c is not None and str(c).strip() != "":
                complexes.add(str(c).strip())

            h = row.get(house_col)
            if h is not None and str(h).strip() != "":
                houses.add(str(h).strip())

            # status distribution (optional)
            st = row.get(status_col)
            if st is None or str(st).strip() == "" or str(st).strip().lower() in {"none", "null", "nan"}:
                status_null += 1
            else:
                s = str(st).strip().lower()
                status_counts[s] = status_counts.get(s, 0) + 1
                if s in avail_set:
                    avail_cnt += 1

    def pct(n: int, d: int) -> float:
        return round((n / d * 100.0) if d else 0.0, 3)

    stats: Dict[str, Any] = {
        "rows": rows,
        "price_null_pct": pct(price_null, rows),
        "area_null_pct": pct(area_null, rows),
        "rooms_null_pct": pct(rooms_null, rows),
        "rooms_dist_raw": rooms_dist,
        "rooms_dist_norm": rooms_norm_dist,
        "complexes_count": len(complexes),
        "houses_count": len(houses),
        "status_null_pct": pct(status_null, rows),
        "status_counts": status_counts,
        "available_pct_total": pct(avail_cnt, rows),
    }

    if areas:
        areas_sorted = sorted(areas)

        def q(p: float) -> float:
            idx = int(round((len(areas_sorted) - 1) * p))
            idx = max(0, min(len(areas_sorted) - 1, idx))
            return float(areas_sorted[idx])

        stats["area_q10"] = round(q(0.10), 3)
        stats["area_q50"] = round(q(0.50), 3)
        stats["area_q90"] = round(q(0.90), 3)

        from collections import Counter
        c = Counter([round(x, 2) for x in areas_sorted])
        v, cnt = c.most_common(1)[0]
        stats["area_mode"] = float(v)
        stats["area_mode_share"] = round(cnt / len(areas_sorted), 4)
    return stats


def find_previous_snapshot(snapshots_dir: str, *, prefix: str) -> Optional[str]:
    if not os.path.isdir(snapshots_dir):
        return None
    cands = [
        os.path.join(snapshots_dir, f)
        for f in os.listdir(snapshots_dir)
        if f.startswith(prefix) and f.lower().endswith(".csv")
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def pct_delta(new: float, old: float) -> float:
    if old <= 0:
        return 0.0
    return (new - old) / old * 100.0


def validate_against_previous(
    candidate_csv: str,
    prev_csv: Optional[str],
    cfg_validation: dict,
) -> Tuple[str, List[str], Dict[str, Any], Optional[str], Optional[str]]:
    """Return (status, flags, metrics, reason_code, reason_summary)."""
    flags: List[str] = []
    metrics: Dict[str, Any] = {}

    curr = compute_snapshot_stats(candidate_csv, available_values=cfg_validation.get('available_values'))
    metrics.update({f"curr_{k}": v for k, v in curr.items()})

    prev = None
    if prev_csv and os.path.exists(prev_csv):
        prev = compute_snapshot_stats(prev_csv, available_values=cfg_validation.get('available_values'))
        metrics.update({f"prev_{k}": v for k, v in prev.items()})

    # thresholds
    warn_row = float(cfg_validation.get("warn_row_delta_pct", 25))
    q_row = float(cfg_validation.get("quarantine_row_delta_pct", 45))

    warn_price_null = float(cfg_validation.get("warn_price_null_pct", 15))
    q_price_null = float(cfg_validation.get("quarantine_price_null_pct", 30))

    warn_area_null = float(cfg_validation.get("warn_area_null_pct", 20))
    q_area_null = float(cfg_validation.get("quarantine_area_null_pct", 40))

    warn_rooms_null = float(cfg_validation.get("warn_rooms_null_pct", 20))
    q_rooms_null = float(cfg_validation.get("quarantine_rooms_null_pct", 40))

    warn_area_mode_share = float(cfg_validation.get("warn_area_mode_share", 0.35))
    warn_area_q50_factor = float(cfg_validation.get("warn_area_q50_factor", 2.0))

    # row delta vs prev
    if prev:
        d = pct_delta(float(curr.get("rows") or 0), float(prev.get("rows") or 0))
        metrics["row_delta_pct"] = round(d, 3)
        if abs(d) >= warn_row:
            flags.append(f"WARN_ROW_DELTA_{d:.1f}%")
        if abs(d) >= q_row:
            flags.append(f"QUARANTINE_ROW_DELTA_{d:.1f}%")

    

    # complexes/houses disappearance (optional columns)
    warn_house_drop = float(cfg_validation.get("warn_house_drop_pct", 20))
    q_house_drop = float(cfg_validation.get("quarantine_house_drop_pct", 50))
    warn_complex_drop = float(cfg_validation.get("warn_complex_drop_pct", 20))
    q_complex_drop = float(cfg_validation.get("quarantine_complex_drop_pct", 50))

    if prev:
        prev_h = float(prev.get("houses_count") or 0)
        curr_h = float(curr.get("houses_count") or 0)
        if prev_h > 0:
            house_drop_pct = round((prev_h - curr_h) / prev_h * 100.0, 3)
            metrics["house_drop_pct"] = house_drop_pct
            if house_drop_pct >= warn_house_drop:
                flags.append(f"WARN_HOUSE_DROP_{house_drop_pct:.1f}%")
            if house_drop_pct >= q_house_drop:
                flags.append(f"QUARANTINE_HOUSE_DROP_{house_drop_pct:.1f}%")

        prev_c = float(prev.get("complexes_count") or 0)
        curr_c = float(curr.get("complexes_count") or 0)
        if prev_c > 0:
            complex_drop_pct = round((prev_c - curr_c) / prev_c * 100.0, 3)
            metrics["complex_drop_pct"] = complex_drop_pct
            if complex_drop_pct >= warn_complex_drop:
                flags.append(f"WARN_COMPLEX_DROP_{complex_drop_pct:.1f}%")
            if complex_drop_pct >= q_complex_drop:
                flags.append(f"QUARANTINE_COMPLEX_DROP_{complex_drop_pct:.1f}%")

        # status=available share drop (percentage points)
        warn_avail_drop_pp = float(cfg_validation.get("warn_available_drop_pp", 15))
        q_avail_drop_pp = float(cfg_validation.get("quarantine_available_drop_pp", 30))

        prev_av = float(prev.get("available_pct_total") or 0.0)
        curr_av = float(curr.get("available_pct_total") or 0.0)
        metrics["available_drop_pp"] = round(prev_av - curr_av, 3)
        if (prev_av - curr_av) >= warn_avail_drop_pp:
            flags.append(f"WARN_AVAILABLE_DROP_{(prev_av - curr_av):.1f}pp")
        if (prev_av - curr_av) >= q_avail_drop_pp:
            flags.append(f"QUARANTINE_AVAILABLE_DROP_{(prev_av - curr_av):.1f}pp")

    # null rates
    if float(curr.get("price_null_pct") or 0) >= warn_price_null:
        flags.append(f"WARN_PRICE_NULL_{curr.get('price_null_pct'):.1f}%")
    if float(curr.get("price_null_pct") or 0) >= q_price_null:
        flags.append(f"QUARANTINE_PRICE_NULL_{curr.get('price_null_pct'):.1f}%")

    if float(curr.get("area_null_pct") or 0) >= warn_area_null:
        flags.append(f"WARN_AREA_NULL_{curr.get('area_null_pct'):.1f}%")
    if float(curr.get("area_null_pct") or 0) >= q_area_null:
        flags.append(f"QUARANTINE_AREA_NULL_{curr.get('area_null_pct'):.1f}%")

    if float(curr.get("rooms_null_pct") or 0) >= warn_rooms_null:
        flags.append(f"WARN_ROOMS_NULL_{curr.get('rooms_null_pct'):.1f}%")
    if float(curr.get("rooms_null_pct") or 0) >= q_rooms_null:
        flags.append(f"QUARANTINE_ROOMS_NULL_{curr.get('rooms_null_pct'):.1f}%")

    warn_status_null = float(cfg_validation.get("warn_status_null_pct", 20))
    q_status_null = float(cfg_validation.get("quarantine_status_null_pct", 40))

    if float(curr.get("status_null_pct") or 0) >= warn_status_null:
        flags.append(f"WARN_STATUS_NULL_{curr.get('status_null_pct'):.1f}%")
    if float(curr.get("status_null_pct") or 0) >= q_status_null:
        flags.append(f"QUARANTINE_STATUS_NULL_{curr.get('status_null_pct'):.1f}%")


    # distribution sanity
    if float(curr.get("area_mode_share") or 0.0) >= warn_area_mode_share:
        flags.append(f"WARN_AREA_MODE_SHARE_{float(curr.get('area_mode_share')):.3f}")

    if prev and curr.get("area_q50") and prev.get("area_q50"):
        q50_factor = (float(curr["area_q50"]) / float(prev["area_q50"])) if float(prev["area_q50"]) > 0 else 1.0
        metrics["area_q50_factor"] = round(q50_factor, 4)
        if q50_factor >= warn_area_q50_factor or q50_factor <= (1.0 / warn_area_q50_factor):
            flags.append(f"WARN_AREA_Q50_FACTOR_{q50_factor:.2f}")

    # choose reason_code
    status = "OK"
    reason_code = None
    reason_summary = None

    if any(f.startswith("QUARANTINE_") for f in flags):
        status = "WARN"
    elif any(f.startswith("WARN_") for f in flags):
        status = "WARN"

    if status == "WARN":
        # if the most visible issue is about distribution (area/rooms), label it so
        dist_markers = ["AREA_NULL", "ROOMS_NULL", "AREA_MODE", "AREA_Q50", "HOUSE_DROP", "COMPLEX_DROP", "AVAILABLE_DROP", "STATUS_NULL"]
        if any(any(m in f for m in dist_markers) for f in flags):
            reason_code = REASON_CODE["DISTRIBUTION_SHIFT"]
            reason_summary = "Подозрение на сдвиг парсинга площади/комнат/цен (аномальные null/квантили/повторы)."
        else:
            reason_code = REASON_CODE["FIELD_PARSE_SHIFT"]
            reason_summary = "Частичные ошибки парсинга ключевых полей (см. флаги/метрики)."

    return status, flags, metrics, reason_code, reason_summary


# ----------------------------- retry / softness -----------------------------


def soften_params(base: dict, attempt: int) -> dict:
    workers = int(base.get("workers", 6))
    in_flight = int(base.get("in_flight", 60))
    submit_sleep = float(base.get("submit_sleep_sec", 0.15))
    jitter = float(base.get("jitter_sec", 0.25))

    factor = {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.35}.get(attempt, 0.35)
    return {
        **base,
        "workers": max(1, int(round(workers * factor))),
        "in_flight": max(1, int(round(in_flight * factor))),
        "submit_sleep_sec": submit_sleep * (1.0 / factor),
        "jitter_sec": jitter * (1.0 / factor),
    }


def jitter_sleep(base_sec: float, jitter_sec: float) -> None:
    time.sleep(max(0.0, base_sec + random.uniform(0.0, jitter_sec)))


# ----------------------------- reason builder (dev-friendly) -----------------------------


def build_reason_json(
    *,
    category: str,
    severity: str,
    reason_code: str,
    summary: str,
    what_happened: List[str],
    why_it_matters: List[str],
    checks: List[str],
    next_actions: List[str],
    manual_accept_rule: List[str],
    evidence: Dict[str, Any],
) -> dict:
    return {
        "category": category,
        "severity": severity,
        "reason_code": reason_code,
        "summary": summary,
        "what_happened": what_happened,
        "why_it_matters": why_it_matters,
        "checks": checks,
        "next_actions": next_actions,
        "manual_accept_rule": manual_accept_rule,
        "evidence": evidence,
    }


def emit_reason(
    *,
    base_dir: str,
    logger: Logger,
    errors: ErrorSink,
    ts: str,
    run_id: str,
    developer_key: str,
    attempt: int,
    stage: str,
    severity: str,
    reason_code: str,
    summary: str,
    reason_json: dict,
    url: Optional[str] = None,
    http_status: Optional[int] = None,
    context: Optional[dict] = None,
) -> None:
    # write reason.json
    write_json(os.path.join(base_dir, "reason.json"), reason_json)

    # log + errors.csv
    msg = f"{reason_code} | {summary}"
    if severity.lower() == "fail":
        logger.error(stage, msg)
        level = "ERROR"
    else:
        logger.warn(stage, msg)
        level = "WARN"

    errors.emit(
        ts=dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        run_id=run_id,
        developer_key=developer_key,
        level=level,
        stage=stage,
        attempt=attempt,
        reason_code=reason_code,
        message=summary,
        error_type=REASON_CODE_TO_ERROR_TYPE.get(reason_code, "unknown"),
        url=url,
        http_status=http_status,
        context=context or {},
    )


# ----------------------------- core: fetching/parsing -----------------------------


def collect_lots(cfg: dict, logger: Logger, errors: ErrorSink, *, attempt: int, runtime: dict) -> List[dict]:
    """TODO: Replace this function with real network parsing.

    В реальном парсере:
    - любые HTTP/парсинг ошибки: фиксируй через errors.emit(...)
    - при 401/403: reason_code TOKEN_EXPIRED/HTTP_403_FORBIDDEN
    - при 429: reason_code HTTP_429_RATE_LIMIT
    - при schema change: SCHEMA_CHANGED
    """
    # --- DEMO MODE (works out-of-the-box) ---
    if cfg.get("demo_mode", False):
        sample_file = cfg.get("demo_sample_file", "sample_lots.json")
        sample_path = os.path.join(os.path.dirname(__file__), sample_file)
        if not os.path.exists(sample_path):
            raise RuntimeError(f"demo sample file not found: {sample_path}")
        lots = read_json(sample_path)
        if not isinstance(lots, list):
            raise RuntimeError("demo sample must be a JSON list")
        return lots

    raise NotImplementedError("collect_lots() is not implemented. Set demo_mode=true or implement real fetching/parsing.")


def normalize_row(lot: dict, *, developer_key: str, source: str, captured_at: str) -> dict:
    """Make row robust: never crash on missing fields."""
    row: Dict[str, Any] = {}
    row["developer_key"] = developer_key
    row["captured_at"] = captured_at
    row["source"] = source

    row["lot_id"] = lot.get("lot_id") or lot.get("id") or lot.get("uid")
    row["complex"] = lot.get("complex")
    row["house"] = lot.get("house")
    row["rooms"] = lot.get("rooms")
    row["area_m2"] = lot.get("area_m2") or lot.get("area")
    row["floor"] = lot.get("floor")
    row["price"] = lot.get("price")
    row["status"] = lot.get("status")
    row["url"] = lot.get("url")

    for k, v in lot.items():
        if k in row:
            continue
        if isinstance(v, (str, int, float)) or v is None:
            row[f"extra_{k}"] = v
    return row


def write_snapshot(csv_path: str, rows: List[dict]) -> None:
    ensure_dir(os.path.dirname(csv_path))
    if not rows:
        raise RuntimeError("no rows to write")

    base_cols = ["developer_key", "captured_at", "source"]
    other_cols = sorted({k for r in rows for k in r.keys() if k not in set(base_cols)})
    cols = base_cols + other_cols

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


# ----------------------------- main -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg = read_json(cfg_path)

    developer_key = str(cfg.get("developer_key") or "developer_key").strip()
    city = str(cfg.get("city") or "city").strip()
    source = str(cfg.get("source") or "").strip() or "(unknown)"

    run_id = os.environ.get("RUN_ID") or dt.datetime.now().astimezone().replace(microsecond=0).strftime("%Y%m%d_%H%M%S")
    captured_at = run_id_to_iso(run_id)

    out_cfg = cfg.get("output") or {}
    suite_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir_rel = out_cfg.get("base_dir") or f"parser/{developer_key}/output_{developer_key}"
    base_dir = os.path.normpath(os.path.join(suite_root, base_dir_rel))

    snapshots_dir = os.path.join(base_dir, out_cfg.get("snapshots_raw_dir", "snapshots_raw"))
    logs_dir = os.path.join(base_dir, out_cfg.get("logs_dir", "logs"))
    errors_dir = os.path.join(base_dir, out_cfg.get("errors_dir", "errors"))
    quarantine_dir = os.path.join(base_dir, out_cfg.get("quarantine_dir", "quarantine"))

    for d in [snapshots_dir, logs_dir, errors_dir, quarantine_dir]:
        ensure_dir(d)

    log_path = os.path.join(logs_dir, f"{developer_key}__{city}__{run_id}.log")
    logger = Logger(
        log_path=log_path,
        run_id=run_id,
        developer_key=developer_key,
        print_to_console=bool(cfg.get("print_logs_to_console", True)),
    )

    errors_csv_path = os.path.join(errors_dir, "errors.csv")
    errors = ErrorSink(errors_csv_path)

    logger.info("bootstrap", f"START | city={city} | config={cfg_path}")
    rel_base_dir = os.path.relpath(base_dir, start=suite_root).replace("\\", "/")
    logger.info("bootstrap", f"output.base_dir={rel_base_dir}")
    rel_errors_csv = os.path.relpath(errors_csv_path, start=suite_root).replace("\\", "/")
    logger.info("bootstrap", f"errors.csv={rel_errors_csv}")

    max_retries = int(cfg.get("max_retries", 3))
    retry_base_sleep = float(cfg.get("retry_base_sleep_sec", 9))
    base_soft = cfg.get("softness") or {}
    validation_cfg = cfg.get("validation") or {}

    prev = find_previous_snapshot(snapshots_dir, prefix=f"{developer_key}__{city}__")

    final_result: Optional[ParserResult] = None
    final_metrics: Dict[str, Any] = {}
    last_trace: Optional[str] = None
    final_reason_json: Optional[dict] = None

    for attempt in range(0, max_retries + 1):
        softened = soften_params(base_soft, attempt)
        logger.info(
            "retry",
            f"Attempt {attempt}/{max_retries}: workers={softened.get('workers')} in_flight={softened.get('in_flight')} "
            f"submit_sleep={softened.get('submit_sleep_sec'):.3f}s jitter={softened.get('jitter_sec'):.3f}s",
        )

        try:
            lots = collect_lots(cfg, logger, errors, attempt=attempt, runtime=softened)

            if not lots:
                # hard fail: got empty list
                reason_code = REASON_CODE["EMPTY_RESULT"]
                summary = "Пустой результат (0 лотов) — источник вернул пустой список, это не похоже на норму."
                evidence = {
                    "run_id": run_id,
                    "developer_key": developer_key,
                    "attempt": attempt,
                    "stage": "fetch",
                    "source": source,
                    "prev_snapshot": os.path.basename(prev) if prev else None,
                    "log_path": log_path,
                    "errors_csv": errors_csv_path,
                }
                final_reason_json = build_reason_json(
                    category="data",
                    severity="fail",
                    reason_code=reason_code,
                    summary=summary,
                    what_happened=[f"Получили 0 лотов для city={city}."],
                    why_it_matters=["Snapshot будет пустой/некорректный, публиковать нельзя."],
                    checks=[
                        "Проверь параметры фильтра города/URL/endpoint в collect_lots().",
                        "Проверь блокировки 403/капчу/требование токена по логам и errors.csv.",
                    ],
                    next_actions=[
                        "Если сайт требует токен/сессию — реализуй browser-assist (получение токена без падения).",
                        "Если изменился endpoint/параметры — обнови запрос и парсинг списка лотов.",
                    ],
                    manual_accept_rule=["Не принимать вручную (данные отсутствуют)."],
                    evidence=evidence,
                )
                emit_reason(
                    base_dir=base_dir,
                    logger=logger,
                    errors=errors,
                    ts=captured_at,
                    run_id=run_id,
                    developer_key=developer_key,
                    attempt=attempt,
                    stage="fetch",
                    severity="fail",
                    reason_code=reason_code,
                    summary=summary,
                    reason_json=final_reason_json,
                    context={"city": city, "source": source},
                )
                raise RuntimeError("empty lots list")

            # normalize
            rows: List[dict] = []
            bad_rows = 0
            for i, lot in enumerate(lots):
                try:
                    if not isinstance(lot, dict):
                        bad_rows += 1
                        errors.emit(
                            ts=now_iso(),
                            run_id=run_id,
                            developer_key=developer_key,
                            level="WARN",
                            stage="normalize",
                            attempt=attempt,
                            reason_code=REASON_CODE["FIELD_PARSE_SHIFT"],
                            message="lot is not a dict",
                            context={"index": i, "type": str(type(lot))},
                        )
                        continue

                    row = normalize_row(lot, developer_key=developer_key, source=source, captured_at=captured_at)
                    if not row.get("lot_id"):
                        bad_rows += 1
                        errors.emit(
                            ts=now_iso(),
                            run_id=run_id,
                            developer_key=developer_key,
                            level="WARN",
                            stage="normalize",
                            attempt=attempt,
                            reason_code=REASON_CODE["FIELD_PARSE_SHIFT"],
                            message="missing lot_id (skipped)",
                            url=lot.get("url"),
                            context={"index": i},
                        )
                        continue
                    rows.append(row)
                except Exception as e:
                    bad_rows += 1
                    errors.emit(
                        ts=now_iso(),
                        run_id=run_id,
                        developer_key=developer_key,
                        level="WARN",
                        stage="normalize",
                        attempt=attempt,
                        reason_code=REASON_CODE["FIELD_PARSE_SHIFT"],
                        message=f"exception while normalizing: {e}",
                        url=lot.get("url") if isinstance(lot, dict) else None,
                        context={"index": i},
                    )

            if not rows:
                reason_code = REASON_CODE["FIELD_PARSE_SHIFT"]
                summary = "После нормализации не осталось валидных строк (lot_id отсутствует или парсинг сломан)."
                evidence = {
                    "run_id": run_id,
                    "developer_key": developer_key,
                    "attempt": attempt,
                    "stage": "normalize",
                    "bad_rows": bad_rows,
                    "source": source,
                    "log_path": log_path,
                    "errors_csv": errors_csv_path,
                }
                final_reason_json = build_reason_json(
                    category="parsing",
                    severity="fail",
                    reason_code=reason_code,
                    summary=summary,
                    what_happened=[f"Нормализовано 0 строк, bad_rows={bad_rows}."],
                    why_it_matters=["Snapshot будет пустой/невалидный."],
                    checks=[
                        "Открой errors/errors.csv и посмотри причины в stage=normalize.",
                        "Проверь, что lot_id действительно есть в данных (какой ключ у источника).",
                    ],
                    next_actions=[
                        "Обнови normalize_row(): корректно достань lot_id и ключевые поля.",
                        "Если источник поменял схему — добавь fallback ключей (.get()).",
                    ],
                    manual_accept_rule=["Не принимать вручную (нет данных)."],
                    evidence=evidence,
                )
                emit_reason(
                    base_dir=base_dir,
                    logger=logger,
                    errors=errors,
                    ts=captured_at,
                    run_id=run_id,
                    developer_key=developer_key,
                    attempt=attempt,
                    stage="normalize",
                    severity="fail",
                    reason_code=reason_code,
                    summary=summary,
                    reason_json=final_reason_json,
                )
                raise RuntimeError("no valid rows after normalization")

            snapshot_name = f"{developer_key}__{city}__{run_id}.csv"
            snapshot_path = os.path.join(snapshots_dir, snapshot_name)

            try:
                write_snapshot(snapshot_path, rows)
            except Exception as e:
                reason_code = REASON_CODE["WRITE_SNAPSHOT_FAIL"]
                summary = f"Не удалось записать snapshot CSV: {e}"
                evidence = {
                    "run_id": run_id,
                    "developer_key": developer_key,
                    "attempt": attempt,
                    "stage": "write_snapshot",
                    "snapshot_path": snapshot_path,
                    "log_path": log_path,
                }
                final_reason_json = build_reason_json(
                    category="runtime",
                    severity="fail",
                    reason_code=reason_code,
                    summary=summary,
                    what_happened=[f"Ошибка записи файла {snapshot_path}."],
                    why_it_matters=["Snapshot не сохранён, запуск не может быть опубликован."],
                    checks=[
                        "Проверь права на папку output_<dev>/snapshots_raw.",
                        "Проверь, нет ли блокировки файла (открыт в Excel).",
                        "Проверь свободное место на диске.",
                    ],
                    next_actions=["Исправь проблему IO и перезапусти."],
                    manual_accept_rule=["Не принимать вручную (snapshot не создан)."],
                    evidence=evidence,
                )
                emit_reason(
                    base_dir=base_dir,
                    logger=logger,
                    errors=errors,
                    ts=captured_at,
                    run_id=run_id,
                    developer_key=developer_key,
                    attempt=attempt,
                    stage="write_snapshot",
                    severity="fail",
                    reason_code=reason_code,
                    summary=summary,
                    reason_json=final_reason_json,
                    context={"snapshot_path": snapshot_path},
                )
                raise

            logger.info("write", f"Snapshot saved: {snapshot_path} | rows={len(rows)} | bad_rows={bad_rows}")

            # validation vs previous
            v_status, flags, metrics, v_reason_code, v_reason_summary = validate_against_previous(snapshot_path, prev, validation_cfg)
            final_metrics = metrics

            # decision
            status = "OK" if v_status == "OK" and bad_rows == 0 else "WARN"
            reason_code = v_reason_code
            reason_summary = v_reason_summary

            if bad_rows > 0:
                flags = list(flags) + [f"PARTIAL_NORMALIZE_ERRORS_{bad_rows}"]
                if not reason_code:
                    reason_code = REASON_CODE["FIELD_PARSE_SHIFT"]
                    reason_summary = f"Часть лотов пропущена при нормализации: bad_rows={bad_rows}."

            msg = "OK"
            if status == "WARN":
                msg = "Данные собраны, но есть подозрительные изменения/частичные ошибки. См. reason.json и errors.csv."

            final_result = ParserResult(
                status=status,
                retries_used=attempt,
                suspect_flags=flags,
                output_snapshot_path=os.path.relpath(snapshot_path, start=suite_root).replace("\\", "/"),
                message_for_human=msg,
                reason_code=reason_code,
                reason_summary=reason_summary,
            )

            # if WARN/FAIL — write reason.json (dev-friendly)
            if final_result.status in {"WARN", "FAIL"}:
                # build evidence with paths
                evidence = {
                    "run_id": run_id,
                    "developer_key": developer_key,
                    "attempt": attempt,
                    "stage": "validate",
                    "source": source,
                    "snapshot": os.path.basename(snapshot_path),
                    "prev_snapshot": os.path.basename(prev) if prev else None,
                    "log_path": log_path,
                    "errors_csv": errors_csv_path,
                    "flags": flags,
                    "metrics_preview": {
                        k: metrics.get(k)
                        for k in ["row_delta_pct", "curr_rows", "prev_rows", "curr_price_null_pct", "curr_area_null_pct", "curr_rooms_null_pct", "area_q50_factor"]
                        if k in metrics
                    },
                }
                rc = reason_code or REASON_CODE["FIELD_PARSE_SHIFT"]
                summary = reason_summary or "Snapshot помечен как WARNING по результатам валидации."
                final_reason_json = build_reason_json(
                    category="validation",
                    severity="warn",
                    reason_code=rc,
                    summary=summary,
                    what_happened=[
                        f"rows={metrics.get('curr_rows')} (prev={metrics.get('prev_rows')})",
                        f"price_null_pct={metrics.get('curr_price_null_pct')} (prev={metrics.get('prev_price_null_pct')})",
                        f"area_null_pct={metrics.get('curr_area_null_pct')} (prev={metrics.get('prev_area_null_pct')})",
                        f"rooms_null_pct={metrics.get('curr_rooms_null_pct')} (prev={metrics.get('prev_rooms_null_pct')})",
                    ],
                    why_it_matters=["Есть риск неполной/искажённой выгрузки (возможен сдвиг парсинга или блокировка)."],
                    checks=[
                        "Открой snapshot и вручную проверь 20–50 строк: price/area_m2/rooms выглядят правдоподобно.",
                        "Открой errors/errors.csv и посмотри события текущего RUN_ID.",
                        "Если видны признаки 'x10/x100' по площади/цене — исправь нормализацию форматов/единиц.",
                    ],
                    next_actions=[
                        "Если это реальное изменение данных — можно принять snapshot вручную (manual accept) после проверки.",
                        "Если это поломка парсинга — обнови transform/normalize и повтори запуск.",
                    ],
                    manual_accept_rule=[
                        "Можно принять, если после ручной проверки значения корректны и rows не аномально малы.",
                        "Не принимать, если ключевые поля массово пустые/нули или значения явно неадекватны.",
                    ],
                    evidence=evidence,
                )
                emit_reason(
                    base_dir=base_dir,
                    logger=logger,
                    errors=errors,
                    ts=captured_at,
                    run_id=run_id,
                    developer_key=developer_key,
                    attempt=attempt,
                    stage="validate",
                    severity="warn",
                    reason_code=rc,
                    summary=summary,
                    reason_json=final_reason_json,
                    context={"flags": flags},
                )

            # stop policy
            if final_result.status == "OK":
                logger.info("retry", "Status OK — stop retries")
                break

            if attempt < max_retries:
                logger.warn("retry", f"Status WARN — retry after {retry_base_sleep}s (softer mode)")
                jitter_sleep(retry_base_sleep, float(softened.get("jitter_sec", 0.25)))
                continue

            logger.warn("retry", "Reached max retries")
            break

        except Exception as e:
            last_trace = traceback.format_exc()
            logger.error("runtime", f"Attempt failed: {e}")
            errors.emit(
                ts=now_iso(),
                run_id=run_id,
                developer_key=developer_key,
                level="ERROR",
                stage="runtime",
                attempt=attempt,
                reason_code=REASON_CODE["UNHANDLED_RUNTIME_ERROR"],
                message=str(e),
                context={"trace": last_trace[-8000:] if last_trace else None},
            )
            if attempt < max_retries:
                logger.warn("retry", f"Retry after {retry_base_sleep}s")
                jitter_sleep(retry_base_sleep, float(softened.get("jitter_sec", 0.25)))
                continue

            # final FAIL
            reason_code = REASON_CODE["UNHANDLED_RUNTIME_ERROR"]
            summary = "Не удалось завершить парсинг из-за ошибки выполнения (см. лог/trace)."
            evidence = {
                "run_id": run_id,
                "developer_key": developer_key,
                "attempt": attempt,
                "stage": "runtime",
                "source": source,
                "log_path": log_path,
                "errors_csv": errors_csv_path,
            }
            final_reason_json = build_reason_json(
                category="runtime",
                severity="fail",
                reason_code=reason_code,
                summary=summary,
                what_happened=[str(e)],
                why_it_matters=["Snapshot невалиден или не создан."],
                checks=[
                    "Открой лог парсера и найди ERROR runtime.",
                    "Открой errors/errors.csv и посмотри context_json.trace.",
                ],
                next_actions=["Исправь причину исключения и перезапусти парсер."],
                manual_accept_rule=["Не принимать вручную, пока не понятно, что выгрузка корректна."],
                evidence=evidence,
            )
            emit_reason(
                base_dir=base_dir,
                logger=logger,
                errors=errors,
                ts=captured_at,
                run_id=run_id,
                developer_key=developer_key,
                attempt=attempt,
                stage="runtime",
                severity="fail",
                reason_code=reason_code,
                summary=summary,
                reason_json=final_reason_json,
                context={"exception": str(e)},
            )

            final_result = ParserResult(
                status="FAIL",
                retries_used=attempt,
                suspect_flags=["HARD_FAIL_RUNTIME"],
                output_snapshot_path=None,
                message_for_human="Парсер завершился с FAIL. См. reason.json / errors.csv / лог.",
                reason_code=reason_code,
                reason_summary=summary,
            )
            break

    if final_result is None:
        final_result = ParserResult(
            status="FAIL",
            retries_used=max_retries,
            suspect_flags=["NO_RESULT"],
            output_snapshot_path=None,
            message_for_human="Не сформирован результат. Проверь логи.",
            reason_code=REASON_CODE["UNHANDLED_RUNTIME_ERROR"],
            reason_summary="Нет итогового результата (unexpected).",
        )

    # write result.json (always)
    result_path = os.path.join(base_dir, "result.json")
    payload = {
        **final_result.to_dict(),
        "run_id": run_id,
        "developer_key": developer_key,
        "city": city,
        "source": source,
        "captured_at": captured_at,
        "metrics": final_metrics,
        "paths": {
            "log_path": os.path.relpath(log_path, start=suite_root).replace("\\", "/"),
            "errors_csv": os.path.relpath(errors_csv_path, start=suite_root).replace("\\", "/"),
        },
    }
    write_json(result_path, payload)
    logger.info("finalize", f"Wrote result.json: {result_path}")

    # reason.json already written via emit_reason; if WARN/FAIL but not emitted (edge), emit minimal
    if final_result.status in {"WARN", "FAIL"} and not os.path.exists(os.path.join(base_dir, "reason.json")):
        rc = final_result.reason_code or REASON_CODE["FIELD_PARSE_SHIFT"]
        summary = final_result.reason_summary or "WARNING/FAIL без детализации (edge case)."
        evidence = {"run_id": run_id, "developer_key": developer_key, "log_path": log_path, "errors_csv": errors_csv_path}
        rj = build_reason_json(
            category="unknown",
            severity=("fail" if final_result.status == "FAIL" else "warn"),
            reason_code=rc,
            summary=summary,
            what_happened=["См. result.json / лог."],
            why_it_matters=["Нужна проверка корректности snapshot."],
            checks=["Открой лог парсера и errors.csv."],
            next_actions=["Проверь и перезапусти при необходимости."],
            manual_accept_rule=["Принимать только после ручной проверки."],
            evidence=evidence,
        )
        emit_reason(
            base_dir=base_dir,
            logger=logger,
            errors=errors,
            ts=captured_at,
            run_id=run_id,
            developer_key=developer_key,
            attempt=final_result.retries_used,
            stage="finalize",
            severity=("fail" if final_result.status == "FAIL" else "warn"),
            reason_code=rc,
            summary=summary,
            reason_json=rj,
        )

    logger.info("finalize", f"DONE status={final_result.status} retries_used={final_result.retries_used}")

    # Exit code policy: never crash pipeline; launcher decides by result.json.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
