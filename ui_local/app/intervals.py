from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import pandas as pd

SNAP_RE = re.compile(r"(?P<dev>.+?)__.+?__(?P<dt>\d{8}_\d{6})\.csv$")


def parse_snapshot_dt(path: Path) -> datetime | None:
    m = SNAP_RE.search(path.name)
    if m:
        try:
            return datetime.strptime(m.group("dt"), "%Y%m%d_%H%M%S")
        except Exception:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None


def build_daily_close_index(snapshot_files: list[Path]) -> pd.DataFrame:
    rows = []
    for p in snapshot_files:
        dt = parse_snapshot_dt(p)
        if dt is None:
            continue
        rows.append({"path": p, "captured_at": dt, "close_date": dt.date()})
    if not rows:
        return pd.DataFrame(columns=["path", "captured_at", "close_date"])
    df = pd.DataFrame(rows).sort_values("captured_at")
    return df.groupby("close_date", as_index=False).tail(1).sort_values("close_date")


def build_intervals(close_df: pd.DataFrame) -> list[dict]:
    if close_df.empty:
        return []
    out = []
    prev_date = None
    for _, row in close_df.iterrows():
        curr = row["close_date"]
        start = curr if prev_date is None else (prev_date + pd.Timedelta(days=1)).date()
        days = (curr - start).days + 1
        label = f"{start}…{curr} ({days}д)"
        out.append({"start": str(start), "end": str(curr), "interval_days": days, "label": label, "path": str(row["path"])})
        prev_date = curr
    return out
