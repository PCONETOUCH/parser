from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .intervals import build_daily_close_index, build_intervals
from .schema import resolve_schema
from .segments import classify_segment


def _sold_like(status: str, ignore_statuses: list[str]) -> bool:
    return str(status).lower().strip() in ignore_statuses


def _read_snapshot(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=128)
def compute_run_dashboard(run_dir: str, ignore_statuses_key: str, synonyms_key: str):
    run = Path(run_dir)
    files = sorted(run.glob("*.csv"))
    close_df = build_daily_close_index(files)
    intervals = build_intervals(close_df)
    if close_df.empty:
        return {"kpi": {}, "intervals": [], "charts": {}, "insights": []}
    latest_path = Path(close_df.iloc[-1]["path"])
    latest_df = _read_snapshot(latest_path)
    schema = resolve_schema(list(latest_df.columns))
    lot = schema.get("lot_id")
    status = schema.get("status")
    price_m2 = schema.get("price_m2")
    project = schema.get("project")

    active_inventory = 0
    median_price = None
    unknown_project_share = 1.0
    null_price_effective_pct = 0.0
    if not latest_df.empty:
        if status and lot:
            active_inventory = int((~latest_df[status].fillna("").map(lambda s: _sold_like(s, ignore_statuses_key.split("|")))).sum())
        if price_m2 in latest_df.columns:
            median_price = float(pd.to_numeric(latest_df[price_m2], errors="coerce").median()) if not latest_df.empty else None
        if project and project in latest_df.columns:
            unknown_project_share = float((latest_df[project].fillna("").astype(str).str.strip() == "").mean())
        if status and price_m2 and status in latest_df.columns and price_m2 in latest_df.columns:
            eff = latest_df[~latest_df[status].fillna("").str.lower().isin(ignore_statuses_key.split("|"))]
            if len(eff) > 0:
                null_price_effective_pct = float(pd.to_numeric(eff[price_m2], errors="coerce").isna().mean())

    sales_points = []
    for i in range(1, len(close_df)):
        prev_df = _read_snapshot(Path(close_df.iloc[i - 1]["path"]))
        curr_df = _read_snapshot(Path(close_df.iloc[i]["path"]))
        rs = resolve_schema(list(curr_df.columns))
        lotc, stc = rs.get("lot_id"), rs.get("status")
        if not lotc or lotc not in prev_df.columns or lotc not in curr_df.columns:
            continue
        prev_active = set(prev_df.loc[~prev_df.get(stc, "").astype(str).str.lower().isin(ignore_statuses_key.split("|")), lotc].dropna().astype(str)) if stc in prev_df.columns else set(prev_df[lotc].dropna().astype(str))
        curr_active = set(curr_df.loc[~curr_df.get(stc, "").astype(str).str.lower().isin(ignore_statuses_key.split("|")), lotc].dropna().astype(str)) if stc in curr_df.columns else set(curr_df[lotc].dropna().astype(str))
        removed = max(len(prev_active - curr_active), 0)
        sold_status = 0
        if stc and stc in prev_df.columns and stc in curr_df.columns:
            prev_m = prev_df.set_index(lotc)[stc].astype(str).str.lower()
            curr_m = curr_df.set_index(lotc)[stc].astype(str).str.lower()
            common = prev_m.index.intersection(curr_m.index)
            sold_status = int(((~prev_m.loc[common].isin(ignore_statuses_key.split("|"))) & (curr_m.loc[common].isin(ignore_statuses_key.split("|")))).sum())
        sales = max(removed, sold_status)
        days = intervals[i]["interval_days"] if i < len(intervals) else 1
        sales_points.append({"label": intervals[i]["label"], "sales": sales, "sales_per_day": sales / max(days, 1)})

    sales_latest = sales_points[-1]["sales"] if sales_points else 0
    sales_latest_per_day = sales_points[-1]["sales_per_day"] if sales_points else 0

    return {
        "kpi": {
            "sales_proxy": sales_latest,
            "sales_per_day": sales_latest_per_day,
            "active_inventory": active_inventory,
            "median_price_m2": median_price,
            "quarantine_count": 0,
            "data_coverage": 1 - unknown_project_share,
            "unknown_project_share": unknown_project_share,
            "null_price_effective_pct": null_price_effective_pct,
        },
        "intervals": intervals,
        "charts": {
            "sales": sales_points,
            "inventory": [{"date": str(r["close_date"]), "value": active_inventory} for _, r in close_df.iterrows()],
        },
        "insights": [],
    }


def build_monthly_parity(df: pd.DataFrame, schema: dict[str, str | None]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    project = schema.get("project")
    price_m2 = schema.get("price_m2")
    rooms = schema.get("rooms")
    area = schema.get("area")
    if not project or not price_m2:
        return pd.DataFrame()
    d = df.copy()
    d["segment"] = d.apply(lambda r: classify_segment(r.get(rooms), r.get(area)), axis=1)
    d[price_m2] = pd.to_numeric(d[price_m2], errors="coerce")
    out = d.groupby([project, "segment"])[price_m2].agg(["min", "max", "median"]).reset_index()
    return out
