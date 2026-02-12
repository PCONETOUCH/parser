from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOTS_ROOT = ROOT_DIR / "launcher" / "data" / "published_snapshots"
DATA_DIR = Path(__file__).resolve().parent / "data"

STATUS_ALIASES = {
    "sold": "sold",
    "продано": "sold",
    "available": "available",
    "в продаже": "available",
    "свободно": "available",
    "reserved": "reserved",
    "бронь": "reserved",
}

COLUMN_ALIASES = {
    "captured_at": ["captured_at", "snapshot_at", "parsed_at", "created_at", "timestamp"],
    "developer_key": ["developer_key", "developer", "builder", "company"],
    "project_name": ["project_name", "project", "complex_name", "jk", "residential_complex"],
    "house_name": ["house_name", "house", "building", "corpus", "section"],
    "flat_id": ["flat_id", "unit_id", "lot_id", "id", "apartment_id", "flat"],
    "status_bucket": ["status_bucket", "status", "state", "availability", "flat_status"],
    "rooms": ["rooms", "room_count", "rooms_count"],
    "area_m2": ["area_m2", "area", "square", "total_area"],
    "floor": ["floor", "storey", "level"],
    "price_rub": ["price_rub", "price", "amount", "cost", "price_total"],
    "price_m2_raw": ["price_m2_raw", "price_m2", "price_per_m2", "sqm_price"],
}


@dataclass
class SnapshotFile:
    run_id: str
    path: Path


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def discover_snapshot_files() -> list[SnapshotFile]:
    files: list[SnapshotFile] = []
    if not SNAPSHOTS_ROOT.exists():
        return files
    for run_dir in sorted([p for p in SNAPSHOTS_ROOT.iterdir() if p.is_dir()]):
        for csv_path in sorted(run_dir.glob("*.csv")):
            files.append(SnapshotFile(run_id=run_dir.name, path=csv_path))
    return files


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols:
            return cols[candidate.lower()]
    return None


def _parse_run_id_to_ts(run_id: str) -> pd.Timestamp:
    ts = pd.to_datetime(run_id, format="%Y%m%d_%H%M%S", utc=True, errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp("1970-01-01", tz="UTC")
    return ts


def _normalize_status(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    return STATUS_ALIASES.get(text, text if text else "unknown")


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _segment(rooms: object, area_m2: object) -> str:
    room_part = "студии" if pd.isna(rooms) else f"{int(rooms) if float(rooms).is_integer() else rooms}-комн"
    if pd.isna(area_m2):
        return f"{room_part} / площадь неизвестна"
    area = float(area_m2)
    if area < 35:
        area_part = "до 35 м²"
    elif area < 50:
        area_part = "35-50 м²"
    elif area < 70:
        area_part = "50-70 м²"
    elif area < 90:
        area_part = "70-90 м²"
    else:
        area_part = "90+ м²"
    return f"{room_part} / {area_part}"


def normalize_snapshot(run_id: str, csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    norm = pd.DataFrame()

    for target_col, aliases in COLUMN_ALIASES.items():
        source_col = _find_column(raw, aliases)
        norm[target_col] = raw[source_col] if source_col else pd.NA

    fallback_ts = _parse_run_id_to_ts(run_id)
    captured = pd.to_datetime(norm["captured_at"], utc=True, errors="coerce")
    norm["captured_at"] = captured.fillna(fallback_ts)

    norm["run_id"] = run_id
    norm["status_bucket"] = norm["status_bucket"].map(_normalize_status)
    norm["developer_key"] = norm["developer_key"].fillna("unknown_developer").astype(str).str.strip()
    norm["project_name"] = norm["project_name"].fillna("Неизвестный проект").astype(str).str.strip()
    norm["house_name"] = norm["house_name"].fillna("Дом/корпус не указан").astype(str).str.strip()
    norm["flat_id"] = norm["flat_id"].fillna(raw.index.astype(str)).astype(str).str.strip()

    norm["rooms"] = _to_numeric(norm["rooms"])
    norm["area_m2"] = _to_numeric(norm["area_m2"])
    norm["floor"] = _to_numeric(norm["floor"])
    norm["price_rub"] = _to_numeric(norm["price_rub"])
    norm["price_m2_raw"] = _to_numeric(norm["price_m2_raw"])

    derived_price_m2 = norm["price_rub"] / norm["area_m2"]
    norm["price_m2_effective"] = norm["price_m2_raw"].where(norm["price_m2_raw"].notna(), derived_price_m2)
    norm["price_source"] = "parser"
    norm.loc[norm["price_m2_raw"].isna() & norm["price_m2_effective"].notna(), "price_source"] = "derived"
    norm.loc[norm["price_m2_effective"].isna(), "price_source"] = "none"

    norm["manual_rule_id"] = pd.NA
    norm["price_kind"] = "base"
    norm["price_note"] = pd.NA
    norm["segment"] = [
        _segment(rooms=r, area_m2=a)
        for r, a in zip(norm["rooms"], norm["area_m2"])
    ]

    ordered_cols = [
        "run_id",
        "captured_at",
        "developer_key",
        "project_name",
        "house_name",
        "flat_id",
        "status_bucket",
        "rooms",
        "area_m2",
        "floor",
        "price_rub",
        "price_m2_raw",
        "price_m2_effective",
        "price_source",
        "manual_rule_id",
        "price_kind",
        "price_note",
        "segment",
    ]
    return norm[ordered_cols]


def load_all_snapshots() -> pd.DataFrame:
    frames = []
    for snapshot in discover_snapshot_files():
        frames.append(normalize_snapshot(snapshot.run_id, snapshot.path))
    if not frames:
        return pd.DataFrame(
            columns=[
                "run_id",
                "captured_at",
                "developer_key",
                "project_name",
                "house_name",
                "flat_id",
                "status_bucket",
                "rooms",
                "area_m2",
                "floor",
                "price_rub",
                "price_m2_raw",
                "price_m2_effective",
                "price_source",
                "manual_rule_id",
                "price_kind",
                "price_note",
                "segment",
            ]
        )
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows.sort_values(["captured_at", "developer_key", "project_name", "flat_id"], kind="stable")
    return all_rows
