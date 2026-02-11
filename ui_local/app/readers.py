from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd


def safe_resolve(root: Path, path: str | Path) -> Path:
    root = root.resolve()
    candidate = (root / Path(path)).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("Unsafe path traversal blocked")
    return candidate


def safe_read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def safe_read_csv(path: Path, usecols: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, nrows=nrows)
    except Exception:
        return pd.DataFrame()


def safe_tail(path: Path, n: int = 120) -> str:
    try:
        with path.open("rb") as f:
            lines = f.readlines()[-n:]
        return b"".join(lines).decode("utf-8", errors="replace")
    except Exception as e:
        return f"Could not read log tail: {e}"


def preview_file(path: Path, max_rows: int = 40) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return {"type": "json", "content": safe_read_json(path, default={})}
    if suffix == ".csv":
        df = safe_read_csv(path, nrows=max_rows)
        return {"type": "csv", "columns": list(df.columns), "rows": df.to_dict("records")}
    return {"type": "text", "content": safe_tail(path, max_rows)}


def read_zip_json_preview(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    try:
        with ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("reason.json") or name.endswith("result.json"):
                    with zf.open(name) as f:
                        data[name] = json.loads(f.read().decode("utf-8", errors="replace"))
    except Exception as e:
        data["error"] = str(e)
    return data


def csv_head_text(path: Path, rows: int = 15) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            out = []
            for i, row in enumerate(reader):
                out.append(",".join(row))
                if i >= rows:
                    break
            return "\n".join(out)
    except Exception as e:
        return f"CSV preview error: {e}"
