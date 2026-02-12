from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

RULES_PATH = Path(__file__).resolve().parent / "manual_price_rules.yaml"
BACKUP_DIR = Path(__file__).resolve().parent / "data" / "rules_backups"
AUDIT_COLUMNS = ["run_id", "rule_id", "matched_rows", "applied_rows"]


@dataclass
class RuleApplicationResult:
    df: pd.DataFrame
    audit: pd.DataFrame


def load_rules() -> dict[str, Any]:
    if not RULES_PATH.exists():
        return {"rules": []}
    data = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")) or {}
    data.setdefault("rules", [])
    return data


def save_rules(data: dict[str, Any]) -> None:
    RULES_PATH.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def backup_rules() -> Path | None:
    if not RULES_PATH.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"manual_price_rules_{stamp}.yaml"
    shutil.copy2(RULES_PATH, backup_path)
    return backup_path


def latest_backup() -> Path | None:
    if not BACKUP_DIR.exists():
        return None
    backups = sorted(BACKUP_DIR.glob("manual_price_rules_*.yaml"))
    return backups[-1] if backups else None


def restore_latest_backup() -> bool:
    backup = latest_backup()
    if not backup:
        return False
    shutil.copy2(backup, RULES_PATH)
    return True


def _match_mask(df: pd.DataFrame, filters: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        if key not in df.columns or value in (None, ""):
            continue
        if isinstance(value, list):
            mask &= df[key].isin(value)
        else:
            mask &= df[key].astype(str) == str(value)
    return mask


def preview_rule(df: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    mask = _match_mask(df, rule.get("filters", {}))
    cols = ["run_id", "developer_key", "project_name", "house_name", "flat_id", "price_rub", "price_m2_effective", "price_source"]
    existing = [c for c in cols if c in df.columns]
    return df.loc[mask, existing].head(20)


def apply_rules(df: pd.DataFrame, rules_data: dict[str, Any] | None = None) -> RuleApplicationResult:
    rules_data = rules_data or load_rules()
    rules = rules_data.get("rules", [])
    result = df.copy()
    audits: list[dict[str, Any]] = []

    for rule in rules:
        if not rule.get("enabled", True):
            continue
        rule_id = str(rule.get("id", "rule_without_id"))
        filters = rule.get("filters", {})
        mask = _match_mask(result, filters)
        matched_rows = int(mask.sum())
        applied_rows = 0

        if matched_rows:
            if rule.get("set_price_rub") is not None:
                result.loc[mask, "price_rub"] = float(rule["set_price_rub"])
                applied_rows = matched_rows
            if rule.get("set_price_m2") is not None:
                result.loc[mask, "price_m2_effective"] = float(rule["set_price_m2"])
                applied_rows = matched_rows
            if rule.get("multiply_price_rub") is not None:
                result.loc[mask, "price_rub"] = result.loc[mask, "price_rub"] * float(rule["multiply_price_rub"])
                applied_rows = matched_rows

            if rule.get("set_price_m2") is None:
                result.loc[mask, "price_m2_effective"] = result.loc[mask, "price_rub"] / result.loc[mask, "area_m2"]

            result.loc[mask, "price_source"] = "manual"
            result.loc[mask, "manual_rule_id"] = rule_id
            result.loc[mask, "price_kind"] = rule.get("price_kind", "manual")
            result.loc[mask, "price_note"] = rule.get("note", "Ручная корректировка")

        for run_id, run_count in result.loc[mask].groupby("run_id").size().items():
            audits.append(
                {
                    "run_id": run_id,
                    "rule_id": rule_id,
                    "matched_rows": int(run_count),
                    "applied_rows": int(run_count if applied_rows else 0),
                }
            )

    audit_df = pd.DataFrame(audits, columns=AUDIT_COLUMNS)
    if audit_df.empty:
        audit_df = pd.DataFrame(columns=AUDIT_COLUMNS)
    return RuleApplicationResult(df=result, audit=audit_df)
