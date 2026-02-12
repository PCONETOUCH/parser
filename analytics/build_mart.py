from __future__ import annotations

import argparse
from pathlib import Path

from analytics.history import build_project_snapshot_metrics, build_sales_events, build_snapshot_index
from analytics.manual_pricing import apply_rules, load_rules
from analytics.normalize import ensure_data_dir, load_all_snapshots


def build_all() -> dict[str, Path]:
    data_dir = ensure_data_dir()

    raw_history = load_all_snapshots()
    rules_data = load_rules()
    applied = apply_rules(raw_history, rules_data)
    history_df = applied.df

    outputs = {
        "lots_history": data_dir / "lots_history.csv",
        "snapshot_index": data_dir / "snapshot_index.csv",
        "project_snapshot_metrics": data_dir / "project_snapshot_metrics.csv",
        "sales_events": data_dir / "sales_events.csv",
        "manual_pricing_audit": data_dir / "manual_pricing_audit.csv",
    }

    history_df.to_csv(outputs["lots_history"], index=False)
    build_snapshot_index(history_df).to_csv(outputs["snapshot_index"], index=False)
    build_project_snapshot_metrics(history_df).to_csv(outputs["project_snapshot_metrics"], index=False)
    build_sales_events(history_df).to_csv(outputs["sales_events"], index=False)
    applied.audit.to_csv(outputs["manual_pricing_audit"], index=False)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build analytics mart")
    parser.add_argument("--all", action="store_true", help="Rebuild all outputs")
    args = parser.parse_args()

    if args.all:
        outputs = build_all()
        for name, path in outputs.items():
            print(f"{name}: {path}")
    else:
        parser.error("Use --all to run full rebuild")


if __name__ == "__main__":
    main()
