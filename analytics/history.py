from __future__ import annotations

import pandas as pd


def build_snapshot_index(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=["run_id", "snapshot_at"])
    idx = (
        history_df.groupby("run_id", as_index=False)["captured_at"]
        .min()
        .rename(columns={"captured_at": "snapshot_at"})
        .sort_values("snapshot_at")
    )
    return idx


def build_project_snapshot_metrics(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "snapshot_at",
                "developer_key",
                "project_name",
                "total_units",
                "sold_units",
                "available_units",
                "sold_rate",
                "price_m2_median_available",
                "price_coverage_share",
                "manual_price_share",
            ]
        )

    work = history_df.copy()
    work["is_sold"] = work["status_bucket"].eq("sold")
    work["is_available"] = work["status_bucket"].eq("available")
    work["has_price"] = work["price_m2_effective"].notna()
    work["is_manual"] = work["price_source"].eq("manual")

    group_cols = ["run_id", "captured_at", "developer_key", "project_name"]
    grouped = work.groupby(group_cols, as_index=False)

    metrics = grouped.agg(
        total_units=("flat_id", "count"),
        sold_units=("is_sold", "sum"),
        available_units=("is_available", "sum"),
        priced_units=("has_price", "sum"),
        manual_units=("is_manual", "sum"),
    )
    metrics["sold_rate"] = (metrics["sold_units"] / metrics["total_units"]).fillna(0.0)
    metrics["price_coverage_share"] = (metrics["priced_units"] / metrics["total_units"]).fillna(0.0)
    metrics["manual_price_share"] = (metrics["manual_units"] / metrics["total_units"]).fillna(0.0)

    avail_prices = (
        work[work["is_available"]]
        .groupby(group_cols)["price_m2_effective"]
        .median()
        .rename("price_m2_median_available")
        .reset_index()
    )

    metrics = metrics.merge(avail_prices, on=group_cols, how="left")
    metrics = metrics.rename(columns={"captured_at": "snapshot_at"})
    metrics = metrics[
        [
            "run_id",
            "snapshot_at",
            "developer_key",
            "project_name",
            "total_units",
            "sold_units",
            "available_units",
            "sold_rate",
            "price_m2_median_available",
            "price_coverage_share",
            "manual_price_share",
        ]
    ]
    return metrics.sort_values(["snapshot_at", "developer_key", "project_name"])


def build_sales_events(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(
            columns=[
                "sold_at",
                "run_id",
                "developer_key",
                "project_name",
                "house_name",
                "flat_id",
                "rooms",
                "area_m2",
                "floor",
                "sold_price_rub",
                "sold_price_m2",
                "last_listed_price_rub",
                "last_listed_price_m2",
                "sold_price_source",
            ]
        )

    sort_cols = ["developer_key", "flat_id", "captured_at"]
    work = history_df.sort_values(sort_cols).copy()

    group_keys = ["developer_key", "flat_id"]
    work["prev_status"] = work.groupby(group_keys)["status_bucket"].shift(1)
    work["prev_price_rub"] = work.groupby(group_keys)["price_rub"].shift(1)
    work["prev_price_m2_effective"] = work.groupby(group_keys)["price_m2_effective"].shift(1)
    work["prev_price_source"] = work.groupby(group_keys)["price_source"].shift(1)

    sold_transition = work["status_bucket"].eq("sold") & work["prev_status"].ne("sold")
    sold = work[sold_transition].copy()

    sold["sold_price_rub"] = sold["price_rub"].where(sold["price_rub"].notna(), sold["prev_price_rub"])
    sold["sold_price_m2"] = sold["price_m2_effective"].where(
        sold["price_m2_effective"].notna(), sold["prev_price_m2_effective"]
    )
    sold["last_listed_price_rub"] = sold["prev_price_rub"]
    sold["last_listed_price_m2"] = sold["prev_price_m2_effective"]
    sold["sold_price_source"] = sold["price_source"].where(sold["price_rub"].notna(), sold["prev_price_source"])

    sold_events = sold.rename(columns={"captured_at": "sold_at"})[
        [
            "sold_at",
            "run_id",
            "developer_key",
            "project_name",
            "house_name",
            "flat_id",
            "rooms",
            "area_m2",
            "floor",
            "sold_price_rub",
            "sold_price_m2",
            "last_listed_price_rub",
            "last_listed_price_m2",
            "sold_price_source",
        ]
    ]
    return sold_events.sort_values("sold_at")
