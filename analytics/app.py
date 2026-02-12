from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.build_mart import build_all
from analytics.manual_pricing import backup_rules, load_rules, preview_rule, restore_latest_backup, save_rules
from analytics.normalize import DATA_DIR

st.set_page_config(page_title="Мониторинг конкурентов", page_icon="📊", layout="wide")


@st.cache_data(ttl=120)
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=[c for c in ["captured_at", "snapshot_at", "sold_at"] if c in pd.read_csv(path, nrows=0).columns])


def refresh_data() -> None:
    load_csv.clear()


def fmt_int(value: float) -> str:
    return f"{int(value):,}".replace(",", " ") if pd.notna(value) else "—"


def smart_insights(metrics: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    if metrics.empty:
        return ["Недостаточно данных для аналитических выводов."]

    latest_ts = metrics["snapshot_at"].max()
    latest = metrics[metrics["snapshot_at"] == latest_ts].copy()

    top = latest.sort_values("sold_rate", ascending=False).head(3)
    if not top.empty:
        txt = ", ".join([f"{r.project_name} ({r.sold_rate:.1%})" for r in top.itertuples()])
        insights.append(f"Лидеры по доле проданных лотов на последнем срезе: {txt}.")

    recent = metrics.sort_values("snapshot_at").groupby("project_name").tail(2)
    if recent.groupby("project_name").size().max() >= 2:
        pivot = recent.pivot_table(index="project_name", columns="snapshot_at", values="sold_rate")
        if pivot.shape[1] >= 2:
            delta = (pivot.iloc[:, -1] - pivot.iloc[:, -2]).sort_values(ascending=False)
            best = delta.head(1)
            if not best.empty:
                insights.append(f"Максимальный прирост sold_rate: {best.index[0]} (+{best.iloc[0]:.1%} к прошлому срезу).")

    wash = metrics.sort_values("snapshot_at").groupby("project_name").tail(2)
    if wash.groupby("project_name").size().max() >= 2:
        w_pivot = wash.pivot_table(index="project_name", columns="snapshot_at", values="available_units")
        if w_pivot.shape[1] >= 2:
            wash_delta = (w_pivot.iloc[:, -1] - w_pivot.iloc[:, -2]).sort_values()
            fastest = wash_delta.head(1)
            if not fastest.empty:
                insights.append(
                    f"Самое быстрое вымывание предложения: {fastest.index[0]} ({int(fastest.iloc[0])} шт. за последний интервал)."
                )

    low_coverage = latest[latest["price_coverage_share"] < 0.6]
    if not low_coverage.empty:
        projects = ", ".join(low_coverage["project_name"].head(4).tolist())
        insights.append(f"Внимание: низкое покрытие цен (<60%) у проектов: {projects}. Метрики цены могут быть неполными.")

    manual = latest[latest["manual_price_share"] > 0]
    if not manual.empty:
        insights.append("Для части проектов используются ручные цены; проверяйте вкладку 'Ручные цены' и аудит.")

    return insights


def header_actions() -> None:
    left, mid, right = st.columns([1, 6, 1])
    with left:
        if st.button("⟲", help="Отменить последнее изменение правил ручных цен"):
            if restore_latest_backup():
                build_all()
                refresh_data()
                st.success("Последние изменения правил отменены, витрина пересобрана.")
            else:
                st.warning("Нет бэкапа для отката.")
    with right:
        if st.button("Rebuild all", type="primary"):
            build_all()
            refresh_data()
            st.success("Витрина пересобрана.")


def page_overview(metrics: pd.DataFrame) -> None:
    st.subheader("Обзор динамики")
    if metrics.empty:
        st.info("Нет данных. Нажмите Rebuild all после появления snapshot CSV.")
        return

    min_date = metrics["snapshot_at"].min().date()
    max_date = metrics["snapshot_at"].max().date()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Проектов", metrics["project_name"].nunique())
    col2.metric("Срезов", metrics["run_id"].nunique())
    col3.metric("Продано (последний срез)", fmt_int(metrics.sort_values("snapshot_at").groupby("project_name").tail(1)["sold_units"].sum()))
    col4.metric("Доступно (последний срез)", fmt_int(metrics.sort_values("snapshot_at").groupby("project_name").tail(1)["available_units"].sum()))

    f1, f2, f3 = st.columns(3)
    dev = f1.multiselect("Девелопер", sorted(metrics["developer_key"].dropna().unique()))
    proj = f2.multiselect("Проект", sorted(metrics["project_name"].dropna().unique()))
    date_range = f3.date_input("Период", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    filtered = metrics.copy()
    if dev:
        filtered = filtered[filtered["developer_key"].isin(dev)]
    if proj:
        filtered = filtered[filtered["project_name"].isin(proj)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        filtered = filtered[(filtered["snapshot_at"] >= start) & (filtered["snapshot_at"] < end)]

    for field, title, yfmt in [
        ("sold_rate", "Sold rate по проектам", ".0%"),
        ("available_units", "Доступные лоты", None),
        ("sold_units", "Проданные лоты", None),
    ]:
        fig = px.line(filtered, x="snapshot_at", y=field, color="project_name", markers=True, title=title)
        if yfmt:
            fig.update_layout(yaxis_tickformat=yfmt)
        st.plotly_chart(fig, use_container_width=True)

    priced = filtered[filtered["price_coverage_share"] >= 0.4]
    if not priced.empty:
        fig = px.line(
            priced,
            x="snapshot_at",
            y="price_m2_median_available",
            color="project_name",
            markers=True,
            title="Медианная цена за м² (доступные лоты)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Недостаточное покрытие цен для надежного графика медианной цены за м².")

    st.markdown("### Smart insights")
    for text in smart_insights(filtered):
        st.write(f"• {text}")


def page_sales(events: pd.DataFrame) -> None:
    st.subheader("Продажи")
    if events.empty:
        st.info("События продаж появятся, когда будет минимум 2 среза со сменой статуса на sold.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Событий продаж", len(events))
    c2.metric("Проектов с продажами", events["project_name"].nunique())
    c3.metric("Сумма продаж, ₽", fmt_int(events["sold_price_rub"].fillna(0).sum()))

    table = events.copy().sort_values("sold_at", ascending=False)
    st.dataframe(table, use_container_width=True)

    daily = events.set_index("sold_at").resample("D").size().rename("sales_cnt").reset_index()
    weekly = events.set_index("sold_at").resample("W").size().rename("sales_cnt").reset_index()

    st.plotly_chart(px.bar(daily, x="sold_at", y="sales_cnt", title="Продажи по дням"), use_container_width=True)
    st.plotly_chart(px.bar(weekly, x="sold_at", y="sales_cnt", title="Продажи по неделям"), use_container_width=True)

    speed = events.groupby("project_name").size().sort_values(ascending=False).reset_index(name="sales_cnt")
    st.plotly_chart(px.bar(speed.head(10), x="project_name", y="sales_cnt", title="Топ проектов по скорости продаж"), use_container_width=True)


def page_manual_prices(history: pd.DataFrame, audit: pd.DataFrame) -> None:
    st.subheader("Ручные цены")
    rules_data = load_rules()

    with st.expander("Текущие правила", expanded=True):
        st.json(rules_data)

    with st.form("new_rule"):
        st.markdown("#### Добавить правило")
        rule_id = st.text_input("ID правила", value=f"rule_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}")
        developer = st.text_input("developer_key (опционально)")
        project = st.text_input("project_name (опционально)")
        set_price_m2 = st.number_input("set_price_m2", min_value=0.0, step=1000.0)
        note = st.text_input("Комментарий", value="Ручная корректировка")
        submitted_preview = st.form_submit_button("Preview")
        submitted_apply = st.form_submit_button("Apply")

        draft_rule = {
            "id": rule_id,
            "enabled": True,
            "filters": {"developer_key": developer, "project_name": project},
            "set_price_m2": set_price_m2 if set_price_m2 > 0 else None,
            "note": note,
        }

        if submitted_preview:
            prev = preview_rule(history, draft_rule)
            st.write(f"Совпало строк: {len(prev)}")
            st.dataframe(prev, use_container_width=True)

        if submitted_apply:
            backup_rules()
            rules_data.setdefault("rules", []).append(draft_rule)
            save_rules(rules_data)
            build_all()
            refresh_data()
            st.success("Правило сохранено и применено. Витрина пересобрана.")

    st.markdown("#### Аудит применения правил")
    st.dataframe(audit.sort_values(["run_id", "rule_id"]) if not audit.empty else audit, use_container_width=True)


def page_differences(history: pd.DataFrame) -> None:
    st.subheader("Различия и покрытие цен")
    if history.empty:
        st.info("Нет данных")
        return
    coverage = history.groupby("price_source").size().reset_index(name="rows")
    st.plotly_chart(px.pie(coverage, names="price_source", values="rows", title="Источник цен"), use_container_width=True)

    manual = history[history["price_source"] == "manual"]
    if manual.empty:
        st.caption("Ручных цен пока нет.")
    else:
        st.dataframe(
            manual[["run_id", "project_name", "flat_id", "price_rub", "price_m2_effective", "manual_rule_id", "price_note"]].head(200),
            use_container_width=True,
        )


def main() -> None:
    st.title("Аналитика мониторинга конкурентов")
    header_actions()

    history = load_csv("lots_history.csv")
    metrics = load_csv("project_snapshot_metrics.csv")
    sales = load_csv("sales_events.csv")
    audit = load_csv("manual_pricing_audit.csv")

    tabs = st.tabs(["Overview", "Sales", "Manual prices", "Differences"])
    with tabs[0]:
        page_overview(metrics)
    with tabs[1]:
        page_sales(sales)
    with tabs[2]:
        page_manual_prices(history, audit)
    with tabs[3]:
        page_differences(history)

    st.caption("Примечание: часть цен может быть рассчитана или задана вручную. Смотрите покрытие цен и аудит.")


if __name__ == "__main__":
    main()
