from __future__ import annotations


def build_insights(kpi: dict, sales_points: list[dict]) -> list[dict]:
    insights = []
    if sales_points:
        last = sales_points[-1]
        if last.get("sales", 0) == 0 and kpi.get("active_inventory", 0) > 30:
            insights.append({"severity": "WARN", "title": "Ноль продаж при заметном остатке", "cta": "Открыть в Паритете"})
    if kpi.get("null_price_effective_pct", 0) > 0.3:
        insights.append({"severity": "INFO", "title": "Высокая доля эффективных null price", "cta": "Открыть проект"})
    return insights[:15]
