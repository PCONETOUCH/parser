# Analytics dashboard

Легковесная витрина и Streamlit-дашборд для мониторинга конкурентов.

## Запуск

```bash
pip install -r analytics/requirements.txt
python -m analytics.build_mart --all
streamlit run analytics/app.py
```

## Источник данных

Снапшоты читаются **из всех папок run_id** по пути:

`launcher/data/published_snapshots/<run_dir>/*.csv`

## Выходные файлы

Генерируются в `analytics/data/`:

- `lots_history.csv` — нормализованная история лотов по всем срезам + ручные цены.
- `snapshot_index.csv` — индекс run_id -> snapshot_at.
- `project_snapshot_metrics.csv` — метрики по проектам и срезам.
- `sales_events.csv` — события продаж (цена продажи = последняя листинговая цена).
- `manual_pricing_audit.csv` — аудит применения ручных правил по run_id.

## Ручные цены

- Правила лежат в `analytics/manual_price_rules.yaml`.
- В UI есть Preview перед Apply.
- Перед применением создается backup; кнопка `⟲` в левом верхнем углу возвращает последний backup и пересобирает витрину.
