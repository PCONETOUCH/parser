# UI Local (FastAPI)

## Запуск
- Linux/macOS: `bash scripts/run_ui.sh`
- Windows: `scripts\\run_ui.bat`
- URL: `http://127.0.0.1:8787`

## Пути к данным
- По умолчанию: `talan_suite_v28/launcher/data`
- Переопределение:
  - env: `LAUNCHER_DATA_ROOT`
  - `config/ui_config.json` -> `launcher_data_root`

## Manual accept audit
- Аудит пишется в: `launcher/data/reports/<RUN_ID>/manual_accept.jsonl`.
- API: `POST /api/manual_accept` (reason обязателен).

## Registry
- Файл справочника: `config/project_registry.csv` (MVP).
- Обновляйте строки для нормализации паспорта проектов.

## Метрики
- Data Trust: coverage + unknown project share.
- Интервалы: `prev_close+1..curr_close` (устранение ложных понедельничных всплесков).
- Sales proxy: `max(removed, sold_status)` и `sales/day`.
