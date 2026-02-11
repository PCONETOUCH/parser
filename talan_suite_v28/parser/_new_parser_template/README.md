# Новый парсер — шаблон (копируй-переименуй)

## Как использовать
1) Скопируй папку:
   `parser/_new_parser_template/` → `parser/<developer_key>/`

2) Переименуй файлы внутри папки (строго так):
- `developer_key_parser.py` → `<developer_key>_parser.py`
- `developer_key_config.json` → `<developer_key>_config.json`
- `run_developer_key_parser_auto.bat` → `run_<developer_key>_parser_auto.bat`
- `run_developer_key_parser_auto.sh` → `run_<developer_key>_parser_auto.sh`
- `output_developer_key/` → `output_<developer_key>/`

3) В `<developer_key>_config.json`:
- `developer_key`: ключ девелопера (латиница, без пробелов)
- `source`: базовый URL сайта/источника
- `output.base_dir`: поставь `parser/<developer_key>/output_<developer_key>`
- `demo_mode`: поставь `false`, когда подключишь реальный сбор

4) Проверь локальный запуск:
- Windows: `run_<developer_key>_parser_auto.bat`
- Linux/macOS: `run_<developer_key>_parser_auto.sh`

## Контракт с launcher (обязательный)
Парсер **всегда** пишет:
- `output_<developer_key>/snapshots_raw/<developer_key>__<city>__<YYYYMMDD_HHMMSS>.csv`
- `output_<developer_key>/result.json`
- (если WARN/FAIL) `output_<developer_key>/reason.json`

CSV: 1 строка = 1 лот. Обязательные поля в каждой строке:
- `developer_key`
- `captured_at`
- `source`

## Статусы
- `OK` — можно публиковать
- `WARN` — данные собраны, но есть подозрительные изменения (лучше карантин/ручная проверка)
- `FAIL` — данных нет или сломана сессия/авторизация
