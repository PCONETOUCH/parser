# Launcher

Назначение: запуск парсеров и публикация «зелёных» snapshots в общий пул.

## Запуск

Windows:

- `launcher\run_launcher.bat` — запускает launcher по `launcher_config.json`
- `launcher\run_manual_accept.bat --run_id 20260210_120501 --developer_key talan --reason "проверили вручную, данные корректны"` — вручную принять файл из quarantine в published

Linux/macOS (опционально):

- `launcher/run_launcher.sh`
- `launcher/run_manual_accept.sh ...`

## Что делает launcher

1) Генерирует `RUN_ID` вида `YYYYMMDD_HHMMSS`.
2) Запускает парсеры по списку в `launcher_config.json` (последовательно или параллельно по `max_parallel`).
3) После каждого парсера читает `result.json` (если есть) и на основании `status`:
   - `OK` → копирует snapshot в `launcher/data/published_snapshots/<RUN_ID>/`
   - `WARN/FAIL` → копирует snapshot в `launcher/data/quarantine/<RUN_ID>/`
4) Формирует отчёт в `launcher/data/reports/<RUN_ID>/summary.json` и `summary.txt`.
5) Для `WARN/FAIL` автоматически собирает **support bundle**:
   `launcher/data/reports/<RUN_ID>/support_bundles/<developer_key>__<RUN_ID>__...zip`

## Полное логирование

- `launcher/logs/<RUN_ID>/launcher.log` — общий лог launchera
- `launcher/logs/<RUN_ID>/parsers/<developer_key>.stdout.log` — stdout парсера
- `launcher/logs/<RUN_ID>/parsers/<developer_key>.stderr.log` — stderr парсера
- `launcher/logs/<RUN_ID>/parsers/<developer_key>.meta.json` — команда/время/exit code

Парсер **сам** пишет свой полный лог и `errors/errors.csv` в `parser/<developer_key>/output_<developer_key>/`.

## Ручное принятие (manual accept)

Когда snapshot ушёл в карантин, можно принять его вручную:
- файл копируется в `published_snapshots/<RUN_ID>/` (исходник в quarantine не удаляется)
- записывается запись в `launcher/data/reports/<RUN_ID>/manual_accept.jsonl`
- обновляется `launcher/data/published_state.json` (используется как база для сравнения в будущих запусках)

## Окна с прогрессом (Windows)

В `launcher_config.json` можно включить открытие отдельного окна консоли на каждый парсер, чтобы видеть прогресс в реальном времени.

Пример:
```json
"console": {
  "enabled": true,
  "show_window": true
}
```

Также можно переопределить на уровне конкретного парсера:
```json
"runner": {
  "type": "python",
  "script": "parser/xxx/xxx_parser.py",
  "show_window": true
}
```

В режиме `show_window=true` stdout/stderr парсера показываются в отдельном окне и одновременно пишутся в `launcher/logs/<RUN_ID>/parsers/<dev>.stdout.log` (через PowerShell Tee-Object).
