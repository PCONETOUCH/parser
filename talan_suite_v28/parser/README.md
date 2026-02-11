# Talan chess parser — v8 (adaptive retry)

## Что добавлено
- После pass1 при наличии fail — спрашивает подтверждение retry.
- Retry параметры адаптивные:
  - если в ошибках есть http_429/http_403/html_instead_json → автоматом снижает параллельность и увеличивает паузы
  - каждый следующий retry-раунд ещё мягче
- `--retry-only errors/<file>.csv` — дособрать цены без прохода шахматок.

## Установка
```bash
pip install -r requirements.txt
```

## Запуск
Полный:
```bash
python talan_parser.py
```

Retry-only:
```bash
python talan_parser.py --retry-only errors/20260126_153012_api_errors_pass1.csv
```

## PyCharm
Для input() и цветов:
Run → Edit Configurations… → ✅ Emulate terminal in output console
или запускай через Terminal.
