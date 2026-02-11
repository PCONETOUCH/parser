#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Talan chess parser (Khabarovsk) — v8

Что добавлено (адаптивный retry):
- После pass1, если есть fail, спрашиваем подтверждение на retry.
- Параметры retry теперь АДАПТИВНЫЕ:
  - если в ошибках есть признаки антибота/лимитов (http_429/http_403/html_instead_json),
    retry автоматически снижает параллельность и повышает паузы.
  - каждый следующий retry-раунд становится мягче (ещё меньше параллельность, больше паузы).
- Сохраняются errors-файлы после каждого раунда.
- --retry-only errors/<file>.csv остаётся (один проход; если останутся ошибки — сохраняет retry_errors.csv).

PyCharm: чтобы работал input() и ANSI-цвета:
  Run → Edit Configurations… → ✅ Emulate terminal in output console
  или запускать через Terminal.

Запуск:
  pip install -r requirements.txt
  python talan_parser.py
  python talan_parser.py --retry-only errors/20260126_153012_api_errors_pass1.csv
'''
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import logging
import os
import re
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Dict, List, Optional, Tuple, Iterable, Any

import requests
from bs4 import BeautifulSoup
from bs4.exceptions import FeatureNotFound

from colorama import init as colorama_init, Fore, Style

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

colorama_init()

BASE = "https://talan.ru"
CITY = "khabarovsk"

COMPLEX_SLUGS = [
    "voronezhskaja",
    "dom-u-ozera",
    "klubnyj-dom-na-kalinina",
    "biografija",
    "serdce-vostoka",
    "amurskij-kvartal-2",
]

# --- Tuning knobs ---
HTTP_TIMEOUT = 25
RETRIES = 3
SLEEP_BETWEEN_PAGE_REQUESTS_SEC = 0.3

# Auto-retry policy (per requirements): 3 automatic retries with ~9s delay, then manual confirmation.
AUTO_RETRY_MAX_ROUNDS = 3
AUTO_RETRY_SLEEP_SEC = 9
AUTO_RETRY_JITTER_SEC = 2.0

# After auto-retries, we switch to the softest (manual) mode.
MANUAL_SOFT_WORKERS = 1
MANUAL_SOFT_IN_FLIGHT = 12
MANUAL_SOFT_SUBMIT_SLEEP_SEC = 1.0

# API parallelism (pass1)
MAX_WORKERS = 8
MAX_IN_FLIGHT = 160
WAIT_TIMEOUT_SEC = 60

# Retry base defaults (adaptive logic will override in interactive retry-loop)
RETRY_BASE_WORKERS = 4
RETRY_BASE_IN_FLIGHT = 60
RETRY_BASE_SUBMIT_SLEEP_SEC = 0.25

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

OK_COLOR = Fore.GREEN
ERR_COLOR = Fore.RED
RESET = Style.RESET_ALL


# ----------------------------- utils: colors + logging -----------------------------

class ColorFormatter(logging.Formatter):
    LEVEL_TO_COLOR = {
        "INFO": Fore.GREEN,
        "WARNING": Fore.RED,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED,
        "DEBUG": Style.DIM,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_TO_COLOR.get(record.levelname, "")
        msg = super().format(record)
        return f"{color}{msg}{RESET}" if color else msg


def setup_logging(ts: str) -> logging.Logger:
    os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
    log_path = os.path.join(os.path.dirname(__file__), "logs", f"run_{ts}.log")

    logger = logging.getLogger("talan_parser")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    plain_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    color_fmt = ColorFormatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(plain_fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(color_fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Logging to %s", log_path)
    return logger


def ask_yes_no(prompt: str, default_no: bool = True) -> bool:
    suffix = " [y/N]: " if default_no else " [Y/n]: "
    while True:
        try:
            ans = input(prompt + suffix).strip().lower()
        except EOFError:
            return False if default_no else True

        if not ans:
            return False if default_no else True
        if ans in ("y", "yes", "д", "да"):
            return True
        if ans in ("n", "no", "н", "нет"):
            return False
        print("Введите y/n (да/нет).")


def t(iterable: Iterable[Any], desc: str):
    if tqdm is None:
        return iterable
    bar_format = f"{OK_COLOR}{{l_bar}}{{bar}}{{r_bar}}{RESET}"
    return tqdm(iterable, desc=desc, bar_format=bar_format)


def make_pbar(total: int, desc: str):
    if tqdm is None:
        return None
    bar_format = f"{OK_COLOR}{{l_bar}}{{bar}}{{r_bar}}{RESET}"
    return tqdm(total=total, desc=desc, bar_format=bar_format)


# ----------------------------- http (per-thread session) -----------------------------

_thread_local = threading.local()


def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    })
    return s


def _get_thread_session() -> requests.Session:
    sess = getattr(_thread_local, "sess", None)
    if sess is None:
        sess = _new_session()
        _thread_local.sess = sess
    return sess


def _get(url: str, params: Optional[dict] = None, logger: Optional[logging.Logger] = None) -> str:
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            sess = _get_thread_session()
            r = sess.get(url, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            if logger:
                logger.warning("GET fail (%s/%s) %s params=%s err=%s", attempt, RETRIES, url, params, e)
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"GET failed after {RETRIES} tries: {url} params={params} err={last_err}")


def classify_error(exc: Exception, http_status: Optional[int], body_prefix: str) -> str:
    if "Got HTML instead of JSON" in str(exc):
        return "html_instead_json"
    if http_status in (403,):
        return "http_403"
    if http_status in (429,):
        return "http_429"
    if http_status and 500 <= http_status <= 599:
        return "http_5xx"
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.ConnectionError):
        return "connection_error"
    if body_prefix.startswith("<"):
        return "html_instead_json"
    return "unknown"


def _get_json(url: str, params: Optional[dict] = None, logger: Optional[logging.Logger] = None) -> dict:
    last_err: Optional[Exception] = None
    last_status: Optional[int] = None
    last_body_prefix = ""

    for attempt in range(1, RETRIES + 1):
        try:
            sess = _get_thread_session()
            r = sess.get(url, params=params, timeout=HTTP_TIMEOUT, headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": f"{BASE}/{CITY}/",
            })
            last_status = r.status_code
            txt = r.text or ""
            last_body_prefix = txt.lstrip()[:30]
            r.raise_for_status()
            if txt.lstrip().startswith("<"):
                raise ValueError("Got HTML instead of JSON (possible anti-bot).")
            return r.json()
        except Exception as e:
            last_err = e
            etype = classify_error(e, last_status, last_body_prefix)
            if logger:
                logger.warning(
                    "GET JSON fail (%s/%s) %s params=%s status=%s type=%s err=%s",
                    attempt, RETRIES, url, params, last_status, etype, e
                )
            if etype in ("http_429", "http_403", "html_instead_json"):
                time.sleep(1.8 * attempt)
            else:
                time.sleep(0.9 * attempt)

    raise RuntimeError(f"GET JSON failed after {RETRIES} tries: {url} params={params} status={last_status} err={last_err}")


# ----------------------------- parsing chess -----------------------------

def make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(html, "html.parser")


@dataclasses.dataclass
class ChessApt:
    apartment_id: str
    complex_slug: str
    house: str
    entrance: str
    room_label: str
    status_hint: str
    has_percent_badge: bool
    title: str
    number_text: str
    floor: Optional[int]
    floor_count_hint: Optional[int]
    area_hint: Optional[float]
    apartment_url: str


def chess_url(slug: str) -> str:
    return f"{BASE}/{CITY}/apartment-complex/{slug}/chess"


def parse_button_values(html: str, title_ru: str) -> List[str]:
    soup = make_soup(html)
    for p in soup.select("p.chess__form-field-title"):
        if p.get_text(strip=True).lower() == title_ru.lower():
            block = p.find_parent(class_="chess__form-field")
            if not block:
                continue
            vals = []
            for b in block.select("button.chess__form-field-item"):
                txt = b.get_text(strip=True)
                if txt:
                    vals.append(txt)
            return vals
    return []


def parse_apartments_from_chess(html: str, complex_slug: str, house: str, entrance: str) -> List[ChessApt]:
    soup = make_soup(html)
    items: List[ChessApt] = []

    for cell in soup.select("div.complex-chess__table-cell"):
        if "complex-chess__table-cell--number" in (cell.get("class") or []):
            continue

        block = cell.select_one("div.complex-chess__table-cell-block")
        card = cell.select_one("div.complex-chess__table-cell-card")
        link = cell.select_one("a.complex-chess__cell-card-link")

        if not link or not link.get("href"):
            continue

        href = link["href"]
        m = re.search(r"/apartments/([0-9a-fA-F-]{36})", href)
        if not m:
            continue

        apt_id = m.group(1)
        apt_url = BASE + href

        room_label = block.get_text(" ", strip=True) if block else ""
        has_percent = bool(cell.select_one(".complex-chess__table-cell-block-percent"))

        status_hint = "free"
        if block:
            classes = block.get("class") or []
            if any("block--sold" in c for c in classes):
                status_hint = "sold"
            elif any("block--booked" in c for c in classes):
                status_hint = "booked"
            elif any("block--presale" in c for c in classes):
                status_hint = "presale"

        title = ""
        number_text = ""
        floor = None
        floor_count = None
        area = None

        if card:
            t_ = card.select_one(".complex-chess__cell-card-title")
            if t_:
                title = t_.get_text(strip=True)
                m2 = re.search(r"№\s*(\d+)", title)
                if m2:
                    number_text = m2.group(1)

            subs = [p.get_text(" ", strip=True) for p in card.select(".complex-chess__cell-card-subtitle")]
            for s in subs:
                fm = re.search(r"(\d+)\s*/\s*(\d+)", s)
                if fm:
                    floor = int(fm.group(1))
                    floor_count = int(fm.group(2))
                am = re.search(r"([\d.,]+)\s*м", s)
                if am:
                    area = float(am.group(1).replace(",", "."))

        items.append(ChessApt(
            apartment_id=apt_id,
            complex_slug=complex_slug,
            house=str(house),
            entrance=str(entrance),
            room_label=room_label,
            status_hint=status_hint,
            has_percent_badge=has_percent,
            title=title,
            number_text=number_text,
            floor=floor,
            floor_count_hint=floor_count,
            area_hint=area,
            apartment_url=apt_url,
        ))

    uniq = {it.apartment_id: it for it in items}
    return list(uniq.values())


# ----------------------------- api fetch (streaming) -----------------------------

ANTI_TYPES = {"http_429", "http_403", "html_instead_json"}


def error_stats(errors: List[Dict[str, Any]]) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for e in errors:
        t_ = e.get("error_type") or "unknown"
        stats[t_] = stats.get(t_, 0) + 1
    return stats


def choose_retry_params(errors: List[Dict[str, Any]], round_no: int) -> Tuple[int, int, float]:
    n = max(1, len(errors))
    stats = error_stats(errors)
    anti = sum(stats.get(t, 0) for t in ANTI_TYPES)
    anti_share = anti / n

    workers = RETRY_BASE_WORKERS
    in_flight = RETRY_BASE_IN_FLIGHT
    sleep_submit = RETRY_BASE_SUBMIT_SLEEP_SEC

    if anti > 0:
        workers = min(workers, 2)
        in_flight = min(in_flight, 25)
        sleep_submit = max(sleep_submit, 0.6)
        if anti_share >= 0.6:
            workers = min(workers, 1)
            in_flight = min(in_flight, 12)
            sleep_submit = max(sleep_submit, 0.9)

    soften = max(0, round_no - 1)
    workers = max(1, workers - soften)
    in_flight = max(10, in_flight - 10 * soften)
    sleep_submit = min(1.5, sleep_submit + 0.2 * soften)

    return workers, in_flight, sleep_submit


def fetch_apartment_info(apartment_id: str, logger: logging.Logger) -> Tuple[bool, Optional[dict], Dict[str, Any]]:
    url = f"{BASE}/api/apartment-info"
    try:
        data = _get_json(url, params={"apartmentId": apartment_id}, logger=logger)
        return True, data, {}
    except Exception as e:
        msg = str(e)
        status = None
        m = re.search(r"status=(\d+)", msg)
        if m:
            status = int(m.group(1))
        err_type = classify_error(e, status, "")
        return False, None, {
            "error_type": err_type,
            "http_status": status,
            "message": msg[:300],
        }


def fetch_api_stream(
    apartment_ids: List[str],
    logger: logging.Logger,
    desc: str,
    max_workers: int,
    max_in_flight: int,
    sleep_between_submit: float = 0.0,
) -> Tuple[Dict[str, dict], List[Dict[str, Any]]]:
    total = len(apartment_ids)
    info_ok: Dict[str, dict] = {}
    errors: List[Dict[str, Any]] = []
    pbar = make_pbar(total=total, desc=desc)

    err_count = 0

    def refresh_postfix():
        if pbar is None:
            return
        pbar.set_postfix({"errors": f"{ERR_COLOR}{err_count}{RESET}"})

    def submit_one(ex: ThreadPoolExecutor, aid: str, pending: Dict) -> None:
        fut = ex.submit(fetch_apartment_info, aid, logger)
        pending[fut] = aid

    it = iter(apartment_ids)
    pending: Dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        while len(pending) < min(max_in_flight, total):
            try:
                aid = next(it)
            except StopIteration:
                break
            submit_one(ex, aid, pending)
            if sleep_between_submit:
                time.sleep(sleep_between_submit)

        refresh_postfix()

        while pending:
            done_set, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED, timeout=WAIT_TIMEOUT_SEC)
            if not done_set:
                logger.warning("[API] No completions for %ds. pending=%d (network stall / anti-bot?)",
                               WAIT_TIMEOUT_SEC, len(pending))
                continue

            for fut in done_set:
                aid = pending.pop(fut)
                ok, data, emeta = fut.result()

                if ok and data is not None:
                    info_ok[aid] = data
                else:
                    err_count += 1
                    errors.append({
                        "apartment_id": aid,
                        "error_type": emeta.get("error_type", "unknown"),
                        "http_status": emeta.get("http_status"),
                        "message": emeta.get("message", ""),
                    })

                if pbar:
                    pbar.update(1)
                    refresh_postfix()

                try:
                    aid2 = next(it)
                    submit_one(ex, aid2, pending)
                    if sleep_between_submit:
                        time.sleep(sleep_between_submit)
                except StopIteration:
                    pass

    if pbar:
        pbar.close()

    return info_ok, errors


# ----------------------------- snapshot building -----------------------------

def snapshot_row(now_iso: str, chess: ChessApt, info: Optional[dict], api_ok: bool, api_err: Optional[dict]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "captured_at": now_iso,
        "city": CITY,
        "complex_slug": chess.complex_slug,
        "house": chess.house,
        "entrance": chess.entrance,
        "apartment_id": chess.apartment_id,
        "apartment_url": chess.apartment_url,
        "room_label_chess": chess.room_label,
        "status_chess_hint": chess.status_hint,
        "has_percent_badge": chess.has_percent_badge,
        "title_chess": chess.title,
        "number_chess": chess.number_text,
        "floor_chess": chess.floor,
        "floor_count_chess": chess.floor_count_hint,
        "area_chess": chess.area_hint,
        "api_ok": 1 if api_ok else 0,
        "api_error_type": (api_err.get("error_type") if api_err else None),
        "api_http_status": (api_err.get("http_status") if api_err else None),
        "api_error_message": (api_err.get("message") if api_err else None),
    }

    if not info:
        return row

    price = info.get("price")
    promo = info.get("promotion_price")
    hyp = info.get("hypothec_price")
    area = info.get("area")
    cps = info.get("cost_per_square")

    discount_abs = None
    discount_pct = None
    try:
        if promo is not None and price is not None:
            discount_abs = float(price) - float(promo)
            if float(price) > 0:
                discount_pct = (discount_abs / float(price)) * 100.0
    except Exception:
        pass

    row.update({
        "status": info.get("status"),
        "price": price,
        "promotion_price": promo,
        "hypothec_price": hyp,
        "discount_abs": discount_abs,
        "discount_pct": discount_pct,
        "not_final_price": info.get("not_final_price"),
        "cost_per_square": cps,
        "area": area,
        "price_per_sqm_calc": (float(price) / float(area)) if (price and area) else None,
        "rooms": info.get("rooms"),
        "str_rooms": info.get("str_rooms"),
        "studio": info.get("studio"),
        "floor": info.get("floor"),
        "floor_count": info.get("floor_count"),
        "section": info.get("section"),
        "block": info.get("block"),
        "building_number": info.get("building_number"),
        "entrance_number": info.get("entrance_number"),
        "queue_number": info.get("queue_number"),
        "release_date": info.get("release_date"),
        "date_acceptance": info.get("date_acceptance"),
        "finishing_type": info.get("finishing_type"),
        "layoutCode": info.get("layoutCode"),
        "layoutBlock": info.get("layoutBlock"),
        "updated_at_unix": info.get("updated_at"),
    })
    return row


def read_csv_as_dict(path: str, key: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row[key]] = row
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def compute_changes(prev: Dict[str, dict], cur: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    changes = []
    for row in cur:
        aid = str(row.get("apartment_id"))
        p = prev.get(aid)
        if not p:
            changes.append({"apartment_id": aid, "change_type": "new", **row})
            continue

        def _norm(x):
            return "" if x is None else str(x)

        price_changed = _norm(p.get("price")) != _norm(row.get("price"))
        promo_changed = _norm(p.get("promotion_price")) != _norm(row.get("promotion_price"))
        status_changed = _norm(p.get("status")) != _norm(row.get("status"))

        if price_changed or promo_changed or status_changed:
            changes.append({
                "apartment_id": aid,
                "change_type": "updated",
                "prev_price": p.get("price"),
                "prev_promotion_price": p.get("promotion_price"),
                "prev_status": p.get("status"),
                **row
            })

    cur_ids = {str(r.get("apartment_id")) for r in cur}
    for aid, prow in prev.items():
        if aid not in cur_ids:
            changes.append({"apartment_id": aid, "change_type": "missing", **prow})
    return changes


# ----------------------------- errors IO + retry-only patch -----------------------------

def write_errors_csv(path: str, errors: List[Dict[str, Any]], apt_meta: Optional[Dict[str, ChessApt]] = None) -> None:
    rows = []
    for e in errors:
        aid = e["apartment_id"]
        meta = apt_meta.get(aid) if apt_meta else None
        rows.append({
            "apartment_id": aid,
            "complex_slug": meta.complex_slug if meta else None,
            "house": meta.house if meta else None,
            "entrance": meta.entrance if meta else None,
            "error_type": e.get("error_type"),
            "http_status": e.get("http_status"),
            "message": e.get("message"),
        })
    write_csv(path, rows)


def read_apartment_ids_from_errors_csv(path: str) -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if "apartment_id" not in (r.fieldnames or []):
            raise ValueError("errors csv должен содержать колонку apartment_id")
        for row in r:
            aid = (row.get("apartment_id") or "").strip()
            if aid:
                ids.append(aid)
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def patch_snapshot_csv(snapshot_path: str, info_ok: Dict[str, dict], logger: logging.Logger, out_path: str) -> None:
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(snapshot_path)

    with open(snapshot_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise ValueError("Пустой snapshot csv, нечего патчить.")

    patched = 0
    for row in rows:
        aid = row.get("apartment_id")
        if not aid or aid not in info_ok:
            continue
        info = info_ok[aid]
        row["api_ok"] = "1"
        row["api_error_type"] = ""
        row["api_http_status"] = ""
        row["api_error_message"] = ""
        row["status"] = str(info.get("status") or "")
        row["price"] = str(info.get("price") or "")
        row["promotion_price"] = str(info.get("promotion_price") or "")
        row["hypothec_price"] = str(info.get("hypothec_price") or "")
        row["cost_per_square"] = str(info.get("cost_per_square") or "")
        row["area"] = str(info.get("area") or "")
        row["rooms"] = str(info.get("rooms") or "")
        row["str_rooms"] = str(info.get("str_rooms") or "")
        row["studio"] = str(info.get("studio") or "")
        row["floor"] = str(info.get("floor") or "")
        row["floor_count"] = str(info.get("floor_count") or "")
        row["finishing_type"] = str(info.get("finishing_type") or "")
        row["updated_at_unix"] = str(info.get("updated_at") or "")
        patched += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    logger.info("Patched snapshot copy: %s (patched_rows=%d)", out_path, patched)


# ----------------------------- main modes -----------------------------

def run_full(logger: logging.Logger) -> Tuple[str, str, Optional[str], str]:
    now = dt.datetime.now(dt.timezone.utc).astimezone().replace(microsecond=0)
    ts = now.strftime("%Y%m%d_%H%M%S")
    now_iso = now.isoformat()

    out_dir = os.path.join(os.path.dirname(__file__), "snapshots")
    changes_dir = os.path.join(os.path.dirname(__file__), "changes")
    errors_dir = os.path.join(os.path.dirname(__file__), "errors")

    latest_path = os.path.join(os.path.dirname(__file__), "latest.csv")
    snapshot_path = os.path.join(out_dir, f"{ts}.csv")

    chess_items: List[ChessApt] = []
    logger.info("Start chess scan: %d complexes", len(COMPLEX_SLUGS))

    for slug in t(COMPLEX_SLUGS, desc="Complexes"):
        base = chess_url(slug)
        html0 = _get(base, params={"from": "apartments"}, logger=logger)
        time.sleep(SLEEP_BETWEEN_PAGE_REQUESTS_SEC)
        houses = parse_button_values(html0, "Дом") or ["1"]
        logger.info("[CHESS] %s houses=%s", slug, houses)

        for house in t(houses, desc=f"{slug}: houses"):
            html_house = _get(base, params={"house": house, "from": "apartments"}, logger=logger)
            time.sleep(SLEEP_BETWEEN_PAGE_REQUESTS_SEC)
            entrances = parse_button_values(html_house, "Подъезд") or ["1"]
            logger.info("[CHESS] %s house=%s entrances=%s", slug, house, entrances)

            for entrance in t(entrances, desc=f"{slug} h{house}: entrances"):
                html_he = _get(base, params={"house": house, "entrance": entrance, "from": "apartments"}, logger=logger)
                time.sleep(SLEEP_BETWEEN_PAGE_REQUESTS_SEC)
                parsed = parse_apartments_from_chess(html_he, slug, house, entrance)
                logger.info("[CHESS] %s house=%s entrance=%s parsed=%d", slug, house, entrance, len(parsed))
                chess_items.extend(parsed)

    uniq: Dict[str, ChessApt] = {}
    for it_ in chess_items:
        uniq.setdefault(it_.apartment_id, it_)
    chess_items = list(uniq.values())
    logger.info("Unique apartments collected: %d", len(chess_items))

    apt_meta = {a.apartment_id: a for a in chess_items}
    all_ids = [a.apartment_id for a in chess_items]

    logger.info("API pass1: /api/apartment-info (workers=%d, in_flight=%d)", MAX_WORKERS, MAX_IN_FLIGHT)
    info_ok, errors = fetch_api_stream(
        apartment_ids=all_ids,
        logger=logger,
        desc="Apartment-info",
        max_workers=MAX_WORKERS,
        max_in_flight=MAX_IN_FLIGHT,
        sleep_between_submit=0.0,
    )

    if errors:
        os.makedirs(errors_dir, exist_ok=True)
        pass1_path = os.path.join(errors_dir, f"{ts}_api_errors_pass1.csv")
        write_errors_csv(pass1_path, errors, apt_meta=apt_meta)
        logger.warning("API pass1 failed: %d. Saved: %s", len(errors), pass1_path)
        logger.warning("API pass1 error stats: %s", error_stats(errors))
    else:
        logger.info("API pass1: no errors.")

    round_no = 0
    remaining_errors = errors

    # Auto-retry: up to 3 rounds automatically (with ~9s delay and adaptive softening).
    # After that, ask for manual confirmation and run in the softest mode.
    while remaining_errors:
        round_no += 1
        is_auto = round_no <= AUTO_RETRY_MAX_ROUNDS

        if is_auto:
            delay = AUTO_RETRY_SLEEP_SEC + random.uniform(0.0, AUTO_RETRY_JITTER_SEC)
            logger.warning(
                "Auto retry-pass #%d scheduled in %.1fs (remaining fails=%d)",
                round_no,
                delay,
                len(remaining_errors),
            )
            time.sleep(delay)
        else:
            # Manual confirmation starts only after all automatic rounds are used.
            if round_no == AUTO_RETRY_MAX_ROUNDS + 1:
                prompt = (
                    f"После {AUTO_RETRY_MAX_ROUNDS} авто-ретраев осталось {len(remaining_errors)} fail. "
                    f"Запустить самый мягкий ручной retry-pass #{round_no}?"
                )
            else:
                prompt = f"Осталось {len(remaining_errors)} fail. Запустить ещё один ручной retry-pass #{round_no}?"

            if not ask_yes_no(prompt, default_no=True):
                logger.warning("Manual retry cancelled by user. Remaining fails=%d", len(remaining_errors))
                break

        retry_ids = [e["apartment_id"] for e in remaining_errors]
        stats = error_stats(remaining_errors)
        workers, inflight, submit_sleep = choose_retry_params(remaining_errors, round_no)

        # After auto rounds, force the softest mode (manual), as required.
        if not is_auto:
            workers = MANUAL_SOFT_WORKERS
            inflight = MANUAL_SOFT_IN_FLIGHT
            submit_sleep = MANUAL_SOFT_SUBMIT_SLEEP_SEC + random.uniform(0.0, 0.4)

        logger.warning("Retry-pass #%d error stats: %s", round_no, stats)
        logger.info("Retry-pass #%d params: workers=%d in_flight=%d submit_sleep=%.2fs",
                    round_no, workers, inflight, submit_sleep)

        info_ok_retry, errors_retry = fetch_api_stream(
            apartment_ids=retry_ids,
            logger=logger,
            desc=f"Retry-failed #{round_no}",
            max_workers=workers,
            max_in_flight=inflight,
            sleep_between_submit=submit_sleep,
        )

        info_ok.update(info_ok_retry)
        remaining_errors = errors_retry

        os.makedirs(errors_dir, exist_ok=True)
        after_path = os.path.join(errors_dir, f"{ts}_api_errors_after_retry{round_no}.csv")
        write_errors_csv(after_path, remaining_errors, apt_meta=apt_meta)
        logger.warning("Retry-pass #%d done. Remaining fails=%d. Saved: %s",
                       round_no, len(remaining_errors), after_path)

        if not remaining_errors:
            logger.info("All API errors resolved after retry-pass #%d.", round_no)
            break

    err_map = {e["apartment_id"]: e for e in remaining_errors} if remaining_errors else {}
    rows: List[Dict[str, Any]] = []
    for apt in chess_items:
        aid = apt.apartment_id
        ok = aid in info_ok
        info = info_ok.get(aid)
        api_err = err_map.get(aid)
        rows.append(snapshot_row(now_iso, apt, info, api_ok=ok, api_err=api_err))

    write_csv(snapshot_path, rows)
    logger.info("Snapshot written: %s (rows=%d)", snapshot_path, len(rows))

    prev = read_csv_as_dict(latest_path, key="apartment_id")
    changes_path = None
    if prev:
        ch = compute_changes(prev, rows)
        changes_path = os.path.join(changes_dir, f"{ts}_changes.csv")
        write_csv(changes_path, ch)
        logger.info("Changes written: %s (rows=%d)", changes_path, len(ch))
    else:
        logger.info("No previous latest.csv found; changes not computed.")

    write_csv(latest_path, rows)
    logger.info("Latest updated: %s", latest_path)

    log_file = os.path.join(os.path.dirname(__file__), "logs", f"run_{ts}.log")
    return snapshot_path, latest_path, changes_path, log_file


def run_retry_only(errors_csv_path: str, logger: logging.Logger) -> Tuple[str, Optional[str], Optional[str]]:
    now = dt.datetime.now(dt.timezone.utc).astimezone().replace(microsecond=0)
    ts = now.strftime("%Y%m%d_%H%M%S")

    retry_dir = os.path.join(os.path.dirname(__file__), "retry")
    os.makedirs(retry_dir, exist_ok=True)

    ids = read_apartment_ids_from_errors_csv(errors_csv_path)
    logger.info("Retry-only: loaded %d apartment_id from %s", len(ids), errors_csv_path)

    info_ok, errors = fetch_api_stream(
        apartment_ids=ids,
        logger=logger,
        desc="Retry-only",
        max_workers=RETRY_BASE_WORKERS,
        max_in_flight=RETRY_BASE_IN_FLIGHT,
        sleep_between_submit=RETRY_BASE_SUBMIT_SLEEP_SEC,
    )

    rows = []
    for aid, info in info_ok.items():
        rows.append({
            "apartment_id": aid,
            "status": info.get("status"),
            "price": info.get("price"),
            "promotion_price": info.get("promotion_price"),
            "hypothec_price": info.get("hypothec_price"),
            "cost_per_square": info.get("cost_per_square"),
            "area": info.get("area"),
            "rooms": info.get("rooms"),
            "floor": info.get("floor"),
            "floor_count": info.get("floor_count"),
            "finishing_type": info.get("finishing_type"),
            "updated_at_unix": info.get("updated_at"),
        })
    results_path = os.path.join(retry_dir, f"{ts}_retry_results.csv")
    write_csv(results_path, rows)

    errors_path = None
    if errors:
        errors_path = os.path.join(retry_dir, f"{ts}_retry_errors.csv")
        write_errors_csv(errors_path, errors, apt_meta=None)
        logger.warning("Retry-only error stats: %s", error_stats(errors))

    logger.info("Retry-only results: %s (ok=%d, fail=%d)", results_path, len(info_ok), len(errors))

    latest_path = os.path.join(os.path.dirname(__file__), "latest.csv")
    patched_latest_path = None
    if os.path.exists(latest_path) and info_ok:
        patched_latest_path = os.path.join(retry_dir, f"{ts}_patched_latest.csv")
        patch_snapshot_csv(latest_path, info_ok, logger, out_path=patched_latest_path)

    return results_path, errors_path, patched_latest_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "--retry-only",
        dest="retry_only",
        default=None,
        help="Путь к errors csv (с колонкой apartment_id), чтобы дособрать цены без шахматок.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ts = dt.datetime.now(dt.timezone.utc).astimezone().replace(microsecond=0).strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(ts)

    if args.retry_only:
        results, errors_path, patched_latest = run_retry_only(args.retry_only, logger)
        print(f"\n{OK_COLOR}=== RETRY-ONLY DONE ==={RESET}")
        print("Results :", results)
        print("Errors  :", errors_path or "(no errors)")
        print("Patched latest copy:", patched_latest or "(latest.csv not found or nothing to patch)")
        return

    snap, latest, changes, logf = run_full(logger)
    print(f"\n{OK_COLOR}=== DONE ==={RESET}")
    print("Snapshot :", snap)
    print("Latest   :", latest)
    print("Changes  :", changes or "(no previous latest.csv)")
    print("Log file :", logf)


if __name__ == "__main__":
    main()
