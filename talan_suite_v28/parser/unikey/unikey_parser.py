#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unikey (unikey.space) MacroCatalog parser (SberCRM / macro.sbercrm.com)

Поддерживаемый поток запросов:
  A) GET /estate/request/get_request_url/  -> возвращает временную подписанную ссылку /estate/catalog/?...&time=...&token=...
  B) POST /estate/catalog/  action=get_data  -> список ЖК/домов, фильтров, промо и пр.
  C) POST /estate/catalog/  action=get_estates (house_id) -> "шахматка" (все квартиры по подъездам/этажам)

Зачем отдельный A:
  url из A содержит временные параметры (time/token). Их нужно получать перед каждым запуском.

Запуск (Windows):
  1) Заполни parser/unikey/unikey_config.json (или укажи --config)
  2) run_unikey_parser.bat

Запуск (CLI):
  python parser/unikey_parser.py --config parser/unikey/unikey_config.json

"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from colorama import Fore, Style, init as colorama_init

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


colorama_init()

OK = Fore.GREEN
WARN = Fore.YELLOW
ERR = Fore.RED
RST = Style.RESET_ALL

SOFT_GREEN = "#7ED957"  # мягкий зелёный для progress-bar

DEFAULT_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "origin": "https://unikey.space",
    "referer": "https://unikey.space/",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}


@dataclass
class Config:
    # A) URL to /estate/request/get_request_url/ ...
    # If empty and auto-refresh method includes 'browser', the parser can discover it automatically.
    get_request_url: str = ""
    # If get_request_url becomes invalid (e.g., "check" changed), the parser can try to
    # re-discover it by downloading Unikey pages and searching for the embedded MacroCatalog URL.
    auto_refresh_get_request_url: bool = True
    # Auto-refresh method order: "probe" (scan HTML) and/or "browser" (Playwright intercept).
    # Examples: "probe", "browser", "probe,browser".
    auto_refresh_method: str = "probe,browser"
    # Entry URL to open in browser auto-refresh mode.
    # IMPORTANT: MacroCatalog uses short-lived signed URLs (token/time) and some deep links
    # can show the message "Запрошенные данные устарели, не найдены или удалены" when opened
    # directly in a fresh browser context.
    # For stability we start from the Projects page: user can pick a city (e.g. Хабаровск)
    # and then open the catalog (complexes / houses / chessboard).
    auto_refresh_browser_entry_url: str = "https://unikey.space/projects/"

    # Interactive assist mode: if popups block the UI or the bot "hangs", the user can
    # manually click through the site while the parser listens for the get_request_url call.
    # When the request is captured, the parser continues automatically.
    auto_refresh_browser_assist_mode: bool = True
    # How long to wait for the user to trigger get_request_url in the opened browser.
    # Set <=0 to wait indefinitely.
    auto_refresh_browser_assist_max_wait_sec: int = 1800
    # Human-like browser pacing to reduce flakiness.
    auto_refresh_browser_slowmo_ms: int = 120
    auto_refresh_browser_wait_sec: int = 45
    # Use headless browser for auto-refresh.
    auto_refresh_browser_headless: bool = False
    # Probe pages to scan for an embedded get_request_url. If not set, defaults are used.
    unikey_probe_urls: Optional[List[str]] = None
    # If refresh succeeds, write updated get_request_url back to config file.
    persist_refreshed_get_request_url: bool = True
    target_city_contains: str = "Хабаровск"
    # Какие категории собирать. Если не задано, парсер попытается собрать квартиры + кладовые + машино-места.
    # Значения должны соответствовать `house.categories` в ответе get_data (например: "flat", "pantry", "parking").
    categories: Optional[List[str]] = None
    # Строгий режим: если не удалось собрать дополнительные категории (не flat) — это считается ошибкой.
    extra_categories_strict: bool = True
    category: str = "flat"
    activity: str = "sell"
    houses: Optional[List[int]] = None
    out_dir: str = "output_unikey"
    timeout_sec: int = 35
    retries: int = 3
    sleep_between_requests_sec: float = 0.1



@dataclass
class BrowserDiscovery:
    get_request_url: Optional[str] = None
    signed_catalog_url: Optional[str] = None
    used_browser: Optional[str] = None


def _now_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _t(it: Iterable[Any], desc: str, total: Optional[int] = None):
    if tqdm is None:
        return it
    return tqdm(it, desc=desc, total=total, dynamic_ncols=True, colour=SOFT_GREEN, leave=True)


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # categories может быть:
    #   - списком ["flat","pantry","parking"]
    #   - строкой "flat,pantry,parking"
    #   - словарём {"flat":"flat","pantry":"storeroom","parking":"parking"} (canonical->actual)
    categories_raw = raw.get("categories")
    categories: Optional[List[str]] = None
    if isinstance(categories_raw, dict):
        categories = []
        for k, v in categories_raw.items():
            ks = str(k).strip()
            vs = str(v).strip()
            if ks and vs:
                categories.append(f"{ks}={vs}")
    elif isinstance(categories_raw, str):
        categories = [x.strip() for x in categories_raw.split(",") if x.strip()]
    elif isinstance(categories_raw, list):
        categories = [str(x).strip() for x in categories_raw if str(x).strip()]
    else:
        categories = None

    # 'get_request_url' can be empty if auto-refresh method includes 'browser'.
    get_request_url = str(raw.get("get_request_url", "") or "").strip()
    auto_refresh_method = str(raw.get("auto_refresh_method", "probe,browser") or "probe,browser")
    auto_refresh_enabled = bool(raw.get("auto_refresh_get_request_url", True))
    # If user left get_request_url empty/placeholder, force-enable the reliable 'browser' method.
    if (not get_request_url) and auto_refresh_enabled and ("browser" not in auto_refresh_method.lower()):
        auto_refresh_method = (auto_refresh_method + ",browser").strip(",")
    if not get_request_url:
        if not (auto_refresh_enabled and "browser" in auto_refresh_method.lower()):
            raise ValueError(
                "Config must contain 'get_request_url' (A request URL), or enable browser auto-refresh "
                "by setting auto_refresh_method to include 'browser'."
            )
    return Config(
        get_request_url=get_request_url,
        auto_refresh_get_request_url=auto_refresh_enabled,
        auto_refresh_method=auto_refresh_method,
        auto_refresh_browser_entry_url=raw.get(
            "auto_refresh_browser_entry_url",
            "https://unikey.space/projects/",
        ),
        auto_refresh_browser_headless=bool(raw.get("auto_refresh_browser_headless", False)),
        auto_refresh_browser_assist_mode=str(raw.get("auto_refresh_browser_assist_mode", "manual")).strip().lower(),
        auto_refresh_browser_assist_max_wait_sec=int(raw.get("auto_refresh_browser_assist_max_wait_sec", 1800)),
        auto_refresh_browser_slowmo_ms=int(raw.get("auto_refresh_browser_slowmo_ms", 120)),
        auto_refresh_browser_wait_sec=int(raw.get("auto_refresh_browser_wait_sec", 45)),
        unikey_probe_urls=raw.get("unikey_probe_urls"),
        persist_refreshed_get_request_url=bool(raw.get("persist_refreshed_get_request_url", True)),
        target_city_contains=raw.get("target_city_contains", "Хабаровск"),
        categories=categories,
        extra_categories_strict=bool(raw.get("extra_categories_strict", True)),
        category=raw.get("category", "flat"),
        activity=raw.get("activity", "sell"),
        houses=raw.get("houses"),
        out_dir=raw.get("out_dir", "output_unikey"),
        timeout_sec=int(raw.get("timeout_sec", 35)),
        retries=int(raw.get("retries", 3)),
        sleep_between_requests_sec=float(raw.get("sleep_between_requests_sec", 0.1)),
    )



class _TqdmLoggingHandler(logging.Handler):
    """Logging handler that plays nice with tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if tqdm is not None:
                try:
                    tqdm.write(msg)
                except Exception:
                    print(msg)
            else:
                print(msg)
        except Exception:
            # logging must never crash the parser
            pass


def setup_logging(ts: str, out_dir: str) -> logging.Logger:
    log_dir = os.path.join(out_dir, "logs")
    _ensure_dir(log_dir)
    log_path = os.path.join(log_dir, f"run_{ts}.log")

    logger = logging.getLogger("unikey_parser")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = _TqdmLoggingHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Logging to %s", log_path)
    return logger


def http_json(session: requests.Session, method: str, url: str, *, json_body: Optional[dict] = None,
             timeout: int = 35, retries: int = 3, logger: Optional[logging.Logger] = None) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            if method.upper() == "GET":
                r = session.get(url, timeout=timeout)
            else:
                r = session.post(url, json=json_body, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if logger:
                logger.warning("HTTP %s fail (%s/%s) url=%s err=%s", method, attempt, retries, url, e)
            time.sleep(0.6 * attempt)
    raise RuntimeError(f"HTTP {method} failed after {retries} tries: {url} err={last_err}")


def fetch_catalog_url(session: requests.Session, cfg: Config, logger: logging.Logger) -> str:
    j = http_json(session, "GET", cfg.get_request_url, timeout=cfg.timeout_sec, retries=cfg.retries, logger=logger)
    url = j.get("url")
    if not url:
        raise RuntimeError(f"Unexpected response from get_request_url: keys={list(j.keys())}")
    return url


def _looks_like_placeholder(url: str) -> bool:
    u = (url or "").strip()
    return (not u) or ("PASTE_HERE" in u) or ("GET_REQUEST_URL" in u)


def _extract_get_request_url_from_text(text: str) -> Optional[str]:
    """Heuristic extractor for embedded MacroCatalog get_request_url.

    We try to find a full URL or a relative path to:
      /estate/request/get_request_url/?domain=unikey.space&check=...

    Note: this is best-effort. If the token/check is generated at runtime and not embedded
    in HTML/JS, you'll still need to copy A from DevTools.
    """
    if not text:
        return None

    # 1) Full URL inside quotes
    m = re.search(r"https://api\\.macro\\.sbercrm\\.com/estate/request/get_request_url/\\?[^\"']+", text)
    if m:
        return m.group(0).rstrip("\"' )];,\n\r\t")

    # 2) Relative path
    m = re.search(r"/estate/request/get_request_url/\\?[^\"']+", text)
    if m:
        return ("https://api.macro.sbercrm.com" + m.group(0)).rstrip("\"' )];,\n\r\t")

    return None


def try_refresh_get_request_url_probe(session: requests.Session, cfg: Config, logger: logging.Logger) -> Optional[str]:
    """Best-effort refresh by scanning HTML/JS for an embedded get_request_url."""
    probe_urls = cfg.unikey_probe_urls or ["https://unikey.space/", "https://unikey.space/catalog/"]
    html_headers = {
        "user-agent": DEFAULT_HEADERS.get("user-agent", "Mozilla/5.0"),
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    for u in probe_urls:
        try:
            r = session.get(u, headers=html_headers, timeout=min(20, cfg.timeout_sec))
            if r.status_code >= 400:
                continue
            guess = _extract_get_request_url_from_text(r.text)
            if guess:
                logger.info("Auto-refresh(probe): found get_request_url in %s", u)
                return guess
        except Exception as e:
            logger.warning("Auto-refresh(probe) failed: %s err=%s", u, e)
            continue
    return None



def _discover_macro_urls_via_browser(cfg: Config, logger: logging.Logger) -> BrowserDiscovery:
    """Discover MacroCatalog URLs via Playwright network interception.

    We try to capture:
      - A request URL: /estate/request/get_request_url/?domain=unikey.space...
      - Signed catalog URL: /estate/catalog/?domain=unikey.space&...&time=...&token=...

    On a normal PC this is most reliable in *headed* mode (headless=False).
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:
        logger.warning(
            "Auto-refresh(browser) unavailable (Playwright not installed). Install playwright. err=%s",
            e,
        )
        return BrowserDiscovery()

    # Entry URLs: first configured, then a couple of known stable pages that always load the grid widget.
    entry_urls: List[str] = []
    if (cfg.auto_refresh_browser_entry_url or "").strip():
        entry_urls.append(cfg.auto_refresh_browser_entry_url.strip())

    # Prefer safe entry points.
    # Deep links to houses/... are sometimes rejected with "Запрошенные данные устарели..." when opened directly
    # in a fresh browser context. Starting from /catalog/ and then routing to the list is more reliable.
    entry_urls.extend([
        "https://unikey.space/catalog/",
        "https://unikey.space/catalog/#/macrocatalog/complexes/list?studio=null&category=flat&activity=sell",
    ])

    # de-dup while keeping order
    seen = set()
    entry_urls = [u for u in entry_urls if not (u in seen or seen.add(u))]

    needle_a = "/estate/request/get_request_url/"
    needle_catalog = "/estate/catalog/"

    found_a: Optional[str] = None
    found_signed: Optional[str] = None
    used_browser: Optional[str] = None

    headless = bool(cfg.auto_refresh_browser_headless)
    slow_mo = int(getattr(cfg, "auto_refresh_browser_slowmo_ms", 60)) if not headless else 0

    def goto_safely(page, url: str) -> None:
        """Open MacroCatalog in a way that avoids expired deep-link errors.

        If the URL contains a hash route (/#/...), first open the base /catalog/ page, then set location.hash.
        This mimics a human path (main -> route) and usually triggers the needed API calls.
        """
        url = url.strip()
        if "#/" in url:
            base, hash_part = url.split("#", 1)
            base = base.rstrip("/") + "/"
            hash_part = "#" + hash_part
            page.goto(base, wait_until="domcontentloaded")
            page.wait_for_timeout(800)
            page.evaluate("(h) => { window.location.hash = h; }", hash_part)
        else:
            page.goto(url, wait_until="domcontentloaded")

        # Some routes show a big "data expired" message in Russian. If spotted, force-open complexes list.
        try:
            txt = (page.inner_text("body") or "").lower()
            if ("данные устарели" in txt) or ("не найдены" in txt and "устар" in txt) or ("удален" in txt and "дан" in txt):
                page.evaluate("() => { window.location.hash = '#/macrocatalog/complexes/list?studio=null&category=flat&activity=sell'; }")
                page.wait_for_timeout(800)
        except Exception:
            pass

        # If the app routes to a 404/expired screen, try to force-open the safe complexes list.
        try:
            if "#/macrocatalog/404" in page.url:
                page.evaluate("() => { window.location.hash = '#/macrocatalog/complexes/list?studio=null&category=flat&activity=sell'; }")
                page.wait_for_timeout(800)
        except Exception:
            pass

        # Some routes show a big "Запрошенные данные устарели, не найдены или удалены" message.
        # If spotted, force-open complexes list (this usually triggers get_request_url).
        try:
            body_txt = page.inner_text("body") or ""
            lower = body_txt.lower()
            if "данные устарели" in lower or "запрошенные данные устарели" in lower:
                page.evaluate("() => { window.location.hash = '#/macrocatalog/complexes/list?studio=null&category=flat&activity=sell'; }")
                page.wait_for_timeout(800)
        except Exception:
            pass

    def is_signed_catalog(u: str) -> bool:
        return (
            ("api.macro.sbercrm.com" in u)
            and (needle_catalog in u)
            and ("domain=unikey.space" in u)
            and ("token=" in u)
            and ("time=" in u)
            and ("type=catalog" in u)
        )

    try:
        with sync_playwright() as p:
            launch_kwargs = {
                "headless": headless,
                "slow_mo": slow_mo,
                "args": [
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            }

            browser = None
            last_launch_err: Optional[Exception] = None
            channels = list(getattr(cfg, "auto_refresh_browser_channels", None) or ["chrome", "msedge"])
            for channel in channels:
                try:
                    browser = p.chromium.launch(channel=channel, **launch_kwargs)
                    used_browser = channel
                    break
                except Exception as e:
                    last_launch_err = e
                    continue

            # Extra fallback for Windows: try common executable paths if Playwright can't resolve a channel.
            if browser is None and os.name == "nt":
                exe_candidates: List[str] = []
                env = os.environ
                pf = env.get("PROGRAMFILES") or "C:\\Program Files"
                pfx86 = env.get("PROGRAMFILES(X86)") or "C:\\Program Files (x86)"
                lad = env.get("LOCALAPPDATA") or ""
                exe_candidates.extend([
                    os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
                    os.path.join(pfx86, "Google", "Chrome", "Application", "chrome.exe"),
                    os.path.join(lad, "Google", "Chrome", "Application", "chrome.exe"),
                    os.path.join(pf, "Microsoft", "Edge", "Application", "msedge.exe"),
                    os.path.join(pfx86, "Microsoft", "Edge", "Application", "msedge.exe"),
                    os.path.join(lad, "Microsoft", "Edge", "Application", "msedge.exe"),
                ])
                for exe in exe_candidates:
                    try:
                        if exe and os.path.exists(exe):
                            browser = p.chromium.launch(executable_path=exe, **launch_kwargs)
                            used_browser = os.path.basename(exe)
                            break
                    except Exception as e:
                        last_launch_err = e
                        continue

            if browser is None:
                # Fallback to bundled Chromium (requires: python -m playwright install chromium)
                try:
                    browser = p.chromium.launch(**launch_kwargs)
                    used_browser = "chromium"
                except Exception as e:
                    logger.warning(
                        "Auto-refresh(browser) failed to launch (chrome/msedge/chromium). "
                        "If you see 'Executable doesn't exist', run: .\\.venv\\Scripts\\python -m playwright install chromium. "
                        "err=%s last_channel_err=%s",
                        e, last_launch_err,
                    )
                    return BrowserDiscovery()

            context = browser.new_context()
            page = context.new_page()

            def on_request(req):
                nonlocal found_a, found_signed
                u = req.url
                if (found_a is None) and (needle_a in u) and ("domain=unikey.space" in u) and ("type=catalog" in u):
                    found_a = u
                if (found_signed is None) and is_signed_catalog(u):
                    found_signed = u

            def on_response(resp):
                nonlocal found_a, found_signed
                u = resp.url
                if (needle_a in u) and ("domain=unikey.space" in u):
                    if found_a is None:
                        found_a = u
                    try:
                        j = resp.json()
                        su = j.get("url")
                        if isinstance(su, str) and is_signed_catalog(su):
                            found_signed = su
                    except Exception:
                        pass

            page.on("request", on_request)
            page.on("response", on_response)

            # --- Assisted (manual) mode ---
            # In this mode we open a stable start page and simply wait while the user navigates.
            # This prevents failures when popups/promos block automated clicks.
            if bool(getattr(cfg, "auto_refresh_browser_assist_mode", True)):
                start_url = str(getattr(cfg, "auto_refresh_browser_entry_url", "") or "").strip() or (entry_urls[0] if entry_urls else "https://unikey.space/projects/")
                try:
                    goto_safely(page, start_url)
                except Exception:
                    # try at least some safe url
                    goto_safely(page, "https://unikey.space/projects/")

                logger.info("Manual token refresh: Chrome window is open.")
                logger.info("Please do the following IN THE OPENED BROWSER (you can click/scroll/close banners):")
                logger.info("  1) On https://unikey.space/projects/ choose city: Хабаровск")
                logger.info("  2) Open Catalog -> Complexes")
                logger.info("  3) Open any house and its chessboard (floor/bigGrid)")
                logger.info("As soon as the site requests /estate/request/get_request_url, the parser will capture it and continue.")
                logger.info("Tip: if a promo/chat blocks UI, just close it manually — parser will keep waiting.")

                max_wait_sec = int(getattr(cfg, "auto_refresh_browser_assist_max_wait_sec", 1800) or 1800)
                poll_ms = 250
                started = time.time()
                last_ping = 0.0
                while not (found_a or found_signed):
                    if max_wait_sec > 0 and (time.time() - started) > max_wait_sec:
                        break
                    page.wait_for_timeout(poll_ms)
                    if (time.time() - last_ping) > 20:
                        elapsed = int(time.time() - started)
                        logger.info("Waiting for token... (%ss elapsed)", elapsed)
                        last_ping = time.time()
            else:
                # --- Auto mode (legacy) ---
                # navigate through entry urls until we capture something
                max_wait_sec = int(getattr(cfg, "auto_refresh_browser_wait_sec", 30) or 30)
                poll_ms = 250
                max_polls = max(1, int((max_wait_sec * 1000) / poll_ms))
                for u in entry_urls:
                    try:
                        goto_safely(page, u)
                    except Exception:
                        continue

                    # wait for the widget to fire its requests
                    for _ in range(max_polls):
                        if found_a or found_signed:
                            break
                        page.wait_for_timeout(poll_ms)

                    if found_a or found_signed:
                        break

                    # sometimes SPA does not fire without reload
                    try:
                        page.reload(wait_until="domcontentloaded", timeout=60000)
                        for _ in range(min(max_polls, 120)):
                            if found_a or found_signed:
                                break
                            page.wait_for_timeout(poll_ms)
                    except Exception:
                        pass

                    if found_a or found_signed:
                        break

            try:
                context.close()
                browser.close()
            except Exception:
                pass
    except Exception as e:
        logger.warning("Auto-refresh(browser) failed: %s", e)
        return BrowserDiscovery()

    if found_a:
        logger.info("Auto-refresh(browser): captured get_request_url request")
    if found_signed:
        logger.info("Auto-refresh(browser): captured signed catalog url")

    return BrowserDiscovery(get_request_url=found_a, signed_catalog_url=found_signed, used_browser=used_browser)


def try_refresh_get_request_url_browser(cfg: Config, logger: logging.Logger) -> Optional[str]:
    """Refresh A (get_request_url) via browser interception (Playwright)."""
    d = _discover_macro_urls_via_browser(cfg, logger)
    return d.get_request_url


def try_discover_signed_catalog_url_browser(cfg: Config, logger: logging.Logger) -> Optional[str]:
    """Discover signed catalog url directly (so we can proceed even if A isn't visible)."""
    d = _discover_macro_urls_via_browser(cfg, logger)
    return d.signed_catalog_url
def try_refresh_get_request_url(session: requests.Session, cfg: Config, logger: logging.Logger) -> Optional[str]:
    """Refresh get_request_url using methods in cfg.auto_refresh_method order."""
    methods = [m.strip().lower() for m in str(cfg.auto_refresh_method or "probe").split(",") if m.strip()]
    if not methods:
        methods = ["probe"]

    for m in methods:
        if m == "probe":
            guess = try_refresh_get_request_url_probe(session, cfg, logger)
        elif m == "browser":
            guess = try_refresh_get_request_url_browser(cfg, logger)
        else:
            logger.warning("Unknown auto_refresh_method entry: %s", m)
            guess = None

        if guess:
            return guess
    return None


def persist_get_request_url(config_path: str, new_url: str, logger: logging.Logger) -> None:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["get_request_url"] = new_url
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        logger.info("Config updated: get_request_url refreshed and saved to %s", config_path)
    except Exception as e:
        logger.warning("Failed to persist refreshed get_request_url to config: %s", e)


def fetch_get_data(session: requests.Session, catalog_url: str, cfg: Config, logger: logging.Logger) -> dict:
    payload = {"action": "get_data", "data": {"cabinetMode": False}, "auth_token": None, "locale": None}
    j = http_json(session, "POST", catalog_url, json_body=payload, timeout=cfg.timeout_sec, retries=cfg.retries, logger=logger)
    if not j.get("success", False):
        raise RuntimeError(f"get_data returned success=false: {str(j)[:300]}")
    return j


def fetch_get_estates(session: requests.Session, catalog_url: str, cfg: Config, house_id: int, logger: logging.Logger, *, category: Optional[str] = None) -> dict:
    payload = {
        "action": "get_estates",
        "data": {"house_id": house_id, "category": (category or cfg.category), "cabinetMode": False},
        "auth_token": None,
        "locale": None,
    }
    j = http_json(session, "POST", catalog_url, json_body=payload, timeout=cfg.timeout_sec, retries=cfg.retries, logger=logger)
    if not j.get("success", False):
        raise RuntimeError(f"get_estates returned success=false house_id={house_id}: {str(j)[:300]}")
    return j


def pick_target_houses(get_data: dict, cfg: Config, *, category: Optional[str] = None) -> List[Tuple[int, str, str]]:
    """Возвращает список (house_id, house_public_name, complex_name)."""
    houses = get_data.get("houses") or []
    complexes = {c.get("id"): c for c in (get_data.get("complexes") or [])}

    # Фильтр по городу — через complexes.city, на всякий случай.
    allowed_complex_ids = set()
    for cid, c in complexes.items():
        city = (c.get("city") or "")
        if cfg.target_city_contains.lower() in city.lower():
            allowed_complex_ids.add(cid)

    def clean_html(s: str) -> str:
        # public_name в ответе иногда содержит &nbsp;
        s = re.sub(r"<[^>]+>", "", s)
        return s.replace("\xa0", " ").replace("&nbsp;", " ").strip()

    out: List[Tuple[int, str, str]] = []
    for h in houses:
        hid = h.get("id")
        if not isinstance(hid, int):
            continue
        if cfg.houses and hid not in cfg.houses:
            continue
        cid = h.get("complex_id")
        if cid not in allowed_complex_ids:
            continue

        # Берём только дома, где в categories присутствует нужная категория.
        cat = category or cfg.category
        categories = h.get("categories") or []
        if cat not in categories:
            continue

        house_name = clean_html(str(h.get("public_name") or h.get("name") or hid))
        complex_name = clean_html(str(h.get("complex_name") or complexes.get(cid, {}).get("name") or cid))
        out.append((hid, house_name, complex_name))

    # Стабильный порядок: по ЖК, потом по дому
    out.sort(key=lambda x: (x[2], x[1], x[0]))
    return out


def collect_available_categories(get_data: dict, cfg: Config) -> List[str]:
    """Собирает все категории, доступные для домов в целевом городе (по complexes.city)."""
    houses = get_data.get("houses") or []
    complexes = {c.get("id"): c for c in (get_data.get("complexes") or [])}
    allowed_complex_ids = set()
    for cid, c in complexes.items():
        city = (c.get("city") or "")
        if cfg.target_city_contains.lower() in city.lower():
            allowed_complex_ids.add(cid)
    cats: List[str] = []
    seen = set()
    for h in houses:
        cid = h.get("complex_id")
        if cid not in allowed_complex_ids:
            continue
        for cat in (h.get("categories") or []):
            sc = str(cat).strip()
            if not sc or sc in seen:
                continue
            seen.add(sc)
            cats.append(sc)
    return cats


CATEGORY_SYNONYMS = {
    # canonical -> possible API tokens
    "flat": ["flat", "apartment"],
    "pantry": ["pantry", "storage", "storeroom", "storageroom", "kladovka"],
    "parking": ["parking", "parking_place", "car_place", "mashinomesto", "garage"],
}


def resolve_categories(available: List[str], wanted: List[str]) -> List[Tuple[str, str]]:
    """Возвращает список (canonical, actual), подбирая actual из available по синонимам.
    Если canonical уже выглядит как реальный токен — пробуем использовать его напрямую."""
    avail_set = {a.strip() for a in available if str(a).strip()}
    out: List[Tuple[str, str]] = []
    used_actual = set()
    for w in wanted:
        w_str = str(w).strip()
        if not w_str:
            continue
        # Явное соответствие: "pantry=storeroom" или "pantry:storeroom"
        if ("=" in w_str) or (":" in w_str):
            sep = "=" if "=" in w_str else ":"
            left, right = w_str.split(sep, 1)
            canonical = left.strip().lower()
            actual = right.strip()
            if canonical and actual and actual not in used_actual:
                out.append((canonical, actual))
                used_actual.add(actual)
            continue
        canonical = w_str.lower()
        if not canonical:
            continue
        # если пользователь дал конкретный токен, и он присутствует — используем
        if canonical in (s.lower() for s in avail_set):
            # найдём exact actual c исходным кейсом
            actual = next(a for a in avail_set if a.lower() == canonical)
            if actual not in used_actual:
                out.append((canonical, actual))
                used_actual.add(actual)
            continue
        candidates = CATEGORY_SYNONYMS.get(canonical, [canonical])
        actual = None
        for cand in candidates:
            for a in avail_set:
                if a.lower() == cand.lower():
                    actual = a
                    break
            if actual:
                break
        if actual and actual not in used_actual:
            out.append((canonical, actual))
            used_actual.add(actual)
    return out


def flatten_estates(get_estates: dict, *, captured_at_iso: str, lot_type: str, requested_category: str, city: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    estates = get_estates.get("estates") or []

    for e in estates:
        house_id = e.get("id")
        house_name = e.get("name")
        house_deadline = e.get("date")

        for entrance in (e.get("entrances") or []):
            entrance_num = entrance.get("number")
            for floor in (entrance.get("floors") or []):
                floor_num = floor.get("number")
                floor_plan = floor.get("floor_plan") or {}
                floor_plan_url = floor_plan.get("file_url")
                floor_plan_name = floor_plan.get("file_name")

                for item in (floor.get("items") or []):
                    item_id = item.get("id")
                    status = item.get("status")
                    ext_link = item.get("external_link")

                    est = item.get("estate")
                    if not isinstance(est, dict) or not est:
                        # пустая ячейка / нет лота
                        continue
                    if not item_id:
                        # бывает пустая ячейка
                        continue

                    # цены/площади часто приходят строками
                    def to_float(v: Any) -> Optional[float]:
                        if v is None:
                            return None
                        try:
                            return float(str(v).replace(" ", "").replace(",", "."))
                        except Exception:
                            return None

                    plans = item.get("plans") or []
                    plan_primary_url = None
                    plan_primary_name = None
                    for p in plans:
                        if p.get("type") == "primary":
                            plan_primary_url = p.get("file_url")
                            plan_primary_name = p.get("file_name")
                            break
                    if plan_primary_url is None and plans:
                        plan_primary_url = plans[0].get("file_url")
                        plan_primary_name = plans[0].get("file_name")

                    promos = item.get("promos") or []
                    promo_ids = ",".join(str(p.get("id")) for p in promos if p.get("id") is not None)
                    promo_names = "; ".join(str(p.get("name")) for p in promos if p.get("name"))
                    promo_discounts = ",".join(str(p.get("discount")) for p in promos if p.get("discount") is not None)
                    promo_date_from = ",".join(str(p.get("date_from")) for p in promos if p.get("date_from"))
                    promo_date_to = ",".join(str(p.get("date_to")) for p in promos if p.get("date_to"))

                    lot_category = (est.get("category") or requested_category)
                    category_mismatch = 0
                    try:
                        if requested_category and (str(lot_category).lower() != str(requested_category).lower()):
                            category_mismatch = 1
                    except Exception:
                        category_mismatch = 0

                    rows.append({
                        "captured_at": captured_at_iso,
                        "developer_key": "unikey",
                        "developer": "Unikey",
                        "city": city,
                        "source": "macro.sbercrm.com",
                        "lot_type": lot_type,
                        "lot_category": lot_category,
                        "lot_item_id": item_id,
                        "lot_status": status,
                        "category_mismatch": category_mismatch,
                        "house_id": house_id,
                        "house_name": house_name,
                        "house_deadline": house_deadline,
                        "entrance": entrance_num,
                        "floor": floor_num,
                        "flat_item_id": item_id,
                        "flat_status": status,
                        "external_link": ext_link,
                        "title": est.get("title"),
                        "flatnum": est.get("geo_flatnum"),
                        "rooms": est.get("estate_rooms"),
                        "area_m2": to_float(est.get("estate_area")),
                        "price": to_float(est.get("estate_price")),
                        "price_m2": to_float(est.get("estate_price_m2")),
                        "price_without_discount": to_float(est.get("priceWithoutDiscount")),
                        "price_minimal": to_float(est.get("estate_price_minimal")),
                        "exclusive": est.get("estate_exclusive"),
                        "restoration_id": est.get("restoration_id"),
                        "plans_id": est.get("plans_id"),
                        "video": est.get("estate_video"),
                        "pano": est.get("estate_pano"),
                        "floor_plan_url": floor_plan_url,
                        "floor_plan_name": floor_plan_name,
                        "plan_primary_url": plan_primary_url,
                        "plan_primary_name": plan_primary_name,
                        "promo_ids": promo_ids,
                        "promo_names": promo_names,
                        "promo_discounts": promo_discounts,
                        "promo_date_from": promo_date_from,
                        "promo_date_to": promo_date_to,
                    })

    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        # чтобы не создавать пустышку
        return
    _ensure_dir(os.path.dirname(path))
    # Собираем все колонки (устойчиво к добавлению новых полей в части строк)
    cols: List[str] = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_errors(path: str, errors: List[Dict[str, Any]]) -> None:
    if not errors:
        return
    _ensure_dir(os.path.dirname(path))
    cols: List[str] = list(errors[0].keys())
    for r in errors[1:]:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in errors:
            w.writerow(r)

def main() -> int:
    ap = argparse.ArgumentParser(description="Unikey macro catalog parser (Khabarovsk)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "unikey_config.json"), help="Path to config JSON")
    ap.add_argument("--houses", default="", help="Comma-separated house_id list (optional override)")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"{ERR}Config error:{RST} {e}")
        return 2

    if args.houses.strip():
        try:
            cfg.houses = [int(x.strip()) for x in args.houses.split(",") if x.strip()]
        except Exception:
            print(f"{ERR}Bad --houses value. Use comma-separated ints.{RST}")
            return 2

    ts = _now_ts()
    base_dir = os.path.join(os.path.dirname(__file__), cfg.out_dir)
    snapshots_dir = os.path.join(base_dir, "snapshots")
    errors_dir = os.path.join(base_dir, "errors")
    _ensure_dir(snapshots_dir)
    _ensure_dir(errors_dir)

    logger = setup_logging(ts, base_dir)
    logger.info("Config: city contains=%s, category=%s, houses=%s", cfg.target_city_contains, cfg.category, cfg.houses or "ALL")

    total_stages = 5
    def stage(n: int, title: str) -> None:
        # concise, human-readable progress (visible in console window)
        logger.info("[PROGRESS %d/%d] %s", n, total_stages, title)


    sess = requests.Session()
    sess.headers.update(DEFAULT_HEADERS)

    captured_at_iso = dt.datetime.now(dt.timezone.utc).isoformat()

    # 1) Получаем временную ссылку /estate/catalog/
    stage(1, "Получаем подписанную ссылку на каталог (token/time)")
    catalog_url: Optional[str] = None
    # NOTE: empty/placeholder get_request_url is OK when auto-refresh is enabled.
    initial_placeholder = _looks_like_placeholder(cfg.get_request_url)
    try:
        if not initial_placeholder:
            catalog_url = fetch_catalog_url(sess, cfg, logger)
        else:
            raise RuntimeError("get_request_url in config looks like a placeholder")
    except Exception as e:
        logger.warning("get_request_url failed: %s", e)
        if not cfg.auto_refresh_get_request_url:
            logger.error(
                "Auto-refresh is disabled. Please paste Request URL A (get_request_url) into parser\\unikey_config.json, or enable auto_refresh_get_request_url=true."
            )
            return 3

        logger.info("Trying to auto-refresh get_request_url (methods=%s)...", cfg.auto_refresh_method)
        new_a = try_refresh_get_request_url(sess, cfg, logger)
        if not new_a:
            logger.error("Auto-refresh failed: couldn't obtain get_request_url.")
            # Try a more robust fallback: capture signed /estate/catalog/?... directly from browser.
            signed = try_discover_signed_catalog_url_browser(cfg, logger)
            if signed:
                catalog_url = signed
                logger.info("Using signed catalog url discovered by browser (fallback).")
            else:
                logger.error(
                    "Fix options:\n"
                    "  1) Ensure Google Chrome or Microsoft Edge is installed. Auto-refresh tries Playwright channels 'chrome' and 'msedge'.\n"
                    "  2) Set auto_refresh_browser_headless=false in parser\\unikey_config.json (recommended on a normal PC).\n"
                    "  3) If you want Playwright bundled Chromium, run: .\\.venv\\Scripts\\python -m playwright install chromium\n"
                    "  4) Manual fallback: open Chrome DevTools -> Network -> get_request_url and paste Request URL into parser\\unikey_config.json\n"
                )
                return 3

        cfg.get_request_url = new_a
        if cfg.persist_refreshed_get_request_url:
            persist_get_request_url(args.config, new_a, logger)
        if catalog_url is None:
            try:
                catalog_url = fetch_catalog_url(sess, cfg, logger)
            except Exception as ee:
                logger.error("Failed to fetch catalog url even after refreshing get_request_url: %s", ee)
                return 3

    if not catalog_url:
        logger.error("Unexpected: catalog_url is empty")
        return 3
    logger.info("Catalog url acquired: %s", catalog_url[:120] + ("..." if len(catalog_url) > 120 else ""))

    time.sleep(cfg.sleep_between_requests_sec)

    # 2) Получаем данные каталога (ЖК/дома)
    stage(2, "Получаем get_data: список ЖК/домов и доступные категории")
    get_data = fetch_get_data(sess, catalog_url, cfg, logger)
    available_cats = collect_available_categories(get_data, cfg)
    wanted = cfg.categories or ["flat", "pantry", "parking"]
    resolved = resolve_categories(available_cats, wanted)
    if not resolved:
        # fallback: используем одиночную cfg.category
        resolved = [(str(cfg.category).strip().lower(), str(cfg.category))]

    stage(3, "Фильтруем дома по городу и разрешаем категории (flat/кладовые/гаражи)")

    logger.info("Available categories (city=%s): %s", cfg.target_city_contains, ", ".join(available_cats) if available_cats else "<none>")
    logger.info("Requested categories: %s", ", ".join([str(x) for x in wanted]))
    logger.info("Resolved categories: %s", ", ".join([f"{c}->{a}" for c,a in resolved]))

    resolved_canon = {c for c, _a in resolved}
    missing_requested: List[str] = []
    for w in wanted:
        cw = str(w).strip().lower()
        if cw in ("pantry", "parking") and cw not in resolved_canon:
            missing_requested.append(cw)
    if missing_requested:
        logger.warning(
            "Requested categories not resolved: %s. Available categories: %s. "
            "If the site uses different tokens, put them into config.categories.",
            ", ".join(missing_requested),
            ", ".join(available_cats) if available_cats else "<none>",
        )

    targets_all: List[Tuple[int, str, str, str, str]] = []  # (house_id, house_name, complex_name, canonical, actual)
    for canonical, actual in resolved:
        t = pick_target_houses(get_data, cfg, category=actual)
        for hid, hname, cname in t:
            targets_all.append((hid, hname, cname, canonical, actual))

    # Уберём дубли на случай повторяющихся категорий
    uniq = {}
    for hid, hname, cname, canonical, actual in targets_all:
        key = (hid, canonical, actual)
        if key not in uniq:
            uniq[key] = (hid, hname, cname, canonical, actual)
    targets_all = list(uniq.values())
    if not targets_all:
        logger.error("No target houses found for requested categories. Check city/category filters or config")
        return 1

    logger.info("Target house+category pairs (%d):", len(targets_all))
    for hid, hname, cname, canonical, actual in targets_all:
        logger.info(" - [%s/%s] %s | %s | house_id=%s", canonical, actual, cname, hname, hid)

    # 3) По каждому дому+категории — get_estates и разворачиваем в строки
    rows_all: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    if missing_requested:
        # В строгом режиме это приведёт к exit_code=1 и карантину, чтобы мы не пропустили неполноту данных.
        errors.append({
            "captured_at": captured_at_iso,
            "house_id": None,
            "house_name": None,
            "complex_name": None,
            "lot_type": None,
            "requested_category": None,
            "error": f"Requested categories not resolved: {', '.join(missing_requested)}",
            "stage": "resolve_categories",
        })

    stage(4, f"Собираем лоты: дом×категория ({len(targets_all)} задач)")

    pbar = None
    iter_targets = targets_all
    if tqdm is not None:
        pbar = _t(targets_all, desc=f"Unikey: дом×категория ({cfg.target_city_contains})", total=len(targets_all))
        iter_targets = pbar

    for hid, hname, cname, canonical, actual in iter_targets:
        try:
            time.sleep(cfg.sleep_between_requests_sec)
            j = fetch_get_estates(sess, catalog_url, cfg, hid, logger, category=actual)
            rows = flatten_estates(j, captured_at_iso=captured_at_iso, lot_type=canonical, requested_category=actual, city=cfg.target_city_contains)
            # дополним общими полями комплекса, чтобы не терять
            for r in rows:
                r["complex_name"] = cname
                r["house_public_name"] = hname
            rows_all.extend(rows)
            logger.info("House %s [%s/%s]: rows=%d", hid, canonical, actual, len(rows))
            if pbar is not None:
                try:
                    pbar.set_postfix({"lots": len(rows_all), "errors": len(errors)})
                except Exception:
                    pass
        except Exception as e:
            logger.error("House %s failed: %s", hid, e)
            if pbar is not None:
                try:
                    pbar.set_postfix({"lots": len(rows_all), "errors": len(errors)})
                except Exception:
                    pass
            errors.append({
                "captured_at": captured_at_iso,
                "house_id": hid,
                "house_name": hname,
                "complex_name": cname,
                "lot_type": canonical,
                "requested_category": actual,
                "error": str(e),
            })

    if pbar is not None:
        try:
            pbar.close()
        except Exception:
            pass

    stage(5, "Записываем snapshot и итоговые артефакты")

    # Запись результатов
    out_csv = os.path.join(snapshots_dir, f"{ts}_unikey_snapshot.csv")
    err_csv = os.path.join(errors_dir, f"{ts}_unikey_errors.csv")
    write_csv(out_csv, rows_all)
    write_errors(err_csv, errors)

    print()
    type_counts = {}
    for r in rows_all:
        lt = str(r.get("lot_type") or "unknown")
        type_counts[lt] = type_counts.get(lt, 0) + 1
    print(f"{OK}Done.{RST} rows={len(rows_all)} house_pairs={len(targets_all)} errors={len(errors)} lot_types={type_counts}")
    if rows_all:
        print(f"Snapshot: {out_csv}")
    if errors:
        print(f"Errors:   {err_csv}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
