#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shantary (shantary.ru) MacroCatalog parser.

This parser is adapted from the Unikey parser used in our project.  It uses the
common SberCRM MacroCatalog API (https://api.macro.sbercrm.com) to fetch data
about flats, storerooms and parking places for the city of Khabarovsk.  The
script complies with the requirements of our parsing framework:

 * It writes a CSV snapshot into the appropriate ``output_shantary`` folder
   under ``snapshots_raw/`` with the naming convention
   ``shantary__<city>__YYYYMMDD_HHMMSS.csv``.
 * It produces a ``result.json`` with a status (OK/WARN/FAIL), number of
   retries used, suspect flags, the path to the snapshot and a human‑readable
   message.  For WARN/FAIL it also writes a ``reason.json`` explaining
   precisely what went wrong and how to proceed.
 * It logs progress to both stdout and a log file and shows a soft green
   progress bar so that an operator can see the current stage of the parser.
 * It performs up to three automatic retry passes in case of transient
   failures (e.g. HTTP 429/5xx) with increasing delays and reduced request
   frequency.  Only after these retries fail does it ask the user for
   confirmation to continue (in a real deployment this hook will be wired to
   the launcher for manual intervention).
 * It is resilient to missing or null fields: all dictionary accesses are
   protected with ``dict.get()`` and row processing is wrapped in try/except
   blocks so that a bad record does not kill the entire snapshot.
 * It validates the resulting snapshot against the previous one stored in
   ``snapshots_raw``.  Large drops in row count or spikes in null values are
   detected and reported as suspect flags.

To run this parser directly:

    python shantary_parser.py --config path/to/shantary_config.json

The accompanying ``shantary_config.json`` must contain at least the
``get_request_url`` (signed catalog URL) or enable ``auto_refresh_get_request_url``
and specify a method to obtain it (e.g. via a browser step).  See the
documentation of ``load_config`` for details.

Note: This parser is designed to live in ``parser/shantary/`` inside the
repository.  You can adjust ``DEFAULT_OUT_DIR`` below when placing it
elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from colorama import Fore, Style, init as colorama_init

try:
    # tqdm is optional; if unavailable the parser will still run but without
    # pretty progress bars.
    from tqdm import tqdm
except Exception:
    tqdm = None  # type: ignore

colorama_init()

# ANSI colours for progress bar and log messages
OK = Fore.GREEN
WARN = Fore.YELLOW
ERR = Fore.RED
RST = Style.RESET_ALL

# Default output directory relative to the parser file.  Can be overridden in
# the configuration file.  When running inside the project structure this
# corresponds to parser/shantary/output_shantary.
DEFAULT_OUT_DIR = "output_shantary"


@dataclass
class Config:
    """Configuration for the Shantary parser.

    These values are loaded from the JSON configuration file.  See
    ``load_config`` for parsing details.  Only ``get_request_url`` or
    auto‑refresh settings are strictly required.  Other parameters have
    sensible defaults suitable for Khabarovsk.
    """

    # Domain of the MacroCatalog widget.
    domain: str = "shantary.ru"

    # Signed URL to /estate/request/get_request_url/. If empty, the parser will try
    # to auto-discover it (see auto_refresh_* settings) and/or fall back to the
    # public domain-only endpoint.
    get_request_url: str = ""

    # Auto-refresh policy for get_request_url.
    auto_refresh_get_request_url: bool = True
    # Methods for auto refresh, comma-separated: "probe_api,probe_html,browser".
    # Order matters.
    auto_refresh_method: str = "probe_api,probe_html,browser"
    # Entry URL for the browser discovery step.
    auto_refresh_browser_entry_url: str = "https://shantary.ru/"
    auto_refresh_browser_headless: bool = False
    auto_refresh_browser_slowmo_ms: int = 60
    auto_refresh_browser_timeout_sec: int = 90

    # Interactive assist mode (manual token refresh): keep browser open while user navigates
    auto_refresh_browser_assist_mode: bool = True
    auto_refresh_browser_assist_max_wait_sec: int = 1800

    # If total flats are extremely low, treat as token/access issue and retry with browser assist
    min_flats_retry_threshold: int = 50

    # A small list of pages to probe for embedded MacroCatalog URLs.
    # If empty, defaults will be used.
    probe_urls: Optional[List[str]] = None

    # Whether to persist a newly discovered get_request_url into config JSON.
    persist_refreshed_get_request_url: bool = True

    # City filter: only houses whose complex city contains this substring (case
    # insensitive) are collected.  For the Shantary project we default to
    # "Хабаровск".
    target_city_contains: str = "Хабаровск"
    # Mapping of canonical categories to actual category tokens used by the
    # API.  By default flats use "flat", storerooms use "storageroom" and
    # parking places use "garage".  If the developer changes these names,
    # update them here.  Categories set to an empty string or missing will be
    # skipped.
    categories: Dict[str, str] = field(default_factory=lambda: {
        "flat": "flat",
        "parking": "garage",
    })
    # If true, the parser will error out when a requested category cannot be
    # resolved from get_data.  If false, missing categories will simply be
    # skipped.  Strict mode is recommended to avoid silently ignoring
    # storerooms or parking.
    extra_categories_strict: bool = True
    # Activity filter: "sell" or "rent".  Only "sell" is used for
    # Khabarovsk.
    activity: str = "sell"
    # Optional list of house IDs to limit the parser to specific houses.  If
    # empty or None, all houses in the target city are processed.
    houses: Optional[List[int]] = None
    # Base output directory.  Relative paths are resolved against the parser
    # file location.  See ``DEFAULT_OUT_DIR``.
    out_dir: str = DEFAULT_OUT_DIR
    # HTTP timeout per request in seconds.
    timeout_sec: int = 35
    # Maximum number of retries to use when making HTTP calls.  The parser
    # automatically retries GET and POST calls on transient failures like 429
    # or 5xx.
    retries: int = 3
    # Initial sleep between HTTP requests in seconds.  On each automatic
    # retry pass this value is multiplied by ``sleep_backoff_factor``.
    sleep_between_requests_sec: float = 0.2
    # Multiplier for sleep_between_requests_sec on each retry pass.
    sleep_backoff_factor: float = 2.0
    # Maximum number of retry passes (not per request).  After this many
    # passes, if errors remain the parser will ask for manual confirmation.
    max_retry_passes: int = 3

    # Estates pagination (some MacroCatalog deployments return only a small
    # first page by default). The parser tries to auto-detect pagination and
    # will fetch all pages.
    estates_page_size: int = 500
    estates_max_pages: int = 300
    estates_small_result_threshold: int = 12
    # Path to the configuration file; stored here for persistence when
    # rewriting get_request_url.  Filled in by ``load_config``.
    config_path: str = ""


def _now_ts() -> str:
    """Return current timestamp formatted as YYYYMMDD_HHMMSS."""
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> None:
    """Ensure that ``path`` exists as a directory."""
    os.makedirs(path, exist_ok=True)


def _t(it: Iterable[Any], desc: str):
    """Wrap an iterable in a tqdm progress bar when available."""
    if tqdm is None:
        return it
    # Use a soft green bar colour; bar_format ensures the colour resets at the
    # end of the bar.  ``colour`` parameter is supported in recent tqdm.
    return tqdm(it, desc=desc, bar_format=f"{OK}{{l_bar}}{{bar}}{{r_bar}}{RST}", colour="#7ED957")


def load_config(path: str) -> Config:
    """Load configuration from ``path`` and return a populated Config object.

    The configuration file is a JSON object.  Unknown keys are ignored.  If
    ``get_request_url`` is empty and auto refresh is disabled, the parser
    requires a valid URL to proceed.  The returned Config includes
    ``config_path`` so that refreshed values can be persisted later.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = Config()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.config_path = os.path.abspath(path)
    # Normalise get_request_url: strip whitespace
    cfg.get_request_url = str(cfg.get_request_url or "").strip()
    return cfg


def setup_logging(ts: str, base_dir: str) -> logging.Logger:
    """Configure logging to file and stdout, returning a logger."""
    log_dir = os.path.join(base_dir, "logs")
    _ensure_dir(log_dir)
    log_path = os.path.join(log_dir, f"run_{ts}.log")
    logger = logging.getLogger("shantary_parser")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Logging to %s", log_path)
    return logger


def http_json(session: requests.Session, method: str, url: str, *, json_body: Optional[dict] = None,
              timeout: int, retries: int, logger: logging.Logger) -> dict:
    """Perform an HTTP request and parse JSON with retry logic.

    Retries on status codes 429 and >=500.  Raises on other 4xx errors or if
    JSON decoding fails.  The ``retries`` parameter controls the number of
    attempts; backoff is handled outside of this function.
    """
    for attempt in range(1, retries + 1):
        try:
            if method.upper() == "GET":
                resp = session.get(url, timeout=timeout)
            else:
                resp = session.post(url, json=json_body, timeout=timeout)
            status = resp.status_code
            if status == 429 or status >= 500:
                logger.warning("HTTP %s on %s (attempt %d/%d)", status, url, attempt, retries)
                # Let loop retry
            elif status >= 400:
                # Hard error: do not retry on 4xx except 429
                raise RuntimeError(f"HTTP {status} on {url}: {resp.text[:200]}")
            else:
                return resp.json()
        except Exception as e:
            last_err = e
        time.sleep(1.0 + random.random())
    raise RuntimeError(f"Failed to {method} {url}: {last_err}")


def _looks_like_placeholder(url: str) -> bool:
    """Treat empty or template values as placeholders."""
    u = (url or "").strip()
    return (not u) or ("PASTE" in u.upper()) or ("GET_REQUEST_URL" in u.upper())


def _extract_get_request_url_from_text(text: str) -> Optional[str]:
    """Heuristically find MacroCatalog get_request_url inside HTML/JS."""
    if not text:
        return None

    # Full URL
    m = re.search(r"https://api\\.macro\\.sbercrm\\.com/estate/request/get_request_url/\\?[^\"']+", text)
    if m:
        return m.group(0).rstrip("\"' )];,\n\r\t")

    # Relative path
    m = re.search(r"/estate/request/get_request_url/\\?[^\"']+", text)
    if m:
        return ("https://api.macro.sbercrm.com" + m.group(0)).rstrip("\"' )];,\n\r\t")

    return None


def _extract_signed_catalog_url_from_get_request_resp(j: dict) -> Optional[str]:
    """Return signed /estate/catalog/ URL from get_request_url response."""
    # Common shapes we have seen across MacroCatalog deployments
    for key in ("url", "catalog_url"):
        v = j.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    data = j.get("data") or {}
    if isinstance(data, dict):
        v = data.get("url") or data.get("catalog_url")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def try_refresh_get_request_url_probe_api(session: requests.Session, cfg: Config, logger: logging.Logger) -> Optional[str]:
    """Try the public domain-only endpoint to obtain a signed catalog URL."""
    candidate = f"https://api.macro.sbercrm.com/estate/request/get_request_url/?domain={cfg.domain}"
    try:
        j = http_json(session, "GET", candidate, timeout=min(25, cfg.timeout_sec), retries=1, logger=logger)
        signed = _extract_signed_catalog_url_from_get_request_resp(j)
        if signed:
            cfg.get_request_url = candidate
            logger.info("Auto-refresh(probe_api): domain endpoint works: %s", candidate)
            return signed
    except Exception as e:
        logger.warning("Auto-refresh(probe_api) failed: %s err=%s", candidate, e)
    return None


def try_refresh_get_request_url_probe_html(session: requests.Session, cfg: Config, logger: logging.Logger) -> Optional[str]:
    """Try to find an embedded get_request_url in site HTML/JS."""
    urls = cfg.probe_urls or [
        f"https://{cfg.domain}/",
        f"https://{cfg.domain}/catalog/",
        f"https://{cfg.domain}/catalog/#/macrocatalog/complexes/list?studio=null&category=flat&activity={cfg.activity}",
        f"https://{cfg.domain}/catalog/#/macrocatalog/complexes/list?studio=null&category=garage&activity={cfg.activity}",
    ]
    html_headers = {
        "user-agent": DEFAULT_HEADERS.get("user-agent", "Mozilla/5.0"),
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    for u in urls:
        try:
            r = session.get(u, headers=html_headers, timeout=min(25, cfg.timeout_sec))
            if r.status_code >= 400:
                continue
            guess = _extract_get_request_url_from_text(r.text)
            if guess:
                cfg.get_request_url = guess
                logger.info("Auto-refresh(probe_html): found get_request_url in %s", u)
                return None  # we only discovered get_request_url; caller will use fetch_catalog_url
        except Exception as e:
            logger.warning("Auto-refresh(probe_html) failed: %s err=%s", u, e)
            continue
    return None


@dataclass
class BrowserDiscovery:
    request_url: Optional[str] = None
    signed_catalog_url: Optional[str] = None


def _discover_macro_urls_via_browser(cfg: Config, logger: logging.Logger) -> BrowserDiscovery:
    """Discover MacroCatalog URLs using Playwright network interception.

    We try to capture:
      - request_url: /estate/request/get_request_url/?domain=...&check=...
      - signed_catalog_url: /estate/catalog/?domain=...&time=...&token=...

    The browser step is used only if probe methods fail.
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:
        logger.warning("Auto-refresh(browser) unavailable (Playwright missing). err=%s", e)
        return BrowserDiscovery()

    entry_urls: List[str] = []
    if (cfg.auto_refresh_browser_entry_url or "").strip():
        entry_urls.append(cfg.auto_refresh_browser_entry_url.strip())
    # Ensure both flats and parking routes are visited (some deployments only fire Macro requests per tab).
    entry_urls.extend([
        f"https://{cfg.domain}/catalog/#/macrocatalog/complexes/list?studio=null&category=flat&activity={cfg.activity}",
        f"https://{cfg.domain}/catalog/#/macrocatalog/complexes/list?studio=null&category=garage&activity={cfg.activity}",
    ])
    # de-dup
    seen = set()
    entry_urls = [u for u in entry_urls if not (u in seen or seen.add(u))]

    needle_a = "/estate/request/get_request_url/"
    needle_catalog = "/estate/catalog/"
    found_a: Optional[str] = None
    found_signed: Optional[str] = None

    headless = bool(cfg.auto_refresh_browser_headless)
    slow_mo = int(cfg.auto_refresh_browser_slowmo_ms) if not headless else 0
    hard_timeout_ms = int(max(15, cfg.auto_refresh_browser_timeout_sec)) * 1000

    def on_request(req) -> None:
        nonlocal found_a, found_signed
        try:
            u = req.url
            if (not found_a) and (needle_a in u):
                found_a = u
                logger.info("Auto-refresh(browser): captured request_url (A)")
            if (not found_signed) and (needle_catalog in u) and ("token=" in u):
                found_signed = u
                logger.info("Auto-refresh(browser): captured signed catalog URL")
        except Exception:
            return

    with sync_playwright() as p:
        try:

            # Prefer system Chrome so we don't depend on Playwright-downloaded browsers.

            browser = p.chromium.launch(channel="chrome", headless=headless, slow_mo=slow_mo)

        except Exception as e:

            logger.warning("Auto-refresh(browser): failed to launch system Chrome via channel='chrome' (%s). Falling back to Playwright chromium.", e)

            browser = p.chromium.launch(headless=headless, slow_mo=slow_mo)

        context = browser.new_context()
        page = context.new_page()
        page.on("request", on_request)
        # Open one entry url, then allow the operator to interact while we capture Macro requests.
        opened_any = False
        for u in entry_urls:
            try:
                logger.info("Auto-refresh(browser): opening %s", u)
                page.goto(u, wait_until="domcontentloaded", timeout=hard_timeout_ms)
                opened_any = True
                break
            except Exception as e:
                logger.warning("Auto-refresh(browser): open failed %s err=%s", u, e)
                continue

        if opened_any:
            logger.info("Auto-refresh(browser): waiting for Macro requests (you may interact with the page)...")
            start_ms = time.time() * 1000.0
            while True:
                if found_a and found_signed:
                    break
                if found_a and (time.time() * 1000.0 - start_ms) > 3500:
                    # request_url is enough; signed url may not always appear
                    break
                if (time.time() * 1000.0 - start_ms) > hard_timeout_ms:
                    break
                page.wait_for_timeout(250)

        # Final wait: sometimes the signed URL arrives after first paint
        try:
            page.wait_for_timeout(2500)
        except Exception:
            pass

        context.close()
        browser.close()

    return BrowserDiscovery(request_url=found_a, signed_catalog_url=found_signed)


def _persist_get_request_url(cfg: Config, logger: logging.Logger) -> None:
    if not cfg.persist_refreshed_get_request_url:
        return
    if not cfg.config_path:
        return
    try:
        with open(cfg.config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["get_request_url"] = cfg.get_request_url
        with open(cfg.config_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        logger.info("Persisted get_request_url into %s", cfg.config_path)
    except Exception as e:
        logger.warning("Failed to persist get_request_url: %s", e)


def ensure_catalog_url(session: requests.Session, cfg: Config, logger: logging.Logger) -> str:
    """Return a working signed catalog URL, auto-discovering if needed."""
    # 1) If we have get_request_url, try it
    if not _looks_like_placeholder(cfg.get_request_url):
        j = http_json(session, "GET", cfg.get_request_url, timeout=cfg.timeout_sec, retries=cfg.retries, logger=logger)
        signed = _extract_signed_catalog_url_from_get_request_resp(j)
        if signed:
            return signed

    # 2) Auto refresh
    if not cfg.auto_refresh_get_request_url:
        raise RuntimeError("get_request_url is empty; please supply it in config")

    methods = [m.strip() for m in (cfg.auto_refresh_method or "").split(",") if m.strip()]
    if not methods:
        methods = ["probe_api", "probe_html", "browser"]

    # Backward-compatible aliases:
    #  - "probe" means try both probe_api and probe_html.
    expanded: List[str] = []
    for m in methods:
        if m == "probe":
            expanded.extend(["probe_api", "probe_html"])
        else:
            expanded.append(m)
    methods = expanded

    # Backward-compatible aliases:
    #  - "probe" means "probe_api" + "probe_html" (same spirit as Unikey config)
    expanded: List[str] = []
    for m in methods:
        if m == "probe":
            expanded.extend(["probe_api", "probe_html"])
        else:
            expanded.append(m)
    methods = expanded

    for m in methods:
        if m == "probe_api":
            signed = try_refresh_get_request_url_probe_api(session, cfg, logger)
            if signed:
                _persist_get_request_url(cfg, logger)
                return signed

        elif m == "probe_html":
            # This method only updates cfg.get_request_url if found.
            try_refresh_get_request_url_probe_html(session, cfg, logger)
            if not _looks_like_placeholder(cfg.get_request_url):
                try:
                    j = http_json(session, "GET", cfg.get_request_url, timeout=cfg.timeout_sec, retries=cfg.retries, logger=logger)
                    signed = _extract_signed_catalog_url_from_get_request_resp(j)
                    if signed:
                        _persist_get_request_url(cfg, logger)
                        return signed
                except Exception as e:
                    logger.warning("Auto-refresh(probe_html) candidate failed: %s", e)

        elif m == "browser":
            disc = _discover_macro_urls_via_browser(cfg, logger)
            if disc.request_url:
                cfg.get_request_url = disc.request_url
                _persist_get_request_url(cfg, logger)
            if disc.signed_catalog_url:
                return disc.signed_catalog_url
            if disc.request_url:
                j = http_json(session, "GET", disc.request_url, timeout=cfg.timeout_sec, retries=cfg.retries, logger=logger)
                signed = _extract_signed_catalog_url_from_get_request_resp(j)
                if signed:
                    return signed

    raise RuntimeError(
        "Failed to auto-discover get_request_url. "
        "Open the catalog in a browser (flat or garage), then retry. "
        "If Playwright is not installed, run: .venv\\Scripts\\python -m pip install playwright && .venv\\Scripts\\playwright install chromium"
    )


def fetch_catalog_url(session: requests.Session, cfg: Config, logger: logging.Logger) -> str:
    """Use the signed get_request_url to obtain the signed catalog URL.

    The get_request_url endpoint returns a JSON with ``catalog_url`` and
    sometimes ``token``.  The returned URL is used to post get_data and
    get_estates requests.  Raises if the response indicates failure.
    """
    if not cfg.get_request_url:
        raise RuntimeError("get_request_url is empty")

    data = http_json(
        session,
        "GET",
        cfg.get_request_url,
        timeout=cfg.timeout_sec,
        retries=cfg.retries,
        logger=logger,
    )

    # Some deployments return {success:false,...} on invalid check/token.
    if isinstance(data, dict) and data.get("success") is False:
        msg = data.get("message") or data.get("error") or data.get("errorMessage") or str(data)[:300]
        raise RuntimeError(f"get_request_url returned success=false: {msg}")

    # Different MacroCatalog builds use different keys.
    catalog_url = (
        (data.get("url") if isinstance(data, dict) else None)
        or (data.get("catalog_url") if isinstance(data, dict) else None)
        or (data.get("data", {}).get("url") if isinstance(data, dict) else None)
        or (data.get("data", {}).get("catalog_url") if isinstance(data, dict) else None)
    )
    if not catalog_url:
        keys = list(data.keys()) if isinstance(data, dict) else []
        raise RuntimeError(f"Catalog URL missing in get_request_url response (keys={keys})")
    return str(catalog_url)


def _looks_like_placeholder(url: str) -> bool:
    u = (url or "").strip()
    return (not u) or ("PASTE_HERE" in u) or ("GET_REQUEST_URL" in u)


def _extract_get_request_url_from_text(text: str) -> Optional[str]:
    """Best-effort extractor for MacroCatalog get_request_url from HTML/JS."""
    if not text:
        return None

    m = re.search(r"https://api\\.macro\\.sbercrm\\.com/estate/request/get_request_url/\\?[^\"']+", text)
    if m:
        return m.group(0).rstrip("\"' )];,\n\r\t")

    m = re.search(r"/estate/request/get_request_url/\\?[^\"']+", text)
    if m:
        return ("https://api.macro.sbercrm.com" + m.group(0)).rstrip("\"' )];,\n\r\t")

    return None


def _default_probe_urls(cfg: Config) -> List[str]:
    # Include both flat and garage pages to satisfy the requirement "for flats and parking".
    return [
        cfg.auto_refresh_browser_entry_url or "https://shantary.ru/",
        "https://shantary.ru/",
        "https://shantary.ru/#/macrocatalog/complexes/list?studio=null&category=flat&activity=sell",
        "https://shantary.ru/#/macrocatalog/complexes/list?studio=null&category=garage&activity=sell",
    ]


def try_refresh_get_request_url_probe_html(session: requests.Session, cfg: Config, logger: logging.Logger) -> Optional[str]:
    """Try to discover get_request_url by scanning HTML/JS."""
    urls = cfg.probe_urls or _default_probe_urls(cfg)
    html_headers = {
        "user-agent": DEFAULT_HEADERS.get("user-agent", "Mozilla/5.0"),
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    for u in urls:
        try:
            r = session.get(u, headers=html_headers, timeout=min(20, cfg.timeout_sec))
            if r.status_code >= 400:
                continue
            guess = _extract_get_request_url_from_text(r.text)
            if guess:
                logger.info("Auto-refresh(probe_html): found get_request_url in %s", u)
                return guess
        except Exception as e:
            logger.warning("Auto-refresh(probe_html) failed: %s err=%s", u, e)
            continue
    return None


def try_refresh_get_request_url_probe_api(cfg: Config, logger: logging.Logger) -> Optional[str]:
    """Try the public domain-only endpoint as a usable get_request_url.

    On many MacroCatalog deployments the endpoint below works and already returns the
    signed catalog URL. If it works, we can use it as cfg.get_request_url directly.
    """
    candidate = f"https://api.macro.sbercrm.com/estate/request/get_request_url/?domain={cfg.domain}"
    logger.info("Auto-refresh(probe_api): trying %s", candidate)
    return candidate


@dataclass
class BrowserDiscovery:
    found_a: Optional[str] = None
    found_signed_catalog: Optional[str] = None
    used_entry: Optional[str] = None
    error: Optional[str] = None




def _discover_macro_urls_via_browser(cfg: Config, logger: logging.Logger) -> BrowserDiscovery:
    """Discover MacroCatalog URLs via Playwright network interception.

    Why this matters for Shantary:
    - The catalog can be gated by a lead-form / cookies.
    - A token captured too early may return only a few promo lots.

    In *assist mode* we keep the browser open and let the user pass the gate
    and open the real chessboard. The parser captures the **latest**
    /estate/request/get_request_url/ and/or signed /estate/catalog/ URL.
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:
        return BrowserDiscovery(error=f"Playwright not installed: {e}")

    entry_urls: list[str] = []
    if (cfg.auto_refresh_browser_entry_url or "").strip():
        entry_urls.append(cfg.auto_refresh_browser_entry_url.strip())

    # Some MacroCatalog deployments only fire requests when a specific tab is opened.
    entry_urls.extend([
        "https://shantary.ru/#/macrocatalog/complexes/list?studio=null&category=flat&activity=sell",
        "https://shantary.ru/#/macrocatalog/complexes/list?studio=null&category=garage&activity=sell",
    ])

    # De-dup
    seen = set()
    entry_urls = [u for u in entry_urls if not (u in seen or seen.add(u))]

    needle_a = "/estate/request/get_request_url/"
    needle_catalog = "/estate/catalog/"

    found_a: str | None = None
    found_signed: str | None = None
    used_entry: str | None = None

    headless = bool(getattr(cfg, "auto_refresh_browser_headless", False))
    slow_mo = int(getattr(cfg, "auto_refresh_browser_slowmo_ms", 120) or 0) if not headless else 0
    timeout_ms = int(max(15, int(getattr(cfg, "auto_refresh_browser_timeout_sec", 90) or 90))) * 1000

    def is_domain_match(url: str) -> bool:
        # We accept either host match or query param domain=...
        return (cfg.domain in url) or (f"domain={cfg.domain}" in url)

    def is_signed_catalog(url: str) -> bool:
        return (needle_catalog in url) and ("token=" in url) and is_domain_match(url)

    def on_request(req) -> None:
        nonlocal found_a, found_signed
        try:
            u = req.url
            if (needle_a in u) and is_domain_match(u):
                found_a = u  # keep the latest
            if is_signed_catalog(u):
                found_signed = u  # keep the latest
        except Exception:
            return

    def on_response(resp) -> None:
        nonlocal found_a, found_signed
        try:
            u = resp.url
            if (needle_a in u) and is_domain_match(u):
                found_a = u
                # Many deployments return JSON with a direct signed catalog URL
                try:
                    j = resp.json()
                    su = None
                    if isinstance(j, dict):
                        su = j.get("url") or (j.get("data") or {}).get("url")
                    if isinstance(su, str) and is_signed_catalog(su):
                        found_signed = su
                except Exception:
                    pass
        except Exception:
            return

    try:
        with sync_playwright() as p:
            # Prefer system Chrome so we don't depend on Playwright-downloaded browsers.
            browser = None
            try:
                browser = p.chromium.launch(channel="chrome", headless=headless, slow_mo=slow_mo)
            except Exception as e:
                logger.warning("Auto-refresh(browser): failed to launch system Chrome via channel='chrome' (%s). Falling back to Playwright chromium.", e)
                browser = p.chromium.launch(headless=headless, slow_mo=slow_mo)

            context = browser.new_context()
            page = context.new_page()
            page.on("request", on_request)
            page.on("response", on_response)

            assist_mode = bool(getattr(cfg, "auto_refresh_browser_assist_mode", True))

            if assist_mode:
                start_url = (entry_urls[0] if entry_urls else "https://shantary.ru/")
                used_entry = start_url
                try:
                    page.goto(start_url, wait_until="domcontentloaded", timeout=timeout_ms)
                except Exception:
                    # still proceed to wait
                    pass

                logger.info("Manual token refresh: Chrome window is open.")
                logger.info("Please do the following IN THE OPENED BROWSER:")
                logger.info("  1) If a form asking for phone/data appears: fill it (can use test data) and submit to unlock the catalog")
                logger.info("  2) Open the apartment selection / catalog and then open ANY house chessboard (floors/grid)")
                logger.info("  3) Close popups/banners manually if they block clicks — parser will keep waiting")
                logger.info("As soon as the site requests /estate/request/get_request_url, the parser will capture it and continue.")

                max_wait_sec = int(getattr(cfg, "auto_refresh_browser_assist_max_wait_sec", 1800) or 1800)
                poll_ms = 250
                started = __import__("time").time()
                last_ping = 0.0
                while not (found_a or found_signed):
                    if max_wait_sec > 0 and (__import__("time").time() - started) > max_wait_sec:
                        break
                    page.wait_for_timeout(poll_ms)
                    now = __import__("time").time()
                    if (now - last_ping) > 20:
                        logger.info("Waiting for token... (%ss elapsed)", int(now - started))
                        last_ping = now

            else:
                # Auto mode: visit a few URLs and wait briefly
                wait_sec = int(getattr(cfg, "auto_refresh_browser_wait_sec", 45) or 45)
                poll_ms = 250
                max_polls = max(1, int((wait_sec * 1000) / poll_ms))
                for u in entry_urls:
                    used_entry = u
                    logger.info("Auto-refresh(browser): opening %s", u)
                    try:
                        page.goto(u, wait_until="domcontentloaded", timeout=timeout_ms)
                    except Exception:
                        pass
                    for _ in range(max_polls):
                        if found_a or found_signed:
                            break
                        page.wait_for_timeout(poll_ms)
                    if found_a or found_signed:
                        break

            context.close()
            browser.close()

    except Exception as e:
        return BrowserDiscovery(used_entry=used_entry, error=str(e))

    return BrowserDiscovery(found_a=found_a, found_signed_catalog=found_signed, used_entry=used_entry)


def persist_get_request_url(cfg: Config, new_url: str, logger: logging.Logger) -> None:
    if not cfg.persist_refreshed_get_request_url:
        return
    if not cfg.config_path:
        return
    try:
        with open(cfg.config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["get_request_url"] = new_url
        with open(cfg.config_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        logger.info("Persisted refreshed get_request_url into %s", cfg.config_path)
    except Exception as e:
        logger.warning("Failed to persist get_request_url: %s", e)


def ensure_catalog_url(session: requests.Session, cfg: Config, logger: logging.Logger) -> str:
    """Ensure we have a valid signed catalog URL.

    - If cfg.get_request_url is usable -> fetch_catalog_url.
    - If empty/placeholder and auto_refresh enabled:
        * probe_api: try domain-only endpoint as get_request_url
        * probe_html: scan site HTML/JS
        * browser: capture the real get_request_url via Playwright
    """
    # 1) If user provided something, try it first.
    if cfg.get_request_url and (not _looks_like_placeholder(cfg.get_request_url)):
        return fetch_catalog_url(session, cfg, logger)

    if not cfg.auto_refresh_get_request_url:
        raise RuntimeError("get_request_url is empty; auto_refresh_get_request_url=false")

    methods = [m.strip() for m in (cfg.auto_refresh_method or "").split(",") if m.strip()]
    if not methods:
        methods = ["probe_api", "probe_html", "browser"]

    # Backward-compatible alias: "probe" -> (probe_api + probe_html)
    expanded: List[str] = []
    for m in methods:
        if m == "probe":
            expanded.extend(["probe_api", "probe_html"])
        else:
            expanded.append(m)
    methods = expanded

    last_err: Optional[str] = None

    for m in methods:
        try:
            if m == "probe_api":
                candidate = try_refresh_get_request_url_probe_api(cfg, logger)
                if candidate:
                    cfg.get_request_url = candidate
                    try:
                        url = fetch_catalog_url(session, cfg, logger)
                        persist_get_request_url(cfg, cfg.get_request_url, logger)
                        logger.info("Auto-refresh(probe_api) OK")
                        return url
                    except Exception as e:
                        last_err = str(e)
                        logger.warning("Auto-refresh(probe_api) failed: %s", e)
                        cfg.get_request_url = ""

            elif m == "probe_html":
                guess = try_refresh_get_request_url_probe_html(session, cfg, logger)
                if guess:
                    cfg.get_request_url = guess
                    try:
                        url = fetch_catalog_url(session, cfg, logger)
                        persist_get_request_url(cfg, cfg.get_request_url, logger)
                        logger.info("Auto-refresh(probe_html) OK")
                        return url
                    except Exception as e:
                        last_err = str(e)
                        logger.warning("Auto-refresh(probe_html) failed: %s", e)
                        cfg.get_request_url = ""

            elif m == "browser":
                disc = _discover_macro_urls_via_browser(cfg, logger)
                if disc.error:
                    last_err = disc.error
                    logger.warning("Auto-refresh(browser) failed: %s", disc.error)
                    continue

                if disc.found_signed_catalog:
                    # Best case: we can use signed catalog directly.
                    if disc.found_a:
                        cfg.get_request_url = disc.found_a
                        persist_get_request_url(cfg, cfg.get_request_url, logger)
                    logger.info("Auto-refresh(browser) OK (signed catalog captured)")
                    return disc.found_signed_catalog

                if disc.found_a:
                    cfg.get_request_url = disc.found_a
                    try:
                        url = fetch_catalog_url(session, cfg, logger)
                        persist_get_request_url(cfg, cfg.get_request_url, logger)
                        logger.info("Auto-refresh(browser) OK")
                        return url
                    except Exception as e:
                        last_err = str(e)
                        logger.warning("Auto-refresh(browser) A captured but unusable: %s", e)
                        cfg.get_request_url = ""

        except Exception as e:
            last_err = str(e)
            continue

    raise RuntimeError(
        "get_request_url is empty/invalid and auto-refresh failed. "
        "If Playwright is missing browsers, run: .venv\\Scripts\\python -m playwright install chromium. "
        f"Last error: {last_err}"
    )


def fetch_get_data(session: requests.Session, catalog_url: str, cfg: Config,
                    logger: logging.Logger) -> dict:
    """POST to /estate/catalog/ with action=get_data and return JSON."""
    payload = {
        "action": "get_data",
        "data": {"cabinetMode": False},
        "auth_token": None,
        "locale": None,
    }
    j = http_json(session, "POST", catalog_url, json_body=payload, timeout=cfg.timeout_sec,
                  retries=cfg.retries, logger=logger)
    if not j.get("success", False):
        raise RuntimeError(f"get_data returned success=false: {str(j)[:300]}")
    return j


def fetch_get_estates(session: requests.Session, catalog_url: str, cfg: Config,
                       house_id: int, category: str, logger: logging.Logger) -> dict:
    """POST to /estate/catalog/ with action=get_estates.

    IMPORTANT: Some MacroCatalog deployments return only a small first page
    (e.g., 4 items) unless pagination parameters are provided.  This function
    auto-detects such cases and attempts to fetch *all* pages.
    """

    def _post(data_overrides: Dict[str, Any]) -> dict:
        data = {
            "house_id": house_id,
            "category": category,
            "cabinetMode": False,
        }
        # Some deployments are sensitive to activity; harmless when ignored.
        if getattr(cfg, "activity", None):
            data["activity"] = cfg.activity
        data.update(data_overrides)
        payload = {"action": "get_estates", "data": data, "auth_token": None, "locale": None}
        j = http_json(session, "POST", catalog_url, json_body=payload, timeout=cfg.timeout_sec,
                      retries=cfg.retries, logger=logger)
        if not j.get("success", False):
            raise RuntimeError(
                f"get_estates returned success=false house_id={house_id} category={category}: {str(j)[:300]}"
            )
        return j

    def _extract_estates(j: dict) -> List[dict]:
        # Most common: top-level "estates". Keep fallback keys just in case.
        estates = j.get("estates")
        if isinstance(estates, list):
            return estates
        estates = (j.get("data") or {}).get("estates")
        if isinstance(estates, list):
            return estates
        estates = (j.get("result") or {}).get("estates")
        if isinstance(estates, list):
            return estates
        return []

    def _extract_total(j: dict) -> Optional[int]:
        # Try a handful of common total/count locations.
        for key in ("total", "count", "total_count"):
            v = j.get(key)
            if isinstance(v, int) and v >= 0:
                return v
        data = j.get("data") or {}
        for key in ("total", "count", "total_count"):
            v = data.get(key)
            if isinstance(v, int) and v >= 0:
                return v
        pag = j.get("pagination") or data.get("pagination") or {}
        v = pag.get("total")
        if isinstance(v, int) and v >= 0:
            return v
        return None

    # 1) Base request (no pagination params)
    j0 = _post({})
    estates0 = _extract_estates(j0)
    total0 = _extract_total(j0)

    # If response already looks complete, return as-is.
    if total0 is not None and len(estates0) >= total0:
        return {**j0, "estates": estates0}

    # 2) Decide whether we need to paginate / increase page size
    need_more = False
    if total0 is not None and total0 > len(estates0):
        need_more = True
    # Heuristic: suspiciously small first page (common value: 4)
    if len(estates0) <= cfg.estates_small_result_threshold:
        need_more = True

    if not need_more:
        return {**j0, "estates": estates0}

    page_size = int(getattr(cfg, "estates_page_size", 500) or 500)
    max_pages = int(getattr(cfg, "estates_max_pages", 300) or 300)

    # 3) Probe which pagination parameters are supported.
    probes: List[Tuple[str, Dict[str, Any]]] = [
        ("offset_limit", {"offset": 0, "limit": page_size}),
        ("page_per_page", {"page": 1, "per_page": page_size}),
        ("page_pageSize", {"page": 1, "pageSize": page_size}),
        ("pagination_obj", {"pagination": {"offset": 0, "limit": page_size}}),
    ]
    best_mode = None
    best_first: List[dict] = estates0
    best_total = total0
    best_payload: Dict[str, Any] = {}
    for mode, extra in probes:
        try:
            jp = _post(extra)
            ep = _extract_estates(jp)
            tp = _extract_total(jp)
            if len(ep) > len(best_first):
                best_mode = mode
                best_first = ep
                best_total = tp
                best_payload = extra
        except Exception as e:
            logger.info("Pagination probe '%s' not supported or failed: %s", mode, e)

    if best_mode is None:
        # Nothing improved; return what we have but log warning.
        logger.warning(
            "get_estates seems paginated but pagination probes didn't improve results; returning %d items",
            len(estates0),
        )
        return {**j0, "estates": estates0}

    # 4) Fetch all pages using the chosen mode.
    collected: List[dict] = []
    seen: set = set()

    def _add(items: List[dict]) -> int:
        added = 0
        for it in items:
            # Prefer stable id if present; fallback to tuple of key fields.
            uid = it.get("id") or it.get("flat_item_id") or it.get("estate_id")
            if uid is None:
                uid = json.dumps(it, ensure_ascii=False, sort_keys=True)
            if uid in seen:
                continue
            seen.add(uid)
            collected.append(it)
            added += 1
        return added

    _add(best_first)
    # Determine total target
    total_target = best_total
    if total_target is None:
        total_target = total0

    # Iterate
    for i in range(2, max_pages + 1):
        if total_target is not None and len(collected) >= total_target:
            break

        if best_mode == "offset_limit":
            offset = (i - 1) * page_size
            extra = {"offset": offset, "limit": page_size}
        elif best_mode == "page_per_page":
            extra = {"page": i, "per_page": page_size}
        elif best_mode == "page_pageSize":
            extra = {"page": i, "pageSize": page_size}
        else:
            # pagination_obj
            offset = (i - 1) * page_size
            extra = {"pagination": {"offset": offset, "limit": page_size}}

        try:
            ji = _post(extra)
            ei = _extract_estates(ji)
            if not ei:
                break
            added = _add(ei)
            if added == 0:
                # No progress -> stop to avoid infinite loop
                break
            # Update total if we discover it later
            ti = _extract_total(ji)
            if isinstance(ti, int) and ti >= 0:
                total_target = ti
        except Exception as e:
            logger.warning("Pagination page %d failed (%s): %s", i, best_mode, e)
            break
        # be polite between pages
        time.sleep(min(1.0, cfg.sleep_between_requests_sec) + random.uniform(0.0, 0.5))

    if len(collected) <= cfg.estates_small_result_threshold:
        logger.warning(
            "Suspiciously small estates count after pagination (house_id=%s category=%s): %d",
            house_id, category, len(collected),
        )

    # Return a dict compatible with downstream flattening
    out = dict(j0)
    out["estates"] = collected
    return out


def pick_houses(get_data: dict, cfg: Config, category_token: str,
                logger: logging.Logger) -> List[Tuple[int, str, str]]:
    """Return list of (house_id, house_name, complex_name) filtered by city and category."""
    houses = get_data.get("houses") or []
    complexes = {c.get("id"): c for c in (get_data.get("complexes") or [])}
    allowed_complex_ids: set[int] = set()
    for cid, complex_data in complexes.items():
        city = (complex_data.get("city") or "").lower()
        if cfg.target_city_contains.lower() in city:
            allowed_complex_ids.add(cid)
    result: List[Tuple[int, str, str]] = []
    for h in houses:
        hid = h.get("id")
        if not isinstance(hid, int):
            continue
        if cfg.houses and hid not in cfg.houses:
            continue
        cid = h.get("complex_id")
        if cid not in allowed_complex_ids:
            continue
        categories = h.get("categories") or []
        if category_token not in categories:
            continue
        # normalise house and complex names (remove HTML and nbsp)
        def clean(s: str) -> str:
            s = re.sub(r"<[^>]+>", "", s)
            return s.replace("\xa0", " ").replace("&nbsp;", " ").strip()
        house_name = clean(str(h.get("public_name") or h.get("name") or hid))
        complex_name = clean(str(h.get("complex_name") or complexes.get(cid, {}).get("name") or cid))
        result.append((hid, house_name, complex_name))
    # stable sort by complex then house
    result.sort(key=lambda x: (x[2], x[1], x[0]))
    return result


def flatten_estates(get_estates: dict, *, captured_at_iso: str, developer_key: str,
                    developer_name: str, category: str, source: str, city: str,
                    requested_category: str) -> List[Dict[str, Any]]:
    """Разворачивает ответ action=get_estates в строки (1 строка = 1 лот).

    Для MacroCatalog структура обычно: estates -> entrances -> floors -> items.
    Мы используем ту же схему, что и в Unikey, чтобы не ломаться от мелких изменений.
    """
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
                        continue
                    if not item_id:
                        continue

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
                        "developer_key": developer_key,
                        "developer": developer_name,
                        "city": city,
                        "source": source,
                        "lot_type": category,
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


def compute_snapshot_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute simple statistics for a list of row dicts.

    Returns a dict containing row count and null rates for price, area and rooms.
    If the corresponding key is missing, it is treated as None.  Median
    computations ignore None and non-numeric values.  Used for validation.
    """
    total = len(rows)
    def null_rate(key: str) -> float:
        if total == 0:
            return 0.0
        nulls = 0
        for r in rows:
            v = r.get(key)
            if v in (None, "", [], {}):
                nulls += 1
        return nulls / total
    def safe_numeric(key: str) -> List[float]:
        vals = []
        for r in rows:
            v = r.get(key)
            try:
                vals.append(float(str(v).replace(" ", "").replace(",", ".")))
            except Exception:
                continue
        return vals
    metrics = {
        "rows": total,
        "price_null_rate": null_rate("price"),
        "area_null_rate": null_rate("area_m2"),
        "rooms_null_rate": null_rate("rooms"),
    }
    # compute quantiles for area if available
    areas = safe_numeric("area_m2")
    if areas:
        areas_sorted = sorted(areas)
        metrics["area_median"] = median(areas_sorted)
        q10_idx = int(0.1 * (len(areas_sorted) - 1))
        q90_idx = int(0.9 * (len(areas_sorted) - 1))
        metrics["area_q10"] = areas_sorted[q10_idx]
        metrics["area_q90"] = areas_sorted[q90_idx]
    else:
        metrics["area_median"] = None
    return metrics


def compare_metrics(prev: Dict[str, Any], curr: Dict[str, Any]) -> List[str]:
    """Compare previous and current metrics; return a list of suspect flags."""
    flags: List[str] = []
    if not prev:
        return flags
    # Row count drop
    rows_prev, rows_curr = prev.get("rows", 0), curr.get("rows", 0)
    if rows_prev and rows_curr < 0.7 * rows_prev:
        flags.append("ROWS_DROP")
    # Null rate increases
    for key in ("price_null_rate", "area_null_rate", "rooms_null_rate"):
        prev_rate, curr_rate = prev.get(key, 0.0), curr.get(key, 0.0)
        if curr_rate - prev_rate > 0.3 and curr_rate > 0.5:
            flags.append(f"{key.upper()}_SPIKE")
    # Area median shift
    prev_med, curr_med = prev.get("area_median"), curr.get("area_median")
    if prev_med and curr_med:
        if curr_med > 2.0 * prev_med or curr_med < 0.5 * prev_med:
            flags.append("AREA_MEDIAN_SHIFT")
    return flags


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to CSV at ``path``.  Columns are union of keys."""
    if not rows:
        # Still write header with mandatory columns for consistency
        rows = [{"developer_key": "", "captured_at": "", "source": ""}]
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_errors_csv(path: str, errors: List[Dict[str, Any]]) -> None:
    """Write errors list to CSV; create file only if errors present."""
    if not errors:
        return
    keys: List[str] = []
    for e in errors:
        for k in e.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for e in errors:
            writer.writerow(e)


def find_last_snapshot_dir(snapshot_dir: str, pattern: str) -> Optional[str]:
    """Return the path to the most recent snapshot file matching pattern."""
    if not os.path.isdir(snapshot_dir):
        return None
    candidates = [f for f in os.listdir(snapshot_dir) if re.fullmatch(pattern, f)]
    if not candidates:
        return None
    # sort by timestamp extracted from filename (assuming YYYYMMDD_HHMMSS)
    candidates.sort(reverse=True)
    return os.path.join(snapshot_dir, candidates[0])


def read_metrics_from_csv(path: str) -> Dict[str, Any]:
    """Read minimal metrics from an existing snapshot CSV.

    Only reads count and null rates for price, area, rooms and area median.
    If the file cannot be read or has no rows, returns empty dict.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return compute_snapshot_metrics(rows)
    except Exception:
        return {}


def build_reason(reason_code: str, severity: str, summary: str,
                 what: List[str], why: List[str], checks: List[str],
                 next_actions: List[str], manual_rule: List[str], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble a structured reason dictionary."""
    return {
        "category": reason_code.split("_")[0].lower(),
        "severity": severity,
        "reason_code": reason_code,
        "summary": summary,
        "what_happened": what,
        "why_it_matters": why,
        "checks": checks,
        "next_actions": next_actions,
        "manual_accept_rule": manual_rule,
        "evidence": evidence,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Shantary MacroCatalog parser (Khabarovsk)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "shantary_config.json"),
                    help="Path to config JSON")
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
    # Resolve base_dir relative to this script
    script_dir = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.join(script_dir, cfg.out_dir)
    snapshots_dir = os.path.join(base_dir, "snapshots_raw")
    errors_dir = os.path.join(base_dir, "errors")
    result_path = os.path.join(base_dir, "result.json")
    reason_path = os.path.join(base_dir, "reason.json")
    # Ensure directories exist
    _ensure_dir(snapshots_dir)
    _ensure_dir(errors_dir)
    logger = setup_logging(ts, base_dir)
    logger.info("Config loaded: city=%s, activity=%s, categories=%s", cfg.target_city_contains, cfg.activity, cfg.categories)
    # Setup HTTP session with appropriate headers
    session = requests.Session()
    session.headers.update({
        "accept": "application/json, text/plain, */*",
        "content-type": "application/json",
        "origin": "https://shantary.ru",
        "referer": "https://shantary.ru/",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
    })
    # Prepare variables for retry passes
    rows_all: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    captured_at_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    suspect_flags: List[str] = []
    retries_used = 0
    status = "FAIL"
    reason_obj: Optional[Dict[str, Any]] = None
    # We will attempt up to cfg.max_retry_passes automatic passes.  Each pass
    # restarts from scratch, clearing rows/errors, and increases sleep.
    sleep_sec = cfg.sleep_between_requests_sec
    for pass_no in range(1, cfg.max_retry_passes + 1):
        logger.info("\n=== Retry pass %d/%d (sleep=%.2fs) ===", pass_no, cfg.max_retry_passes, sleep_sec)
        rows_all.clear()
        errors.clear()
        suspect_flags.clear()
        session.headers.update({})  # ensure headers remain
        try:
            # Acquire catalog URL
            logger.info("[PROGRESS 1/5] Step A: fetching signed catalog URL")
            # Ensure we have a working signed /estate/catalog/ URL.
            # If get_request_url is empty or expired, auto-refresh will try:
            #  - probe_api (domain-only endpoint)
            #  - probe_html (scan pages)
            #  - browser (Playwright capture)  
            catalog_url = ensure_catalog_url(session, cfg, logger)
            logger.info("Catalog URL: %s", catalog_url[:120] + ("..." if len(catalog_url) > 120 else ""))
            time.sleep(sleep_sec)
            # Fetch get_data
            logger.info("[PROGRESS 2/5] Step B: fetching get_data")
            get_data = fetch_get_data(session, catalog_url, cfg, logger)
            # Resolve category tokens present in get_data
            available_categories = set()
            for h in (get_data.get("houses") or []):
                for c in (h.get("categories") or []):
                    available_categories.add(str(c))
            # Determine which canonical categories to process
            categories_to_process: List[Tuple[str, str]] = []  # list of (canonical, token)
            missing_canonical: List[str] = []
            for canon, token in cfg.categories.items():
                token = token or ""
                if not token:
                    continue
                if token not in available_categories:
                    missing_canonical.append(canon)
                else:
                    categories_to_process.append((canon, token))
            if missing_canonical:
                logger.warning("Missing categories in get_data: %s", ", ".join(missing_canonical))
                if cfg.extra_categories_strict:
                    raise RuntimeError(f"Missing required categories: {', '.join(missing_canonical)}")
            # For each category, pick houses and fetch estates
            all_house_count = 0
            target_house_counts: Dict[str, int] = {}
            for canon_cat, token_cat in categories_to_process:
                logger.info("\n[PROGRESS 3/5] Category '%s' (token=%s)", canon_cat, token_cat)
                targets = pick_houses(get_data, cfg, token_cat, logger)
                logger.info("Targets for %s: %d houses", canon_cat, len(targets))
                target_house_counts[canon_cat] = len(targets)
                all_house_count += len(targets)
                for house_id, house_name, complex_name in _t(targets, desc=f"{canon_cat} houses"):
                    try:
                        time.sleep(sleep_sec + random.uniform(0.0, sleep_sec * 0.5))
                        j = fetch_get_estates(session, catalog_url, cfg, house_id, token_cat, logger)
                        rows = flatten_estates(j, captured_at_iso=captured_at_iso, developer_key="shantary", developer_name="Shantary", category=canon_cat, source="https://shantary.ru", city=cfg.target_city_contains, requested_category=token_cat)
                        # enrich with context
                        for r in rows:
                            r["house_id"] = house_id
                            r["house_name"] = house_name
                            r["complex_name"] = complex_name
                        rows_all.extend(rows)
                        logger.info("House %s (%s): rows=%d", house_id, canon_cat, len(rows))
                    except Exception as e:
                        logger.error("House %s failed: %s", house_id, e)
                        errors.append({
                            "captured_at": captured_at_iso,
                            "house_id": house_id,
                            "house_name": house_name,
                            "complex_name": complex_name,
                            "category": canon_cat,
                            "error": str(e),
                        })
            # After processing all categories and houses, evaluate results
            retries_used = pass_no
            if rows_all:
                # Sanity checks independent of baseline (helps on a fresh machine).
                try:
                    flat_target_houses = int(target_house_counts.get("flat", 0) or 0)
                    flat_rows = sum(1 for r in rows_all if str(r.get("lot_type") or "") == "flat")
                    if flat_target_houses > 0 and flat_rows < int(getattr(cfg, "min_flats_retry_threshold", 50) or 50):
                        suspect_flags.append("TOO_FEW_FLATS")
                        logger.warning(
                            "Suspiciously small flat result: houses=%d flats=%d (expected much higher)",
                            flat_target_houses, flat_rows
                        )
                except Exception:
                    # Never fail because of sanity checks
                    pass
                # Compute current metrics
                current_metrics = compute_snapshot_metrics(rows_all)
                # Read previous metrics
                pattern = r"shantary__.*__\d{8}_\d{6}\.csv"
                prev_file = find_last_snapshot_dir(snapshots_dir, pattern)
                prev_metrics: Dict[str, Any] = {}
                if prev_file:
                    prev_metrics = read_metrics_from_csv(prev_file)
                baseline_flags = compare_metrics(prev_metrics, current_metrics)
                for fl in baseline_flags:
                    if fl not in suspect_flags:
                        suspect_flags.append(fl)
                if errors and "FETCH_ERRORS" not in suspect_flags:
                    suspect_flags.append("FETCH_ERRORS")
                status = "OK" if not suspect_flags else "WARN"
            else:
                status = "FAIL"
            # Stop automatic retries only if everything looks OK.
            # If WARN or FAIL, we retry in a softer mode (up to max_retry_passes).
            if status != "OK" and ("TOO_FEW_FLATS" in suspect_flags) and pass_no < cfg.max_retry_passes:
                # For Shantary this usually means we captured a *limited* token before the user unlocked the catalog.
                logger.warning(
                    "TOO_FEW_FLATS: likely gated/limited token. Clearing get_request_url and forcing browser refresh on next pass."
                )
                try:
                    persist_get_request_url(cfg, "", logger)  # clear persisted token so next run doesn't reuse it
                except Exception:
                    pass
                cfg.get_request_url = ""
                cfg.auto_refresh_get_request_url = True
                cfg.auto_refresh_method = "browser"

            if status == "OK":
                break
            logger.warning("Status=%s after pass %d; will auto-retry if attempts remain", status, pass_no)

        except Exception as e:
            # Record a high-level error for this pass
            logger.error("Pass %d failed: %s", pass_no, e)
            errors.append({
                "captured_at": captured_at_iso,
                "house_id": None,
                "house_name": None,
                "complex_name": None,
                "category": None,
                "error": str(e),
            })
                # Auto retry delay (requirement): 9 seconds + jitter, up to max_retry_passes.
        if pass_no < cfg.max_retry_passes:
            delay = float(getattr(cfg, "auto_retry_delay_sec", 9.0) or 9.0)
            delay = delay + random.uniform(0.0, 2.0)
            logger.warning("Auto retry-pass #%d scheduled in %.1fs", pass_no + 1, delay)
            time.sleep(delay)

# Increase sleep for next pass
        sleep_sec *= cfg.sleep_backoff_factor
        # If after all automatic passes status remains WARN due to TOO_FEW_FLATS, treat as FAIL to trigger manual help.
    if status == "WARN" and "TOO_FEW_FLATS" in suspect_flags:
        logger.error("After retries, flats still too few; treating status as FAIL to trigger manual help")
        status = "FAIL"

    # If after all automatic passes status remains FAIL, ask the operator for help (requirement).
    if status == "FAIL" and cfg.max_retry_passes > 0:
        logger.error("All automatic retry passes exhausted; status remains FAIL")
        if getattr(cfg, "manual_help_after_fail", True):
            print(f"{WRN}Нужна помощь пользователя для восстановления доступа/токена.{RST}")
            print("Откроется браузер. Перейди на сайт, выбери Хабаровск (если нужно), открой каталог/шахматку (flat/garage).")
            print("Браузер закроется автоматически, как только будут пойманы нужные запросы, и парсер продолжит.")
            try:
                # Force a browser-assisted refresh (interactive) and then one final soft pass
                cfg.auto_refresh_get_request_url = True
                cfg.auto_refresh_method = "browser"
                sleep_sec = max(float(sleep_sec), float(getattr(cfg, "manual_soft_sleep_sec", 1.2)))
                logger.info("Entering MANUAL SOFT mode: sleep=%.2fs", sleep_sec)

                # Try to refresh the catalog URL interactively
                _ = ensure_catalog_url(session, cfg, logger)

                # One final pass after manual help
                rows_all.clear()
                errors.clear()
                logger.info("[MANUAL PASS] Retrying full data collection once after manual help")

                catalog_url = ensure_catalog_url(session, cfg, logger)
                time.sleep(sleep_sec)
                get_data = fetch_get_data(session, catalog_url, cfg, logger)

                available_categories = set()
                for h in (get_data.get("houses") or []):
                    for c in (h.get("categories") or []):
                        available_categories.add(str(c))

                categories_to_process = []
                missing_canonical = []
                for canon, token in cfg.categories.items():
                    token = token or ""
                    if not token:
                        continue
                    if token not in available_categories:
                        missing_canonical.append(canon)
                    else:
                        categories_to_process.append((canon, token))

                if missing_canonical and cfg.extra_categories_strict:
                    raise RuntimeError("Missing required categories after manual help: " + ", ".join(missing_canonical))

                for canon_cat, token_cat in categories_to_process:
                    targets = pick_houses(get_data, cfg, token_cat, logger)
                    for house_id, house_name, complex_name in _t(targets, desc=f"MANUAL {canon_cat} houses"):
                        try:
                            time.sleep(sleep_sec + random.uniform(0.0, sleep_sec * 0.5))
                            j = fetch_get_estates(session, catalog_url, cfg, house_id, token_cat, logger)
                            rows = flatten_estates(j, captured_at_iso=captured_at_iso, developer_key="shantary", developer_name="Shantary", category=canon_cat, source="https://shantary.ru", city=cfg.target_city_contains, requested_category=token_cat)
                            for r in rows:
                                r["house_id"] = house_id
                                r["house_name"] = house_name
                                r["complex_name"] = complex_name
                            rows_all.extend(rows)
                        except Exception as e:
                            errors.append({
                                "captured_at": captured_at_iso,
                                "house_id": house_id,
                                "house_name": house_name,
                                "complex_name": complex_name,
                                "category": canon_cat,
                                "error": str(e),
                            })

                retries_used = cfg.max_retry_passes
                if rows_all:
                    current_metrics = compute_snapshot_metrics(rows_all)
                    pattern = r"shantary__.*__\d{8}_\d{6}\.csv"
                    prev_file = find_last_snapshot_dir(snapshots_dir, pattern)
                    prev_metrics = read_metrics_from_csv(prev_file) if prev_file else {}
                    suspect_flags = compare_metrics(prev_metrics, current_metrics)
                    if errors:
                        suspect_flags.append("FETCH_ERRORS")
                    status = "OK" if not suspect_flags else "WARN"
                else:
                    status = "FAIL"
            except Exception as e:
                logger.error("Manual help pass failed: %s", e)
    # Build reason if WARN or FAIL
    if status in ("WARN", "FAIL"):
        if status == "FAIL":
            reason_code = "EMPTY_RESULT" if not rows_all else "UNKNOWN_ERROR"
            summary = "Пустой результат: не удалось собрать данные" if not rows_all else "Неизвестная ошибка при сборе данных"
            what = [f"Собрано строк: {len(rows_all)}", f"Ошибок: {len(errors)}"]
            why = ["Без данных аналитика и витрины не смогут обновиться"]
            checks = ["Проверьте правильность get_request_url и категорий в конфиге",
                      "Откройте https://shantary.ru и убедитесь, что каталог доступен"]
            next_actions = ["Получите новый get_request_url через браузер и обновите конфиг",
                            "Уменьшите параллельность и увеличьте задержки, затем попробуйте снова"]
            manual_rule = ["Не принимайте результат вручную, так как данных нет"]
        else:  # WARN
            # Determine first suspect flag as reason_code for summary
            reason_code = suspect_flags[0] if suspect_flags else "UNKNOWN_WARN"
            summary = "Подозрительные изменения в данных – см. suspect_flags"
            what = [f"Флаги: {', '.join(suspect_flags)}", f"Строк: {len(rows_all)}", f"Ошибок: {len(errors)}"]
            why = ["Быстрые изменения могут означать сбой парсинга, а не реальное изменение предложения"]
            checks = ["Сравните текущий snapshot с предыдущим", "Проверьте правильность разборов полей price/area/rooms"]
            next_actions = ["Если изменения подтверждаются, можно принять вручную",
                            "Иначе – подстройте правила разбора и запустите повторно"]
            manual_rule = ["Принять можно, если девелопер подтвердит корректность данных"]
        evidence = {
            "run_id": ts,
            "developer_key": "shantary",
            "attempt": retries_used,
            "suspect_flags": suspect_flags,
            "errors_count": len(errors),
        }
        reason_obj = build_reason(reason_code, "warn" if status == "WARN" else "fail",
                                  summary, what, why, checks, next_actions, manual_rule,
                                  evidence)
    # Write snapshot and errors
    # Snapshot filename as per specification: <developer_key>__<city>__<date_time>.csv
    city_slug = cfg.target_city_contains.replace(" ", "").replace("-", "").lower()
    snapshot_filename = f"shantary__{city_slug}__{ts}.csv"
    snapshot_path = os.path.join(snapshots_dir, snapshot_filename)
    write_csv(snapshot_path, rows_all)
    errors_path = os.path.join(errors_dir, f"{ts}_shantary_errors.csv")
    write_errors_csv(errors_path, errors)
    # Write result.json
    result = {
        "status": status,
        "retries_used": retries_used,
        "suspect_flags": suspect_flags,
        "output_snapshot_path": snapshot_path,
        "message_for_human": "" if status == "OK" else summary,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Result written to %s", result_path)
    # Write reason.json if applicable
    if reason_obj:
        with open(reason_path, "w", encoding="utf-8") as f:
            json.dump(reason_obj, f, ensure_ascii=False, indent=2)
        logger.info("Reason written to %s", reason_path)
    # Final print summary to stdout
    print()
    print(f"{OK}Done.{RST} status={status} rows={len(rows_all)} errors={len(errors)} retries={retries_used}")
    print(f"Snapshot: {snapshot_path}")
    if errors:
        print(f"Errors:   {errors_path}")
    if status in ("WARN", "FAIL"):
        print(f"See reason.json for details: {reason_path}")
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())