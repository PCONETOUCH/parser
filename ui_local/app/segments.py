from __future__ import annotations

import math

SEGMENTS = [
    "СТУДИИ",
    "1к 35-39",
    "1к 40-49",
    "2к 50-59",
    "2кк 60-69",
    "3кк 70-79",
    "3кк80-90",
    "90-99",
    "от 100",
    "UNKNOWN",
]


def _to_num(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(str(v).replace(",", ".").strip())
    except Exception:
        return None


def classify_segment(rooms, area) -> str:
    r = _to_num(rooms)
    a = _to_num(area)
    if r == 0:
        return "СТУДИИ"
    if a is None:
        return "UNKNOWN"
    if 35 <= a <= 39:
        return "1к 35-39"
    if 40 <= a <= 49:
        return "1к 40-49"
    if 50 <= a <= 59:
        return "2к 50-59"
    if 60 <= a <= 69:
        return "2кк 60-69"
    if 70 <= a <= 79:
        return "3кк 70-79"
    if 80 <= a <= 90:
        return "3кк80-90"
    if 90 <= a <= 99:
        return "90-99"
    if a >= 100:
        return "от 100"
    return "UNKNOWN"
