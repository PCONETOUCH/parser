from __future__ import annotations

DEFAULT_SYNONYMS = {
    "lot_id": ["lot_item_id", "flat_item_id", "uid", "id", "lot_id"],
    "status": ["status", "lot_status", "flat_status"],
    "price": ["price", "estate_price"],
    "price_m2": ["price_m2", "price_per_m2", "estate_price_m2"],
    "area": ["area_m2", "estate_area", "area"],
    "rooms": ["rooms", "estate_rooms", "room_count"],
    "project": ["complex_name", "project_name", "object_name", "name", "ЖК", "complex"],
}


def resolve_schema(columns: list[str], overrides: dict[str, list[str]] | None = None) -> dict[str, str | None]:
    cols = {c.lower().strip(): c for c in columns}
    synonyms = {**DEFAULT_SYNONYMS, **(overrides or {})}
    resolved: dict[str, str | None] = {}
    for key, variants in synonyms.items():
        found = None
        for v in variants:
            vv = v.lower().strip()
            if vv in cols:
                found = cols[vv]
                break
        resolved[key] = found
    return resolved
