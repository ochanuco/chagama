"""Rate existing chimera generations and send chimera the numbers."""

from __future__ import annotations

from ..infrastructure.imaging.safety import RATING_KEYS, rate_image

PAGE_SIZE = 50


def rate_generation_by_id(chimera, identifier: str) -> dict:
    generation = chimera.request("GET", f"/api/v1/generations/{identifier}")
    generation_id = (generation or {}).get("id", identifier)
    if (generation or {}).get("original_purged_at"):
        data = chimera.fetch_generation_preview(identifier)
    else:
        data = chimera.fetch_generation_image(identifier)
    payload = rate_image(data)
    chimera.put_safety(generation_id, payload)
    return payload["rating"]


def format_rating(rating: dict) -> str:
    return " ".join(f"{key}={rating[key]:.3f}" for key in RATING_KEYS)


def list_short_ids(chimera, *, published: bool, limit: int | None):
    offset = 0
    while limit is None or offset < limit:
        size = PAGE_SIZE if limit is None else min(PAGE_SIZE, limit - offset)
        query = f"limit={size}&offset={offset}"
        if published:
            query = "published=true&" + query
        page = chimera.request("GET", f"/api/v1/generations?{query}")
        items = page.get("items", []) if isinstance(page, dict) else page
        for item in items:
            yield item["short_id"]
        if len(items) < size:
            return
        offset += size


def backfill(chimera, emit, *, published: bool, limit: int | None) -> tuple[int, int]:
    rated = failed = 0
    for short_id in list_short_ids(chimera, published=published, limit=limit):
        try:
            rating = rate_generation_by_id(chimera, short_id)
        except (Exception, SystemExit) as error:
            failed += 1
            emit(f"{short_id}: {error}")
            continue
        rated += 1
        emit(f"{short_id}: {format_rating(rating)}")
    emit(f"rated {rated}, failed {failed}")
    return rated, failed
