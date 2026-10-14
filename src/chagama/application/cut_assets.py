"""The cut of a picture Generation (alpha, depth) stored as its assets."""

from __future__ import annotations

import json
from collections.abc import Callable

from ..domain.yukari import delivery_style
from ..infrastructure.comfyui.refinement_graph import DEPTH_CKPT, DEPTH_RESOLUTION
from .ingest import asset_key, attach_asset

CUT_ROLE = "cut"
ALPHA_ROLE = "alpha"
DEPTH_ROLE = "depth"


def current_cut() -> dict:
    """How this checkout makes the alpha and depth cut assets."""
    return {
        "alpha": {
            "matte_model": delivery_style.MATTE_MODEL,
            "matting_model": delivery_style.MATTING_MODEL,
            "matting_revision": delivery_style.MATTING_REVISION,
            "trimap_px": delivery_style.MATTING_TRIMAP_PX,
            "tile_px": delivery_style.MATTING_TILE_PX,
            "tile_overlap_px": delivery_style.MATTING_TILE_OVERLAP_PX,
            "hole_key_share": delivery_style.MATTE_HOLE_MIN_KEY_SHARE,
            "enclosed_key_min_shade": delivery_style.ENCLOSED_KEY_MIN_SHADE,
            "translucent_min_key_share": delivery_style.TRANSLUCENT_MIN_KEY_SHARE,
            "translucent_min_area": delivery_style.TRANSLUCENT_MIN_AREA,
        },
        "depth": {"ckpt": DEPTH_CKPT, "resolution": DEPTH_RESOLUTION},
    }


def stored_cut(services, generation_id: str) -> tuple[dict, set[str]]:
    roles = {asset["role"] for asset in
             services.management.list_assets(generation_id)}
    if CUT_ROLE not in roles:
        return {}, roles
    raw = services.management.fetch_asset(generation_id, CUT_ROLE)
    try:
        stored = json.loads(raw) if raw else {}
    except ValueError:
        stored = {}
    return (stored if isinstance(stored, dict) else {}), roles


def reusable(services, generation_id: str, role: str, stored: dict,
             roles: set[str], current: dict) -> bytes | None:
    if role not in roles or stored.get(role) != current[role]:
        return None
    return services.management.fetch_asset(generation_id, role)


def attach_cut(management, emit: Callable[[str], None], generation_id: str, *,
               key_prefix: str | None, fetched: dict, stored: dict,
               roles: set[str], current: dict, cut_roles: list[str]) -> None:
    """Attach the freshly cut `cut_roles` and the merged `cut` description."""
    for role in cut_roles:
        name, data = fetched[role]
        attach_asset(management, generation_id, role=role, name=name, data=data,
                     idempotency_key=asset_key(key_prefix, 0, role))
        emit(f"{name} -> {role} on {generation_id}")
    if cut_roles:
        merged = {role: entry for role, entry in
                  {**stored, **{role: current[role] for role in cut_roles}}.items()
                  if role in roles or role in cut_roles}
        attach_asset(management, generation_id, role=CUT_ROLE, name="cut.json",
                     data=json.dumps(merged, indent=2).encode(),
                     content_type="application/json",
                     idempotency_key=asset_key(key_prefix, 0, CUT_ROLE))
        emit(f"cut.json -> {CUT_ROLE} on {generation_id}")
