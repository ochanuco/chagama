"""Named words for yukari's request options, published in the catalog."""

from __future__ import annotations

from .delivery_style import REDRAW_DENOISE

DIALS = {
    "redraw": {
        "denoise": {"keep": REDRAW_DENOISE},
    },
    "deliver": {
        "keep_legwear": {},
    },
    "patches": {
        "render.width": {"draft": 1024, "full": 1280},
        "render.height": {"draft": 1640, "full": 2048},
    },
}
