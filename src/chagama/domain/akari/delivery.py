"""Akari's delivery bands and backdrop palette: yukari's, with her orange
rim outermost and her backdrops on a peach ground deepened toward it."""

from __future__ import annotations

from ..yukari.delivery_style import BACKDROP_PALETTE as SHARED_PALETTE
from ..yukari.delivery_style import OUTLINES as SHARED_OUTLINES

OUTLINES = (SHARED_OUTLINES[0], {**SHARED_OUTLINES[-1], "color": "#c27138"})

BACKDROP_PALETTE = SHARED_PALETTE._replace(base="#f7d3b5", deep_stroke="#c27138")
