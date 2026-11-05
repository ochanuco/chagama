"""Akari's delivery bands: yukari's, with her orange rim outermost."""

from __future__ import annotations

from ..yukari.delivery_style import OUTLINES as SHARED_OUTLINES

OUTLINES = (SHARED_OUTLINES[0], {**SHARED_OUTLINES[-1], "color": "#c27138"})
