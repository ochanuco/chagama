"""Akari's delivery bands, backdrop palette and stage frame: yukari's, with
her orange rim outermost, her backdrops on a peach ground deepened toward it
and her frame drawn in browns."""

from __future__ import annotations

from ..yukari.delivery_style import BACKDROP_PALETTE as SHARED_PALETTE
from ..yukari.delivery_style import FRAME_LINE as SHARED_FRAME_LINE
from ..yukari.delivery_style import FRAME_RIM as SHARED_FRAME_RIM
from ..yukari.delivery_style import OUTLINES as SHARED_OUTLINES

OUTLINES = (SHARED_OUTLINES[0], {**SHARED_OUTLINES[-1], "color": "#c27138"})

BACKDROP_PALETTE = SHARED_PALETTE._replace(base="#f7d3b5", deep_stroke="#c27138")

FRAME_LINE = {**SHARED_FRAME_LINE, "color": "#56321a"}
FRAME_RIM = {**SHARED_FRAME_RIM, "color": "#33200f"}
