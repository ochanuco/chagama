"""Caller-given redraw regions, in the source picture's own pixel space."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float


def rects_from_fractions(regions: Sequence[Sequence[float]], width: float,
                         height: float) -> list[Rect]:
    """Caller-given `[x0, y0, x1, y1]` fractions (0..1) as pixel Rects."""
    return [Rect(x0 * width, y0 * height, x1 * width, y1 * height)
            for x0, y0, x1, y1 in regions]
