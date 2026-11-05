"""A pose: its own tag blocks plus the expression/costume it defaults to,
on top of the style and the character."""

from __future__ import annotations

from dataclasses import dataclass

from .components import Framing
from .legwear import DEFAULT_LEGWEAR, LegwearState


@dataclass(frozen=True)
class Pose:
    action: str
    mood: str
    gesture: str
    framing: Framing | None
    leg_display: str
    expression: str | None
    costume: str
    negative: str = ""
    negative_tail: str = ""
    canvas: tuple[int, int] | None = None
    legwear: bool = True
    legwear_kind: str = DEFAULT_LEGWEAR
    legwear_state: LegwearState = LegwearState.WORN
    # Each of these, when set, replaces the character's or style's own block.
    body: str | None = None
    style: str | None = None
    face: str | None = None
    identity: str | None = None
    nails: str | None = None
    background: str | None = None
    loras: tuple[tuple[str, float], ...] = ()
    angle: str = ""
    angle_action: str = ""
