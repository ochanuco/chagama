"""The expression overlay: mouth and eyes, layered onto a pose's own face."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EyeQuality(Enum):
    COLD = "cold"
    BLANK = "blank"


# The eye_quality component text is EYE_QUALITY[expression.eye_quality] +
# expression.eyes -- the shared quality tags, then the expression's own extra.
EYE_QUALITY = {
    EyeQuality.COLD: "(half-closed eyes:1.3), (unamused:1.15), ",
    EyeQuality.BLANK: "",
}


@dataclass(frozen=True)
class Expression:
    mouth: str
    eyes: str = ""
    eye_quality: EyeQuality = EyeQuality.COLD
    # None keeps the character's eye shape; jitome 2.0 holds the eyes open
    # and flattens a strong emotion.
    eye_base: str | None = None
    # Bare tags taken out of the face part.
    face_drop: frozenset[str] = frozenset()


# No expression at all: the persona pose leaves the face to its reference's tags.
BLANK = Expression(mouth="", eye_quality=EyeQuality.BLANK)
