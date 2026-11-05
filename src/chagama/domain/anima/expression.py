"""The expression overlay: mouth and eyes, layered onto a pose's own face."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EyeQuality(Enum):
    COLD = "cold"
    BLANK = "blank"


# The eye_quality component text is the character's text for
# `expression.eye_quality` + expression.eyes.
@dataclass(frozen=True)
class Expression:
    mouth: str
    eyes: str = ""
    eye_quality: EyeQuality = EyeQuality.COLD
    # None keeps the character's eye shape.
    eye_base: str | None = None
    # Bare tags taken out of the face part.
    face_drop: frozenset[str] = frozenset()


# No expression at all: the persona pose leaves the face to its reference's tags.
BLANK = Expression(mouth="", eye_quality=EyeQuality.BLANK)
