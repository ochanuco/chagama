"""Expression overlays: mouth and eyes, layered onto a pose's own face."""

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


EXPRESSIONS = {
    "resting": Expression(mouth=""),
    "sleepy": Expression(
        mouth="",
        eyes="(sleepy:1.4), (drowsy:1.3), (half-closed eyes:1.4), ",
        eye_quality=EyeQuality.BLANK),
    "doya": Expression(mouth="(smug:1.35), (doyagao:1.25), "),
    "v": Expression(mouth="(:v:1.5), "),
    "smile": Expression(
        mouth="(closed mouth:1.2), (light smile:1.25), ",
        eyes="(confident:1.18), ", eye_quality=EyeQuality.BLANK),
    "gao": Expression(
        mouth="(open mouth:1.35), (fang:1.3), ",
        eyes="(confident:1.18), ", eye_quality=EyeQuality.BLANK),
    "disdain": Expression(
        mouth="(closed mouth:1.2), ",
        eyes="(half-closed eyes:1.35), (cold eyes:1.3), (unamused:1.2), ",
        eye_quality=EyeQuality.BLANK),
    "dubious": Expression(
        mouth="(closed mouth:1.2), (skeptical:1.25), (light blush:1.1), ",
        eyes=("(half-closed eyes:1.35), (unamused:1.35), "
              "(raised eyebrow:1.25), (confused:1.15), "),
        eye_quality=EyeQuality.BLANK),
    "jitome": Expression(
        mouth="(closed mouth:1.2), ",
        eyes="(half-closed eyes:1.35), (unamused:1.35), (expressionless:1.2), ",
        eye_quality=EyeQuality.BLANK),
}
