"""The expressions every character shares by name; a character's
`ExpressionLook` lays its own eyes over them."""

from __future__ import annotations

from .expression import Expression, EyeQuality

EXPRESSIONS = {
    "resting": Expression(mouth=""),
    "sleepy": Expression(
        mouth="",
        eyes="(sleepy:1.4), (drowsy:1.3), (half-closed eyes:1.4), ",
        eye_quality=EyeQuality.BLANK),
    "doya": Expression(
        mouth="(v-shaped eyebrows:1.25), (smirk:1.3), (smug:1.2), closed mouth, "),
    "v": Expression(mouth="(:v:1.5), "),
    "smile": Expression(
        mouth="(closed mouth:1.2), (light smile:1.25), ",
        eyes="(confident:1.18), ", eye_quality=EyeQuality.BLANK),
    "gao": Expression(
        mouth="(open mouth:1.35), (fang:1.3), ",
        eyes="(confident:1.18), ", eye_quality=EyeQuality.BLANK),
    "disdain": Expression(
        mouth="(closed mouth:1.2), ",
        eyes="(cold eyes:1.3), ",
        eye_quality=EyeQuality.BLANK),
    "dubious": Expression(
        mouth="(closed mouth:1.2), (skeptical:1.25), (light blush:1.1), ",
        eyes="(raised eyebrow:1.25), (confused:1.15), ",
        eye_quality=EyeQuality.BLANK),
    "jitome": Expression(
        mouth="(closed mouth:1.2), ",
        eyes="(expressionless:1.2), ",
        eye_quality=EyeQuality.BLANK),
    "neutral": Expression(
        mouth="(closed mouth:1.2), ",
        eyes="(expressionless:1.2), ",
        eye_quality=EyeQuality.BLANK),
    "joy": Expression(
        mouth="(open mouth:1.25), (smile:1.35), (happy:1.35), (blush:1.1), ",
        eye_quality=EyeQuality.BLANK),
    "amused": Expression(
        mouth="(closed mouth:1.1), (smirk:1.3), (amused:1.3), ",
        eyes="(raised eyebrow:1.15), ", eye_quality=EyeQuality.BLANK),
    "anger": Expression(
        mouth="(clenched teeth:1.3), (angry:1.4), (anger vein:1.25), ",
        eyes="(v-shaped eyebrows:1.4), (furrowed brow:1.25), ",
        eye_quality=EyeQuality.BLANK),
    "sorrow": Expression(
        mouth="(closed mouth:1.1), (frown:1.25), (sad:1.4), (tearing up:1.2), ",
        eyes="(upturned eyebrows:1.35), ", eye_quality=EyeQuality.BLANK),
    "annoyed": Expression(
        mouth="(closed mouth:1.2), (furrowed brow:1.25), (frown:1.25), ",
        eyes="(annoyed:1.3), ", eye_quality=EyeQuality.BLANK),
    "closed": Expression(
        mouth="", eyes="(closed eyes:1.6), ", eye_quality=EyeQuality.BLANK,
        eye_base=""),
    "surprised": Expression(
        mouth="(open mouth:1.35), (:o:1.3), (surprised:1.4), ",
        eyes="(raised eyebrows:1.3), ", eye_quality=EyeQuality.BLANK),
}
