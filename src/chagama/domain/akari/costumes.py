"""Akari's wardrobe: the half-worn jacket over a pinafore, and white tights
with a faint orange tint."""

from __future__ import annotations

from ..anima.character import Wardrobe
from .prompt_style import GARMENT_BLACK_BAN, HOOD_BAN, SHEER_TONE_BAN

_JACKET = ("(very dark brown jacket:1.4), (blackish brown:1.3), "
           "(dark orange-brown tint:1.1), ")

COSTUMES = {
    "half_jacket": (_JACKET + "(open jacket:1.3), (jacket partially removed:1.4), "
                    "(jacket slipping off shoulders:1.3), "
                    "(off-shoulder jacket:1.3), (oversized jacket:1.2), "
                    "(white fur-trimmed collar:1.2), long sleeves, "
                    "(dark grey pinafore dress:1.4), (white shirt:1.2), "
                    "collared shirt, (orange neck ribbon:1.1), "),
}

# From behind the open front, shirt and ribbon are out of sight.
BEHIND_COSTUMES = {
    "half_jacket": (_JACKET + "(jacket slipping off shoulders:1.3), "
                    "(oversized jacket:1.2), (white fur-trimmed collar:1.2), "
                    "long sleeves, (dark grey pinafore dress:1.4), "),
}

LEGWEAR = {
    "half_jacket": ("(white pantyhose:1.5), (opaque pantyhose:1.3), "
                    "(light orange tint:1.1), "),
}

SHEER_GLOSS_LEGWEAR = (
    "(sheer white pantyhose:1.5), (light orange tint:1.1), "
    "(see-through legwear:1.4), (skin visible through legwear:1.3), "
    "(pantyhose:1.4), (subtle sheen on legwear:1.05), "
    "(anime coloring:1.2), ")

SHEER_LEGWEAR = (
    "(white pantyhose:1.5), (white tights:1.2), (light orange tint:1.1), "
    "(thin translucent legwear:1.3), (pantyhose:1.4), "
    "(subtle sheen on legwear:1.15), ")

WARDROBE = Wardrobe(
    costumes=COSTUMES,
    behind=BEHIND_COSTUMES,
    behind_ban={},
    garment_ban={"half_jacket": GARMENT_BLACK_BAN},
    costume_ban={},
    hooded=frozenset(),
    hood_ban=HOOD_BAN,
    opaque=LEGWEAR,
    sheer=SHEER_LEGWEAR,
    sheer_gloss=SHEER_GLOSS_LEGWEAR,
    sheer_tone_ban=SHEER_TONE_BAN,
)
