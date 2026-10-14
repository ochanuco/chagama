"""The wardrobe: garment tags layered between a pose's gesture and scene."""

from __future__ import annotations

from enum import Enum

COSTUMES = {
    "roomwear": ("(oversized shirt:1.35), (sleeves past wrists:1.3), "
                 "(bare shoulders:1.1), "),
    "outing": ("(oversized sweatshirt:1.35), (white sweatshirt:1.2), "
               "(sleeves past wrists:1.25), (denim shorts:1.3), "),
    "standard": ("(eggplant purple hooded cardigan:1.5), (dark violet hoodie:1.25), "
                 "open cardigan, "
                 "(rabbit hood:1.3), long sleeves, drawstring, "
                 "(purple dress:1.25), frills, (sleeves past wrists:1.15), "
                 "hood down, "),
    "suspender": ("(muted orange t-shirt:1.2), (plain t-shirt:1.15), "
                  "(suspender skirt:1.4), (navy blue skirt:1.3), "
                  "(knee-length skirt:1.8), (pleated skirt:1.1), "),
}

# Negative tags a costume adds on top of the recipe's shared bans.
COSTUME_BAN = {
    "suspender": ("(text:1.3), (print:1.2), (logo:1.2), (number:1.2), "
                  "(bright orange:1.2), "),
}

# The `opaque` legwear: each costume's own tights.
LEGWEAR = {
    "roomwear": "",
    "outing": "(black pantyhose:1.5), (opaque pantyhose:1.4), ",
    "suspender": "(black pantyhose:1.5), (opaque pantyhose:1.4), ",
    "standard": ("(dark purple pantyhose:1.45), (opaque pantyhose:1.3), "
                 "(gradient legwear:1.2), (purple gradient:1.1), "),
}

# The `sheer-gloss` legwear: one block for every costume. `recipe.negative`
# pairs it with SHEER_BAN, drops GARMENT_GLOSS_TAGS, and appends
# SHEER_GLOSS_BAN at the end.
SHEER_GLOSS_LEGWEAR = (
    "(sheer black pantyhose:1.5), (dark violet tint:1.2), "
    "(see-through legwear:1.4), (skin visible through legwear:1.3), "
    "(pantyhose:1.4), (subtle sheen on legwear:1.05), "
    "(anime coloring:1.2), ")

# The `sheer` legwear: one flat black-purple tone for every costume, no
# gradient direction words. `recipe.negative` treats it like `sheer-gloss`
# and adds SHEER_TONE_BAN.
SHEER_LEGWEAR = (
    "(sheer dark purple pantyhose:1.5), (see-through black tights:1.2), "
    "(skin clearly visible through pantyhose:1.4), "
    "(thin translucent legwear:1.3), (pantyhose:1.4), "
    "(subtle sheen on legwear:1.15), ")

LEGWEARS = ("opaque", "sheer-gloss", "sheer")
DEFAULT_LEGWEAR = "sheer"


def legwear_block(costume: str, legwear: str) -> str:
    if legwear == "sheer-gloss":
        return SHEER_GLOSS_LEGWEAR
    if legwear == "sheer":
        return SHEER_LEGWEAR
    if legwear != "opaque":
        raise KeyError(legwear)
    return LEGWEAR[costume]

# Costumes whose garments include a hood or cardigan; `recipe.negative`
# skips HOOD_BAN for them.
HOODED_COSTUMES = frozenset({"standard"})


class LegwearState(Enum):
    WORN = "worn"
    REMOVING = "removing"
    OFF = "off"


LEGWEAR_STATES = ("worn", "removing", "off")
DEFAULT_LEGWEAR_STATE = "worn"

# OFF replaces the worn block outright: no costume's legwear tags belong
# on bare legs.
REMOVING_LEGWEAR = ("(pantyhose pull:1.3), (pulled by self:1.25), "
                    "(pantyhose around knees:1.35), "
                    "(pantyhose pulled down:1.3), (bare thighs:1.2), ")
OFF_LEGWEAR = "(bare legs:1.3), (no legwear:1.3), "

REMOVING_LEGWEAR_BAN = "(thighhighs:1.4), (kneehighs:1.4), (socks:1.3), "
OFF_LEGWEAR_BAN = ("(pantyhose:1.3), (thighhighs:1.3), (kneehighs:1.2), "
                   "(socks:1.2), ")


def legwear_text(costume: str, legwear: str, state: LegwearState) -> str:
    if state is LegwearState.OFF:
        return OFF_LEGWEAR
    text = legwear_block(costume, legwear)
    if state is LegwearState.REMOVING:
        return text + REMOVING_LEGWEAR
    return text
