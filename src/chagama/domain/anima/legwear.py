"""Legwear kinds and states every character's wardrobe is drawn with; the
blocks themselves are the character's (`Wardrobe`)."""

from __future__ import annotations

from enum import Enum

LEGWEARS = ("opaque", "sheer-gloss", "sheer")
DEFAULT_LEGWEAR = "sheer"


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

# A worn `sheer` / `sheer-gloss` kind pairs with SHEER_BAN and drops
# `style.GARMENT_GLOSS_TAGS`; `sheer-gloss` also appends SHEER_GLOSS_BAN
# after everything else.
SHEER_BAN = ("(opaque legwear:1.3), (latex:1.3), (photorealistic:1.4), "
             "(realistic:1.3), (photo:1.2), (tan skin:1.35), (dark skin:1.3), "
             "(brown legwear:1.4), (brown pantyhose:1.4), (tan:1.2), "
             "(beige legwear:1.3), ")
SHEER_GLOSS_BAN = ("(glossy:1.3), (specular highlights:1.3), "
                   "(reflection:1.2), (ribbed legwear:1.3), "
                   "(vertical-striped legwear:1.3), ")
