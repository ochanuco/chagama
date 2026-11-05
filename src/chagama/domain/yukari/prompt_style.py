"""Yukari's own prompt blocks: who she is on top of the Anima style.

Every block ends with its own ", " (or, for a block that follows one that
does not, starts with it), as in `anima/style.py`.
"""

from __future__ import annotations

CHARACTER_TAG = "yuzuki yukari, "
SERIES_TAG = "vocaloid, voiceroid, "
IDENTITY = ("light purple hair, short hair with long locks, very long "
            "sidelocks, (dark greyish purple eyes:1.4), (dark irises:1.2), "
            "hair ornament, (pale pink pupils:1.05), (vertical oval pupils:1.0), "
            "(concentric iris rings:0.9), (mechanical eyes:0.6), ")
IDENTITY_TAG_NAMES = frozenset({
    "light purple hair", "short hair with long locks", "very long sidelocks",
    "dark greyish purple eyes", "hair ornament", "tareme", "jitome",
    "eggplant purple hooded cardigan", "rabbit hood",
})
# The sidelock tube goes only when another tie fills its place.
NO_HAIR_ORNAMENT = ("(no hair ornament:1.5), (tied sidelocks:1.3), "
                    "(sidelocks tied with orange hair tie:1.3), "
                    "(orange hair tie:1.2), ")
HAIR_ORNAMENT_BAN = "(hair ornament:1.4), (hairclip:1.3), (hair bobbles:1.2), "
BODY = ("(mature female:1.3), (adult:1.2), (wide hips:1.2), (thick thighs:1.2), "
        "(soft thighs:1.3), (long legs:1.35), (narrow waist:1.25), "
        "adult proportions, long torso, seven heads tall, (collarbone:1.2), ")
NAILS = "(nail polish:1.3), (light purple nails:1.4), "
# Leads every expression's eyes block: placed behind the backdrop (as FACE
# is) the tag stops reaching the eyes at any weight.
EYE_SHAPE = "(jitome:2.0), (tareme:1.2), (long eyes:1.25), "
EYE_SHAPE_SOFT = "(jitome:1.2), (tareme:1.2), (long eyes:1.25), "
FACE = ("(large eyes:1.2), (round face:1.3), (tareme:1.2), "
        "(thick eyelashes:1.3), (long eyelashes:1.15), ")

LINE_BAN = "(purple lineart:1.2), "
EYE_BAN = "(light purple eyes:1.3), (bright eyes:1.2), (pink eyes:1.2), "
VIVID_BAN = ("(magenta:1.45), (pink legwear:1.45), (bright purple:1.35), "
             "(vivid colors:1.3), (neon:1.3), (red:1.3), (maroon:1.35), "
             "(wine red:1.3), ")
HOOD_BAN = "(hood:1.3), (cardigan:1.3), "
GARMENT_BLACK_BAN = "(black jacket:1.35), (black clothes:1.3), (black hoodie:1.35), "
SHEER_TONE_BAN = ("(light purple legwear:1.35), (lavender legwear:1.25), "
                  "(gradient legwear:1.3), ")
# Keeps `mechanical eyes` in the eyes instead of turning the body into a robot.
MECHANICAL_BAN = (", (robot:1.3), (cyborg:1.3), (mechanical parts:1.2), "
                  "(android joints:1.2)")
