"""Akari's own prompt blocks: who she is on top of the Anima style.

Every block ends with its own ", " (or, for a block that follows one that
does not, starts with it), as in `anima/style.py`.
"""

from __future__ import annotations

CHARACTER_TAG = "kizuna akari, "
SERIES_TAG = "vocaloid, voiceroid, "
IDENTITY = ("(white hair:1.3), (very long hair:1.2), (twin braids:1.4), "
            "low twin braids, sidelocks, (deep blue eyes:1.3), "
            "(pale pink pupils:1.3), (vertical oval pupils:1.2), "
            "(light cyan lower iris:1.15), (concentric iris rings:1.1), "
            "(mechanical eyes:0.8), ")
IDENTITY_TAG_NAMES = frozenset({
    "kizuna akari", "white hair", "very long hair", "twin braids",
    "low twin braids", "deep blue eyes", "pale pink pupils", "tareme",
    "droopy eyes",
})
BODY = ("(mature female:1.3), (adult:1.2), (wide hips:1.2), (thick thighs:1.2), "
        "(soft thighs:1.3), (long legs:1.35), (narrow waist:1.25), "
        "adult proportions, long torso, seven heads tall, (collarbone:1.2), ")
EYE_SHAPE = "(tareme:2.3), (long eyes:1.6), (droopy eyes:1.8), "
# Her eyes under every expression: the droop lives in the lower lid.
EYES = ("(half-closed eyes:1.25), (gentle eyes:1.1), (smiling eyes:1.3), "
        "(puffy lower eyelids:1.4), (lower eyelids curving upward:1.7), "
        "(downturned eyes:1.8), (lower eyelid sloping down at outer corner:1.7), "
        "(outer corners of eyes pointing down:1.6), "
        "(thick lower eyelid line:1.5), (defined lower eyelashes:1.2), ")
FACE = "(mature face:1.2), (oval face:1.1), "
# Carries the droopy-eye identity; appended to the style block.
STYLE = r", (1990s \(style\):0.7)"

EYE_BAN = ("(wide-open eyes:1.3), (tsurime:1.4), (sanpaku:1.3), "
           "(white below iris:1.3), (small irises:1.2), "
           "(multicolored eyes:1.2), (heterochromia:1.3), "
           "(upturned eyes:1.5), (cat eyes:1.3), (sharp eyes:1.3), ")
# Keeps her tights off yukari's purple and off black.
PALETTE_BAN = ("(purple legwear:1.35), (black pantyhose:1.35), "
               "(dark legwear:1.25), (pure black:1.3), ")
HOOD_BAN = "(hood:1.3), "
GARMENT_BLACK_BAN = "(black jacket:1.3), "
SHEER_TONE_BAN = "(orange legwear:1.3), (gradient legwear:1.3), "
# Keeps `mechanical eyes` in the eyes, and yukari's accessories off her.
TAIL_BAN = (", (arm warmers:1.3), (detached sleeves:1.3), (headphones:1.3), "
            "(headset:1.3), (headwear:1.2), (robot:1.3), (cyborg:1.3), "
            "(mechanical parts:1.2), (android joints:1.2)")
