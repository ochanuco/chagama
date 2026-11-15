"""The wardrobe: garment tags layered between a pose's gesture and scene."""

from __future__ import annotations

from ..anima.character import Wardrobe
from .prompt_style import GARMENT_BLACK_BAN, HOOD_BAN, SHEER_TONE_BAN

COSTUMES = {
    "roomwear": ("(oversized shirt:1.35), (sleeves past wrists:1.3), "
                 "(bare shoulders:1.1), "),
    "outing": ("(oversized sweatshirt:1.35), (white sweatshirt:1.2), "
               "(sleeves past wrists:1.25), (denim shorts:1.3), "),
    "standard": ("(eggplant purple hooded cardigan:1.5), (dark violet hoodie:1.25), "
                 "open cardigan, "
                 "(rabbit hood:1.3), long sleeves, drawstring, "
                 "(purple dress:1.25), frills, (sleeves past wrists:1.15), "
                 "hood down, (criss-cross halter:1.2), halterneck, "),
    "suspender": ("(muted orange t-shirt:1.2), (plain t-shirt:1.15), "
                  "(suspender skirt:1.4), (navy blue skirt:1.3), "
                  "(knee-length skirt:1.8), (pleated skirt:1.1), "),
    "rei": ("(black dress:1.3), (high-waist dress:1.2), "
            "(short sleeves:1.3), (puffy short sleeves:1.2), "
            "(see-through puffy sleeves:1.2), "
            "(high collar:1.1), (gold choker ornament:1.1), "
            "(dark purple sash:1.25), (black ribbon tails:1.3), "
            "(black hanging ribbon from hip ornament:1.2), (gold tassels:1.15), "
            "(flared skirt:1.35), (a-line skirt:1.2), (knee-length skirt:1.5), "
            "(gradient skirt:1.3), (black to purple gradient:1.25), "
            "(sparkle print:1.1), (gold trim on hem:1.2), "
            "(grey see-through petticoat:1.15), "
            "(large gold spiral ornament on hip:1.2), "
            "(gold chain draped across skirt:1.15), "
            "(white shawl:1.3), (shawl draped over arms:1.2), "
            "(gold stripes on shawl:1.05), "
            "(black headphones:1.45), (headphones on head:1.2), "
            "(gold trim:1.05), (black wrist gloves:1.3), (bare forearms:1.25), "),
}

# A costume's tags when the request's angle shows her back. The official
# halter crosses on the chest only; from behind it is a bow at the nape.
BEHIND_COSTUMES = {
    "standard": COSTUMES["standard"].replace(
        "(criss-cross halter:1.2), halterneck, ",
        "(halterneck:1.3), (halter tied at nape:1.2), "
        "(backless dress:1.3), (bare back:1.2), "),
}
BEHIND_COSTUME_BAN = {
    "standard": ("(criss-cross straps:1.4), (crossed straps:1.4), "
                 "(back straps:1.3), (x-shaped straps:1.3), "),
}

# Negative tags a costume adds on top of the recipe's shared bans.
COSTUME_BAN = {
    "suspender": ("(text:1.3), (print:1.2), (logo:1.2), (number:1.2), "
                  "(bright orange:1.2), "),
    "rei": ("(horizontal-striped legwear:1.3), (side slit:1.3), "
            "(front slit:1.3), (open skirt:1.3), (split skirt:1.3), "
            "(white petticoat:1.3), (long sleeves:1.3), (long skirt:1.3), "
            "(white frills:1.25), "
            "(lace trim:1.2), (pencil skirt:1.25), (coat:1.2), (coin:1.2), "
            "(large stars:1.2), (star print:1.15), (earpiece:1.2), "
            "(sneakers:1.3), "),
}

# Footwear a costume puts in place of a pose's sneakers.
SHOES = {
    "rei": "(black high heels:1.3), (gold-trimmed heels:1.1), ",
}

# The `opaque` legwear: each costume's own tights.
LEGWEAR = {
    "roomwear": "",
    "outing": "(black pantyhose:1.5), (opaque pantyhose:1.4), ",
    "suspender": "(black pantyhose:1.5), (opaque pantyhose:1.4), ",
    "rei": ("(purple pantyhose:1.45), (vertical-striped pantyhose:1.4), "
            "(opaque pantyhose:1.3), "),
    "standard": ("(dark purple pantyhose:1.45), (opaque pantyhose:1.3), "
                 "(gradient legwear:1.2), (purple gradient:1.1), "),
}

# The `sheer-gloss` legwear: one block for every costume.
SHEER_GLOSS_LEGWEAR = (
    "(sheer black pantyhose:1.5), (dark violet tint:1.2), "
    "(see-through legwear:1.4), (skin visible through legwear:1.3), "
    "(pantyhose:1.4), (subtle sheen on legwear:1.05), "
    "(anime coloring:1.2), ")

# The `sheer` legwear: one flat black-purple tone for every costume, no
# gradient direction words; paired with SHEER_TONE_BAN.
SHEER_LEGWEAR = (
    "(sheer dark purple pantyhose:1.5), (see-through black tights:1.2), "
    "(skin clearly visible through pantyhose:1.4), "
    "(thin translucent legwear:1.3), (pantyhose:1.4), "
    "(subtle sheen on legwear:1.15), ")

WARDROBE = Wardrobe(
    costumes=COSTUMES,
    behind=BEHIND_COSTUMES,
    behind_ban=BEHIND_COSTUME_BAN,
    garment_ban={"standard": GARMENT_BLACK_BAN},
    costume_ban=COSTUME_BAN,
    hooded=frozenset({"standard"}),
    hood_ban=HOOD_BAN,
    opaque=LEGWEAR,
    sheer=SHEER_LEGWEAR,
    sheer_gloss=SHEER_GLOSS_LEGWEAR,
    sheer_tone_ban=SHEER_TONE_BAN,
    shoes=SHOES,
)
