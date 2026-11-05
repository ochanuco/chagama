"""The shared Anima art style: the blocks and bans every character wears,
and the render constants.

Every block that is concatenated mid-prompt ends with its own ", " -- the
assembly in `recipe.py` only ever joins strings end to end, never inserts a
separator of its own. `STYLE` is the exception: it is always the tail of the
positive prompt, so it carries no trailing comma.
"""

from __future__ import annotations

QUALITY_TAG = ""
COUNT_TAG = "1girl, solo, "
QUALITY = QUALITY_TAG + COUNT_TAG

ARTIST_TAG = "(@oshiki hitoshi:1.3), (@yoshikawa hideaki:0.5), "
BACKGROUND = "simple background, (green background:1.3), "
STYLE = "(flat color:1.3), (sketch:1.3), (traditional media:1.2)"

DIGIT_BAN = "(extra digits:1.5), bad anatomy, bad hands, "
DETAIL_BAN = ("(detailed:1.3), (intricate:1.3), (highly detailed:1.3), "
              "(fine details:1.2), ")
COLORED_LINE_BAN = "(colored lineart:1.4), (colored outline:1.3), "
THIN_BODY_BAN = ("(skinny:1.3), (thin legs:1.3), (slender legs:1.2), "
                  "(slender:1.1), ")
SHINE_BAN = ("(shiny:1.4), (glossy:1.3), (shiny hair:1.4), "
             "(shiny clothes:1.3), (specular highlights:1.3), "
             "(reflection:1.2), (hair highlights:1.2), (watercolor:1.3), "
             "(ink wash:1.3), (painterly:1.3), ")
# The garment-gloss runs inside SHINE_BAN; `sheer` and `sheer-gloss` legwear drop them.
GARMENT_GLOSS_TAGS = (
    "(shiny:1.4), (glossy:1.3), ",
    ("(shiny clothes:1.3), (specular highlights:1.3), (reflection:1.2), "),
)
GRADIENT_BAN = "(gradient:1.5), (soft shading:1.5), "
NEGATIVE_TAIL = ("(sparkling eyes:1.4), (glitter:1.3), "
                  "(multiple highlights:1.3), (gradient eyes:1.2), "
                  "(speed lines:1.45), (motion lines:1.4), "
                  "(emphasis lines:1.4), ")
SCORE_BAN = "score_1, score_2, score_3"
PROPORTION_BAN = (", (fat:1.35), (chubby:1.35), (short legs:1.35), (muscular:1.3), "
                  "(toned:1.2), (child:1.3), (loli:1.3), (chibi:1.3), (aged down:1.2)")
TEXT_BAN = (", (text:1.3), (speech bubble:1.3), (onomatopoeia:1.3), "
            "(english text:1.2), signature, watermark")

MODEL = "silvermoon-turbo.safetensors"
WIDTH, HEIGHT = 1024, 1640
STEPS = 10
CFG = 2.0
GUIDED_STEPS = 4
SAMPLER = "euler"
SCHEDULER = "normal"
HIRES_DENOISE = 0.4
