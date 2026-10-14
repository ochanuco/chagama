"""Yukari's fixed prompt blocks and render constants.

Every block that is concatenated mid-prompt ends with its own ", " -- the
assembly in `recipe.py` only ever joins strings end to end, never inserts a
separator of its own. `STYLE` is the exception: it is always the tail of the
positive prompt, so it carries no trailing comma.
"""

from __future__ import annotations

QUALITY_TAG = ""
COUNT_TAG = "1girl, solo, "
QUALITY = QUALITY_TAG + COUNT_TAG

CHARACTER_TAG = "yuzuki yukari, "
SERIES_TAG = "vocaloid, voiceroid, "
ARTIST_TAG = "(@oshiki hitoshi:1.3), (@yoshikawa hideaki:0.5), "
CHARACTER = CHARACTER_TAG + SERIES_TAG + ARTIST_TAG
IDENTITY = ("light purple hair, short hair with long locks, very long "
            "sidelocks, (dark greyish purple eyes:1.4), (dark irises:1.2), "
            "hair ornament, ")
BODY = ("(mature female:1.3), (adult:1.2), (wide hips:1.2), (thick thighs:1.2), "
        "(soft thighs:1.3), (long legs:1.35), (narrow waist:1.25), "
        "adult proportions, long torso, seven heads tall, ")
NAILS = "(nail polish:1.3), (light purple nails:1.4), "
BACKGROUND = "simple background, (green background:1.3), "
# Leads every expression's eyes block: placed behind BACKGROUND (as FACE
# is) the tag stops reaching the eyes at any weight.
EYE_SHAPE = "(jitome:2.0), (tareme:1.2), (long eyes:1.25), "
FACE = ("(large eyes:1.2), (round face:1.3), (tareme:1.2), "
        "(thick eyelashes:1.3), (long eyelashes:1.15), ")
STYLE = "(flat color:1.3), (sketch:1.3), (traditional media:1.2)"

DIGIT_BAN = "(extra digits:1.5), bad anatomy, bad hands, "
DETAIL_BAN = ("(detailed:1.3), (intricate:1.3), (highly detailed:1.3), "
              "(fine details:1.2), ")
COLORED_LINE_BAN = ("(colored lineart:1.4), (colored outline:1.3), "
                     "(purple lineart:1.2), ")
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
SHEER_BAN = ("(opaque legwear:1.3), (latex:1.3), (photorealistic:1.4), "
             "(realistic:1.3), (photo:1.2), (tan skin:1.35), (dark skin:1.3), "
             "(brown legwear:1.4), (brown pantyhose:1.4), (tan:1.2), "
             "(beige legwear:1.3), ")
SHEER_TONE_BAN = ("(light purple legwear:1.35), (lavender legwear:1.25), "
                  "(gradient legwear:1.3), ")
# `sheer-gloss` only: `recipe.negative` appends this once, after everything
# else.
SHEER_GLOSS_BAN = ("(glossy:1.3), (specular highlights:1.3), "
                   "(reflection:1.2), (ribbed legwear:1.3), "
                   "(vertical-striped legwear:1.3), ")
GRADIENT_BAN = "(gradient:1.5), (soft shading:1.5), "
NEGATIVE_TAIL = ("(sparkling eyes:1.4), (glitter:1.3), "
                  "(multiple highlights:1.3), (gradient eyes:1.2), "
                  "(speed lines:1.45), (motion lines:1.4), "
                  "(emphasis lines:1.4), (light purple eyes:1.3), "
                  "(bright eyes:1.2), (pink eyes:1.2), ")
HOOD_BAN = "(hood:1.3), (cardigan:1.3), "
GARMENT_BLACK_BAN = "(black jacket:1.35), (black clothes:1.3), (black hoodie:1.35), "
VIVID_BAN = ("(magenta:1.45), (pink legwear:1.45), (bright purple:1.35), "
             "(vivid colors:1.3), (neon:1.3), (red:1.3), (maroon:1.35), "
             "(wine red:1.3), ")
SCORE_BAN = "score_1, score_2, score_3"
PROPORTION_BAN = (", (fat:1.35), (chubby:1.35), (short legs:1.35), (muscular:1.3), "
                  "(toned:1.2), (child:1.3), (loli:1.3), (chibi:1.3), (aged down:1.2)")
TEXT_BAN = (", (text:1.3), (speech bubble:1.3), (onomatopoeia:1.3), "
            "(english text:1.2), signature, watermark")

MODEL = "silvermoonmixAnimaEvolved_v2329BTurbo.safetensors"
WIDTH, HEIGHT = 1024, 1640
STEPS = 10
CFG = 2.0
GUIDED_STEPS = 4
SAMPLER = "euler"
SCHEDULER = "normal"
HIRES_DENOISE = 0.4

# The delivery redraw's own line-breaking and hand/shading guards -- pass 2
# only, run through `refinement_prompt`.
DOT_BAN = ("(dotted line:1.3), (dashed line:1.3), (stipple:1.3), "
           "(halftone:1.2), ")
HAND_BAN = ("(bad hands:1.5), (mutated hands:1.5), (extra digits:1.5), "
            "(fused fingers:1.45), (long fingers:1.4), ")
SHADE_BAN = ("(detailed shading:1.5), (heavy shading:1.5), (impasto:1.45), "
             "(painterly:1.45), ")
