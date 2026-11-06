"""Yuzuki Yukari as an Anima `Character`."""

from __future__ import annotations

from ..anima.character import Character
from ..anima.expression import EyeQuality
from .costumes import WARDROBE
from .looks import EXPRESSION_LOOKS, POSE_LOOKS
from .prompt_style import (BODY, CHARACTER_TAG, COLD_EYES, EYE_BAN, EYE_SHAPE,
                           FACE, IDENTITY, IDENTITY_TAG_NAMES, LINE_BAN,
                           MECHANICAL_BAN, NAILS, SERIES_TAG, VIVID_BAN)

YUKARI = Character(
    tag=CHARACTER_TAG,
    series=SERIES_TAG,
    identity=IDENTITY,
    identity_tag_names=IDENTITY_TAG_NAMES,
    costume="standard",
    eye_shape=EYE_SHAPE,
    eye_quality={EyeQuality.COLD: COLD_EYES, EyeQuality.BLANK: ""},
    nails=NAILS,
    body=BODY,
    face=FACE,
    style="",
    wardrobe=WARDROBE,
    line_ban=LINE_BAN,
    eye_ban=EYE_BAN,
    palette_ban=VIVID_BAN,
    tail_ban=MECHANICAL_BAN,
    persona_pose="yukari_persona",
    pose_looks=POSE_LOOKS,
    expression_looks=EXPRESSION_LOOKS,
)
