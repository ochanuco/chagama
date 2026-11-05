"""Kizuna Akari as an Anima `Character`."""

from __future__ import annotations

from ..anima.character import Character
from ..anima.expression import EyeQuality
from .costumes import WARDROBE
from .looks import POSE_LOOKS
from .prompt_style import (BODY, CHARACTER_TAG, EYE_BAN, EYE_SHAPE, EYES, FACE,
                           IDENTITY, IDENTITY_TAG_NAMES, PALETTE_BAN,
                           SERIES_TAG, STYLE, TAIL_BAN)

AKARI = Character(
    tag=CHARACTER_TAG,
    series=SERIES_TAG,
    identity=IDENTITY,
    identity_tag_names=IDENTITY_TAG_NAMES,
    costume="half_jacket",
    eye_shape=EYE_SHAPE,
    eye_quality={EyeQuality.COLD: EYES, EyeQuality.BLANK: EYES},
    nails="",
    body=BODY,
    face=FACE,
    style=STYLE,
    wardrobe=WARDROBE,
    line_ban="",
    eye_ban=EYE_BAN,
    palette_ban=PALETTE_BAN,
    tail_ban=TAIL_BAN,
    persona_pose="akari_persona",
    pose_looks=POSE_LOOKS,
    expression_looks={},
)
