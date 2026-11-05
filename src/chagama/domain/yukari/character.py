"""Yuzuki Yukari as an Anima `Character`."""

from __future__ import annotations

from ..anima.character import Character
from .costumes import WARDROBE
from .prompt_style import (BODY, CHARACTER_TAG, EYE_BAN, EYE_SHAPE, FACE,
                           IDENTITY, IDENTITY_TAG_NAMES, LINE_BAN,
                           MECHANICAL_BAN, NAILS, SERIES_TAG, VIVID_BAN)

YUKARI = Character(
    tag=CHARACTER_TAG,
    series=SERIES_TAG,
    identity=IDENTITY,
    identity_tag_names=IDENTITY_TAG_NAMES,
    costume="standard",
    eye_shape=EYE_SHAPE,
    nails=NAILS,
    body=BODY,
    face=FACE,
    wardrobe=WARDROBE,
    line_ban=LINE_BAN,
    eye_ban=EYE_BAN,
    palette_ban=VIVID_BAN,
    tail_ban=MECHANICAL_BAN,
    persona_pose="yukari_persona",
)
