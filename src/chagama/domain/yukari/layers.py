"""The three layers every yukari prompt stands on, whatever the pose.

- `CHARACTER`: who she is -- Yuzuki Yukari's own tags and outfit. Nothing
  overrides it (the identity guard checks it).
- `CUSTOMIZATION`: the standing look this project gives her on top --
  tights, nails, build, face, eye shape, art style. A pose may override a
  field of it (`Pose.body`, `Pose.face`, `Pose.style`, `Pose.legwear*`).
- `PRODUCTION`: what the pipeline needs rather than the picture -- the
  quality/count tags and the flat green backdrop `deliver` cuts out.

The text blocks themselves live in `prompt_style.py` and `costumes.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .costumes import DEFAULT_LEGWEAR
from .prompt_style import (ARTIST_TAG, BACKGROUND, BODY, CHARACTER_TAG,
                           COUNT_TAG, EYE_SHAPE, FACE, IDENTITY, NAILS,
                           QUALITY_TAG, SERIES_TAG, STYLE)


@dataclass(frozen=True)
class Character:
    character: str
    series: str
    identity: str
    costume: str


@dataclass(frozen=True)
class Customization:
    artist: str
    eye_shape: str
    nails: str
    body: str
    face: str
    style: str
    legwear_kind: str


@dataclass(frozen=True)
class Production:
    quality: str
    count: str
    background: str


CHARACTER = Character(character=CHARACTER_TAG, series=SERIES_TAG,
                      identity=IDENTITY, costume="standard")
CUSTOMIZATION = Customization(artist=ARTIST_TAG, eye_shape=EYE_SHAPE,
                              nails=NAILS, body=BODY, face=FACE, style=STYLE,
                              legwear_kind=DEFAULT_LEGWEAR)
PRODUCTION = Production(quality=QUALITY_TAG, count=COUNT_TAG,
                        background=BACKGROUND)
