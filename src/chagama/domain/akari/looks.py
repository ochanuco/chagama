"""Akari's own blocks on the shared poses: her adult body where a framing
cuts her, and the bare face. Her eyes need no per-expression look."""

from __future__ import annotations

from dataclasses import replace

from ..anima.character import PoseLook
from ..anima.poses import BANGS_UP, BARE_FACE
from .prompt_style import FACE, IDENTITY

_BUST = PoseLook(body=("(mature female:1.3), (adult:1.2), adult proportions, "
                       "(collarbone:1.2), "))
_NOMAKEUP = replace(_BUST, face=FACE + BARE_FACE)

POSE_LOOKS = {
    "bust": _BUST,
    "gao": PoseLook(
        body=("(mature female:1.3), (adult:1.2), (narrow waist:1.25), "
              "adult proportions, long torso, seven heads tall, ")),
    "avatar": PoseLook(body="(mature female:1.3), (adult:1.2), "),
    "nomakeup": _NOMAKEUP,
    "nomakeup-bangs-up": replace(_NOMAKEUP, identity=IDENTITY + BANGS_UP),
}
