"""Yukari's own blocks on the shared poses and expressions: her adult body
and face where a framing cuts her, the bare face, and her ジト目."""

from __future__ import annotations

from dataclasses import replace

from ..anima.character import ExpressionLook, PoseLook
from ..anima.poses import BANGS_UP, BARE_FACE
from .prompt_style import (EYE_SHAPE_SOFT, FACE, HAIR_ORNAMENT_BAN, IDENTITY,
                           NO_HAIR_ORNAMENT)

_CLOSE_FACE = ("(large eyes:1.15), (mature face:1.2), (tareme:1.2), "
               "(thick eyelashes:1.3), (long eyelashes:1.15), ")

_BUST = PoseLook(body=("(mature female:1.3), (adult:1.2), adult proportions, "
                       "(collarbone:1.2), "))
_NOMAKEUP = replace(
    _BUST,
    costume="roomwear",
    identity=IDENTITY.replace("hair ornament, ", NO_HAIR_ORNAMENT),
    face=FACE.replace("(thick eyelashes:1.3), (long eyelashes:1.15), ",
                      BARE_FACE),
    negative=HAIR_ORNAMENT_BAN)

POSE_LOOKS = {
    "bust": _BUST,
    "gao": PoseLook(
        body=("(mature female:1.3), (adult:1.2), (narrow waist:1.25), "
              "adult proportions, long torso, seven heads tall, "),
        face=_CLOSE_FACE),
    "avatar": PoseLook(body="(mature female:1.3), (adult:1.2), ",
                       face=_CLOSE_FACE),
    "nomakeup": _NOMAKEUP,
    "nomakeup-bangs-up": replace(_NOMAKEUP,
                                 identity=_NOMAKEUP.identity + BANGS_UP),
}

_FLAT_EYES = "(half-closed eyes:1.35), (unamused:1.35), (expressionless:1.2), "

EXPRESSION_LOOKS = {
    "disdain": ExpressionLook(
        eyes="(half-closed eyes:1.35), (cold eyes:1.3), (unamused:1.2), "),
    "dubious": ExpressionLook(
        eyes=("(half-closed eyes:1.35), (unamused:1.35), "
              "(raised eyebrow:1.25), (confused:1.15), ")),
    "jitome": ExpressionLook(eyes=_FLAT_EYES),
    "neutral": ExpressionLook(eyes=_FLAT_EYES),
    "annoyed": ExpressionLook(eye_base=EYE_SHAPE_SOFT),
    "closed": ExpressionLook(
        face_drop=frozenset({"large eyes", "big eyes", "tareme"})),
}
