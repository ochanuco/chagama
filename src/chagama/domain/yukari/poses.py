"""Every yukari pose: the shared poses with her looks laid over them."""

from __future__ import annotations

from ..anima.poses import POSE_LABELS
from .character import YUKARI
from .recipe import RECIPE

POSES = RECIPE.poses
PERSONA_POSE = YUKARI.persona_pose

__all__ = ["PERSONA_POSE", "POSES", "POSE_LABELS"]
