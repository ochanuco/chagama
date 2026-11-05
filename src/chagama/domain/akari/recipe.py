"""The akari recipe: the Anima style drawing `AKARI` in the shared poses
and expressions."""

from __future__ import annotations

from ..anima.expressions import EXPRESSIONS
from ..anima.poses import POSES
from ..anima.recipe import Recipe
from .character import AKARI

RECIPE = Recipe.drawing(AKARI, POSES, EXPRESSIONS)
