"""The yukari recipe: the Anima style drawing `YUKARI` in the shared poses
and expressions."""

from __future__ import annotations

from ..anima.expressions import EXPRESSIONS
from ..anima.poses import POSES
from ..anima.recipe import Recipe
from .character import YUKARI

RECIPE = Recipe.drawing(YUKARI, POSES, EXPRESSIONS)

_components = RECIPE.components
positive_parts = RECIPE.positive_parts
positive = RECIPE.positive
identity_tags = RECIPE.identity_tags
negative = RECIPE.negative
render_spec = RECIPE.render_spec
