"""The yukari recipe: the Anima style drawing `YUKARI` in her poses and
expressions."""

from __future__ import annotations

from ..anima.recipe import Recipe
from .character import YUKARI
from .expressions import EXPRESSIONS
from .poses import POSES

RECIPE = Recipe(character=YUKARI, poses=POSES, expressions=EXPRESSIONS)

_components = RECIPE.components
positive_parts = RECIPE.positive_parts
positive = RECIPE.positive
identity_tags = RECIPE.identity_tags
negative = RECIPE.negative
render_spec = RECIPE.render_spec
