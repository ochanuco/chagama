"""Every recipe a request can name in `generation.recipe`: the worker, the
request check, the catalog and the CLI all read this table, so a new recipe
is one entry here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .akari.recipe import RECIPE as AKARI
from .anima.dials import DIALS
from .anima.poses import POSE_LABELS
from .anima.recipe import Recipe
from .yukari.recipe import RECIPE as YUKARI


@dataclass(frozen=True)
class RecipeEntry:
    recipe: Recipe
    # The display name every pose in `recipe.poses` must have.
    pose_labels: Mapping[str, str]
    dials: Mapping[str, Mapping[str, Mapping[str, float]]]
    rejected_parameters: frozenset[str] = frozenset()


RECIPES: Mapping[str, RecipeEntry] = {
    "yukari": RecipeEntry(YUKARI, POSE_LABELS, DIALS),
    "akari": RecipeEntry(AKARI, POSE_LABELS, DIALS),
}

# What a refining request records when its source names no recipe.
DEFAULT_RECIPE = "yukari"
