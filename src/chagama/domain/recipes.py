"""Every recipe a request can name in `generation.recipe`: the worker, the
request check, the catalog and the CLI all read this table, so a new recipe
is one entry here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .akari.delivery import BACKDROP_PALETTE as AKARI_BACKDROP_PALETTE
from .akari.delivery import FRAME_LINE as AKARI_FRAME_LINE
from .akari.delivery import FRAME_RIM as AKARI_FRAME_RIM
from .akari.delivery import OUTLINES as AKARI_OUTLINES
from .akari.recipe import RECIPE as AKARI
from .anima.dials import DIALS
from .anima.poses import POSE_LABELS
from .anima.recipe import Recipe
from .yukari.delivery_style import (
    BACKDROP_PALETTE,
    FRAME_LINE,
    FRAME_RIM,
    OUTLINES,
    BackdropPalette,
)
from .yukari.recipe import RECIPE as YUKARI


@dataclass(frozen=True)
class RecipeEntry:
    recipe: Recipe
    # The display name every pose in `recipe.poses` must have.
    pose_labels: Mapping[str, str]
    dials: Mapping[str, Mapping[str, Mapping[str, float]]]
    rejected_parameters: frozenset[str] = frozenset()
    # The deliver bands a request that names no `outlines` gets.
    outlines: tuple[Mapping[str, object], ...] = OUTLINES
    # The colours the backdrop patterns paint a delivery of this recipe in.
    backdrop_palette: BackdropPalette = BACKDROP_PALETTE
    # The stage frame's line and rim a request that names no colour gets.
    frame_line: Mapping[str, object] = field(default_factory=lambda: dict(FRAME_LINE))
    frame_rim: Mapping[str, object] = field(default_factory=lambda: dict(FRAME_RIM))


RECIPES: Mapping[str, RecipeEntry] = {
    "yukari": RecipeEntry(YUKARI, POSE_LABELS, DIALS),
    "akari": RecipeEntry(AKARI, POSE_LABELS, DIALS, outlines=AKARI_OUTLINES,
                         backdrop_palette=AKARI_BACKDROP_PALETTE,
                         frame_line=AKARI_FRAME_LINE, frame_rim=AKARI_FRAME_RIM),
}

# What a refining request records when its source names no recipe.
DEFAULT_RECIPE = "yukari"


def _entry(recipe: str | None) -> RecipeEntry:
    return RECIPES.get(recipe or DEFAULT_RECIPE) or RECIPES[DEFAULT_RECIPE]


def deliver_outlines(recipe: str | None) -> list[dict]:
    """The default deliver bands for a picture drawn by `recipe`."""
    return [dict(outline) for outline in _entry(recipe).outlines]


def backdrop_palette(recipe: str | None) -> BackdropPalette:
    """The backdrop pattern colours for a picture drawn by `recipe`."""
    return _entry(recipe).backdrop_palette


def frame_bands(recipe: str | None) -> tuple[dict, dict]:
    """The default stage frame (line, rim) for a picture drawn by `recipe`."""
    entry = _entry(recipe)
    return dict(entry.frame_line), dict(entry.frame_rim)
