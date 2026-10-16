"""The recipe catalog: every pose, costume and patch target this checkout
serves, published to chimera under the worker's branch.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Protocol

from ..domain.generation.patches import (
    LORAS_TARGETS,
    NUMBER_CONSTRAINTS,
    NUMBER_TARGETS,
    PART_TARGET_PREFIX,
    PATCH_KEYS,
    STRING_TARGETS,
    TEXT_OP_KEYS,
    TEXT_TARGETS,
)
from ..domain.yukari.costumes import COSTUMES, LEGWEAR_STATES, LEGWEARS
from ..domain.yukari.delivery_style import (
    BACKDROP,
    BACKDROP_LABELS,
    DELIVER_DEFAULTS,
    DOF_F_NUMBER,
    DOF_SCOPE,
    DOF_VIEWFINDER,
    FRAME_COORD_RANGE,
    FRAME_LINE,
    FRAME_MARGIN_PCT,
    FRAME_PAPER,
    FRAME_POINTS_MAX,
    FRAME_RIM,
    FRAME_SIDE_CHOICES,
    FRAME_SIDE_DEFAULTS,
    FRAME_STROKE_MODES,
    FRAME_STROKE_POINTS_MAX,
    FRAME_STROKE_WIDTH_MAX_PCT,
    FRAME_STROKES_MAX,
    LENS_CENTER_DEFAULT,
    LENS_RADIUS,
    LENS_SHAPE,
    LENS_STRENGTH,
    LIGHTING_STRENGTH,
    PHOTO_BOKEH,
    PHOTO_FOCAL_LENGTH,
    PHOTO_HEIGHT,
    PHOTO_NEARER,
    PHOTO_PLACE_DEFAULT,
    PHOTO_PLACE_MAX_Y,
    PHOTO_TILT,
    LIGHT_FROM_DEFAULT,
    LIGHT_SCENES,
    OUTLINE_WIDTH_MAX_PCT,
    OUTLINES_MAX,
    REDRAW_DENOISE,
    REDRAW_SIZE,
    STROKE_CHOICES,
    STROKE_LIGHTS,
)
from ..domain.yukari.dials import DIALS
from ..domain.yukari.expressions import EXPRESSIONS
from ..domain.yukari.poses import POSES
from ..domain.yukari.recipe import PART_GROUPS, identity_tags, render_spec
from ..infrastructure.imaging.backdrops import PATTERNS as BACKDROP_PATTERNS
from ..infrastructure.imaging.backdrops import thumbnail as backdrop_thumbnail
from ..infrastructure.imaging.depth_blur import sharp_reach_per_f
from .generate import KNOWN_PARAMETERS, RECIPE_REJECTED_PARAMETERS
from .redraw import HIRES_DENOISE

SCHEMA_VERSION = 3
_SEED, _PREFIX = 0, "catalog"


class Management(Protocol):
    def put_catalog(self, recipe_ref: str, catalog: dict) -> dict: ...


def _parameters(recipe: str) -> dict:
    rejected = RECIPE_REJECTED_PARAMETERS.get(recipe, frozenset())
    return {
        "allowed": sorted(KNOWN_PARAMETERS - rejected),
        "rejected": sorted(rejected),
    }


def _yukari_recipe() -> dict:
    poses = []
    model = None
    for name in sorted(POSES):
        pose = POSES[name]
        spec = render_spec(name, _SEED, _PREFIX)
        model = spec.model_path
        poses.append({
            "name": name,
            "costume": pose.costume,
            "face": None,
            "expression": pose.expression,
            "legwear": pose.legwear_kind if pose.legwear else None,
            "legwear_state": pose.legwear_state.value if pose.legwear else None,
            "canvas": [spec.width, spec.height],
            "positive": spec.prompts.positive,
            "negative": spec.prompts.negative,
            "parts": [{"name": part_name, "text": text}
                     for part_name, text in spec.positive_parts],
        })
    return {
        "name": "yukari",
        "model": model,
        "parameters": _parameters("yukari"),
        "costumes": sorted(COSTUMES),
        "legwear": list(LEGWEARS),
        "legwear_state": list(LEGWEAR_STATES),
        "expressions": sorted(EXPRESSIONS),
        "poses": poses,
        "parts": [name for name, _ in
                 render_spec(sorted(POSES)[0], _SEED, _PREFIX).positive_parts],
        "part_groups": {name: list(members)
                        for name, members in PART_GROUPS.items()},
        "identity_tags": sorted(identity_tags(sorted(POSES)[0])),
        "dials": DIALS,
        "deliver": {
            "defaults": {key: value for key, value in DELIVER_DEFAULTS.items()
                         if key != "outlines"},
            "outlines": {"default": DELIVER_DEFAULTS["outlines"],
                         "max_count": OUTLINES_MAX,
                         "max_width": OUTLINE_WIDTH_MAX_PCT},
            "stroke_light": list(STROKE_CHOICES),
            "backdrop_color": BACKDROP,
        },
        "redraw": {
            "defaults": {
                "canvas": {"denoise": REDRAW_DENOISE, "size": REDRAW_SIZE,
                           "route": "pixel"},
                "hires": {"hires_denoise": HIRES_DENOISE},
            },
            "light": {"scenes": list(LIGHT_SCENES), "from": sorted(STROKE_LIGHTS),
                      "default_from": LIGHT_FROM_DEFAULT},
        },
    }


def _patches_block() -> dict:
    return {
        "keys": list(PATCH_KEYS),
        "text": {
            "targets": list(TEXT_TARGETS),
            "part_target": PART_TARGET_PREFIX + "<part>",
            "ops": {op: list(keys) for op, keys in TEXT_OP_KEYS.items()},
        },
        "number": {
            target: {"op": "set", "constraints": NUMBER_CONSTRAINTS[target]}
            for target in NUMBER_TARGETS
        },
        "string": {
            target: {"op": "set", "values": None}
            for target in STRING_TARGETS
        },
        "loras": {target: {"op": "set"} for target in LORAS_TARGETS},
        "overrides": {
            "identity_override": (
                "non-empty reason string; required when patches or a prompt "
                "override remove identity_tags"),
        },
    }


def _backdrops_block() -> list[dict]:
    return [
        {
            "name": name,
            "label": BACKDROP_LABELS[name],
            "thumbnail": ("data:image/png;base64,"
                         + base64.b64encode(backdrop_thumbnail(name)).decode("ascii")),
        }
        for name in BACKDROP_PATTERNS
    ]


def _dof_block() -> dict:
    return {
        "f_number": DOF_F_NUMBER,
        "scope": DOF_SCOPE,
        "viewfinder": DOF_VIEWFINDER,
        "focus": "fractions [x, y] of the delivered picture",
        "guide_radius_per_f": round(sharp_reach_per_f(), 4),
        "photo": {"stand_at": {"default": None}, "nearer": PHOTO_NEARER,
                  "tilt": PHOTO_TILT, "bokeh": PHOTO_BOKEH,
                  "focal_length": PHOTO_FOCAL_LENGTH,
                  "help": "only for a picture set in a photo by stage: the card's "
                          "depth and lean, the photo's bokeh and the lens, as the photo "
                          "block describes; scope does not apply there"},
    }


def _lens_block() -> dict:
    return {
        "strength": LENS_STRENGTH,
        "shape": LENS_SHAPE,
        "radius": {**LENS_RADIUS,
                   "help": "sphere only: the bent disc's radius as a share of "
                           "the shorter side"},
        "center": {"default": list(LENS_CENTER_DEFAULT),
                   "help": "fractions [x, y] of the picture; the fisheye swells "
                           "what sits here, the inverse shrinks it"},
    }


def _photo_block() -> dict:
    return {
        "place": {"default": list(PHOTO_PLACE_DEFAULT), "max_y": PHOTO_PLACE_MAX_Y,
                  "help": "fractions [x, y] of the photo where the middle of the "
                          "card's bottom edge (the stand's base) goes; y above 1 "
                          "puts the base below the frame, cropping the feet"},
        "height": {**PHOTO_HEIGHT, "help": "the card's height as a share of the photo's"},
        "stand_at": {"default": None,
                     "help": "a photo point [x, y] whose depth the card stands at, "
                             "e.g. a signboard to share its focus; null stands it at "
                             "its own base"},
        "nearer": {**PHOTO_NEARER, "help": "how far to pull the card toward the "
                                           "camera, in normalised scene depth"},
        "tilt": {**PHOTO_TILT, "help": "how much farther the card's top is than "
                                       "its base (negative: nearer); with a focus on "
                                       "the face the body blurs below it"},
        "focus": {"default": "subject",
                  "help": '"subject" focuses on the card (its middle when tilted); '
                          "[x, y] on that point of the picture"},
        "f_number": DOF_F_NUMBER,
        "focal_length": {**PHOTO_FOCAL_LENGTH,
                         "help": "the full-frame lens the composite is shot through; "
                                 "the card stands for a 1.5 m figure, so a closer-up "
                                 "card or a longer lens blurs the scene more"},
        "bokeh": {**PHOTO_BOKEH, "help": "how strongly the photo's out-of-focus "
                                         "highlights (street lamps) open into bright "
                                         "discs; 0 leaves them as they blur"},
    }


def _stage_block() -> dict:
    return {
        "photo": {"help": "a photo Generation's id: the delivery's figure and outline "
                          "are set in it, unblurred, at place and height"},
        "place": {"default": list(PHOTO_PLACE_DEFAULT), "max_y": PHOTO_PLACE_MAX_Y},
        "height": PHOTO_HEIGHT,
        "backdrop": {"default": DELIVER_DEFAULTS["backdrop"],
                     "help": "without a photo: a backdrops pattern name or #RRGGBB"},
        "shadow_from": {"values": sorted(STROKE_LIGHTS),
                        "help": "without a photo: the light the drop shadow is "
                                "thrown away from; null for none, absent for the "
                                "delivery's own light"},
        "light": {"values": sorted(LIGHT_SCENES),
                  "help": "without a photo: {scene, from} tints the backdrop as a "
                          "delivery in that scene does; null for none, absent for the "
                          "nearest light redraw in the lineage"},
        "frame": {
            "sides": dict(FRAME_SIDE_DEFAULTS),
            "side_values": list(FRAME_SIDE_CHOICES),
            "line": dict(FRAME_LINE),
            "rim": dict(FRAME_RIM),
            "line_width_max": OUTLINE_WIDTH_MAX_PCT,
            "paper": FRAME_PAPER,
            "margin": FRAME_MARGIN_PCT,
            "range": list(FRAME_COORD_RANGE),
            "stroke_modes": list(FRAME_STROKE_MODES),
            "stroke_width_max": FRAME_STROKE_WIDTH_MAX_PCT,
            "limits": {"strokes": FRAME_STROKES_MAX,
                       "points_per_stroke": FRAME_STROKE_POINTS_MAX,
                       "points": FRAME_POINTS_MAX},
            "help": "without a photo: set the delivery in a square panel the backdrop "
                    "fills. box and canvas are [x0, y0, x1, y1] shares of the "
                    "delivery's canvas; the rim edges both sides of the line; a cut "
                    "side clips the figure at the border's inner edge, an out side "
                    "lets it over the border, a behind side keeps it on both sides "
                    "of the border with the border over it; strokes erase or "
                    "restore the figure in order. Widths are percent of the "
                    "delivery's longest side",
        },
    }


def _lighting_block() -> dict:
    return {
        "direction": {"values": sorted(STROKE_LIGHTS), "default": None,
                      "help": "also shade the figure from this side; null for "
                              "exposure and colour only"},
        "strength": LIGHTING_STRENGTH,
        "scene": {"values": sorted(LIGHT_SCENES), "default": None,
                  "help": "light the figure as this scene (default direction nw) and "
                          "tint a pattern backdrop to it; a photo stays as it is"},
    }


def build_catalog(git: dict) -> dict:
    """The catalog document for this worker checkout. Pure -- no I/O."""
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": bool(git.get("dirty")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recipes": [_yukari_recipe()],
        "patches": _patches_block(),
        "dof": _dof_block(),
        "lens": _lens_block(),
        "photo": _photo_block(),
        "stage": _stage_block(),
        "lighting": _lighting_block(),
        "backdrops": _backdrops_block(),
    }


def publish_catalog(client: Management, git: dict, catalog: dict | None = None) -> dict:
    """PUT the catalog document to chimera, upserting on the worker's branch."""
    document = catalog if catalog is not None else build_catalog(git)
    return client.put_catalog(git.get("branch"), document)
