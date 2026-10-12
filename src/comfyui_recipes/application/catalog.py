"""The recipe catalog: every pose, costume and patch target this checkout
serves, published to chimera under the worker's branch.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Protocol

from ..domain.generation.patches import (
    NUMBER_CONSTRAINTS,
    NUMBER_TARGETS,
    STRING_TARGETS,
    TEXT_OPS,
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
    LENS_CENTER_DEFAULT,
    LENS_STRENGTH,
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


_TEXT_OP_KEYS = {
    "append": ["value"], "prepend": ["value"],
    "replace": ["old", "value"], "remove": ["old"],
}


def _patches_block() -> dict:
    return {
        "keys": ["target", "op", "value", "old", "reason"],
        "text": {
            "targets": list(TEXT_TARGETS),
            "part_target": "prompt.positive.<part>",
            "ops": {op: _TEXT_OP_KEYS[op] for op in TEXT_OPS},
        },
        "number": {
            target: {"op": "set", "constraints": NUMBER_CONSTRAINTS[target]}
            for target in NUMBER_TARGETS
        },
        "string": {
            target: {"op": "set", "values": None}
            for target in STRING_TARGETS
        },
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
    }


def _lens_block() -> dict:
    return {
        "strength": LENS_STRENGTH,
        "center": {"default": list(LENS_CENTER_DEFAULT),
                   "help": "fractions [x, y] of the picture; the fisheye swells "
                           "what sits here, the inverse shrinks it"},
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
        "backdrops": _backdrops_block(),
    }


def publish_catalog(client: Management, git: dict, catalog: dict | None = None) -> dict:
    """PUT the catalog document to chimera, upserting on the worker's branch."""
    document = catalog if catalog is not None else build_catalog(git)
    return client.put_catalog(git.get("branch"), document)
