"""The trail: one stroke drawn across a staged picture, behind the figure on
some stretches and over it on others. Shared by every recipe; only the
colour is the character's own."""
from __future__ import annotations

from typing import NamedTuple


class TrailTemplate(NamedTuple):
    name: str
    label: str
    points: tuple[tuple[float, float], ...]  # four, shares of the picture
    front: tuple[bool, bool, bool, bool]


# Placed for a standing full-body figure; other poses move the points.
TRAIL_TEMPLATES = (
    TrailTemplate("tsu", "つ", ((0.02, 0.34), (0.72, 0.20), (1.00, 0.52), (0.22, 0.94)), (False, False, True, True)),
    TrailTemplate("tsu-mirror", "逆つ", ((0.98, 0.34), (0.28, 0.20), (0.00, 0.52), (0.78, 0.94)), (False, False, True, True)),
    TrailTemplate("z", "Z", ((0.10, 0.20), (0.92, 0.13), (0.06, 0.86), (0.95, 0.80)), (False, False, True, True)),
    TrailTemplate("s", "S", ((0.88, 0.12), (0.10, 0.32), (0.90, 0.66), (0.12, 0.93)), (False, False, True, True)),
    TrailTemplate("crescent", "三日月", ((0.88, 0.04), (0.02, 0.34), (0.04, 0.72), (0.82, 0.98)), (False, False, True, True)),
    TrailTemplate("diagonal", "斜め", ((0.00, 0.12), (0.33, 0.38), (0.66, 0.64), (1.00, 0.90)), (False, False, True, True)),
    TrailTemplate("feet", "足元", ((0.00, 0.70), (0.60, 0.64), (1.00, 0.82), (0.30, 0.99)), (False, False, True, True)),
    TrailTemplate("rise", "昇る", ((0.15, 0.98), (0.05, 0.62), (0.70, 0.40), (0.92, 0.06)), (True, True, False, False)),
)
TRAIL_DEFAULT_TEMPLATE = "tsu"
TRAIL_PATHS = ("curve", "line")
TRAIL_RULES = ("both", "split", "start")
TRAIL_HEADS = ("light", "sparkle")
TRAIL_WIDTH = {"min": 1.0, "max": 15.0, "default": 5.0}
TRAIL_POINT_RANGE = (-1.0, 2.0)
TRAIL_SEED_MAX = 2**31 - 1

# The default colour is the character's outer outline band, this much darker.
TRAIL_COLOR_FROM_OUTLINE = 0.88

# A night backdrop: the pattern kept at NIGHT_KEEP over NIGHT_TINT of
# NIGHT_BLUE, so a trail's light reads against it.
NIGHT_SUFFIX = "-night"
NIGHT_LABEL_SUFFIX = "（夜）"
NIGHT_BLUE = (28.0, 33.0, 64.0)
NIGHT_KEEP = 0.3
NIGHT_TINT = 0.75


def default_color(outline_outer: str) -> str:
    """The character's outer outline colour, TRAIL_COLOR_FROM_OUTLINE darker."""
    rgb = (int(outline_outer[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(v * TRAIL_COLOR_FROM_OUTLINE):02x}" for v in rgb)


def template(name: str) -> TrailTemplate | None:
    return next((t for t in TRAIL_TEMPLATES if t.name == name), None)
