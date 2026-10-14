"""The component model behind the positive prompt.

`Section` follows the Anima model card's own tag-section order; `Priority`
only orders components within `Section.GENERAL`. The sorted component
names double as `recipe.positive_parts()` and the patch targets
`patches.py` resolves.
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import NamedTuple


class Section(IntEnum):
    QUALITY = 0
    COUNT = 1
    CHARACTER = 2
    SERIES = 3
    ARTIST = 4
    GENERAL = 5


class Priority(IntEnum):
    LEAD = 0
    MAIN = 1
    TAIL = 2


class Framing(Enum):
    BUST = "bust"
    UPPER = "upper"
    COWBOY = "cowboy"
    FULL = "full"
    LYING = "lying"
    CLOSE = "close"


class Component(NamedTuple):
    name: str
    section: Section
    priority: Priority
    text: str


# Which external part (a `recipe.PART_NAMES` entry) each component name
# belongs to. A part's components must stay contiguous once `assemble` sorts.
PART_OF: dict[str, str] = {
    "quality": "quality", "count": "quality",
    "character": "identity", "series": "identity", "artist": "identity",
    "identity": "identity",
    "action": "pose",
    "mouth": "mouth",
    "mood": "mood",
    "eye_base": "eyes", "eye_quality": "eyes",
    "gesture": "gesture",
    "nails": "nails",
    "costume": "costume", "legwear": "costume",
    "framing_tags": "scene", "leg_display": "scene",
    "body_build": "body",
    "cutout": "background",
    "face": "face",
    "style": "style",
}


def assemble(components: tuple[Component, ...]) -> tuple[Component, ...]:
    return tuple(sorted(components, key=lambda c: (c.section, c.priority)))


def part_groups(part_names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {name: [] for name in part_names}
    for component_name, part_name in PART_OF.items():
        groups[part_name].append(component_name)
    return {name: tuple(members) for name, members in groups.items()}
