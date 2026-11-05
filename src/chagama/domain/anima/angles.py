"""Camera angle words a request can lay over a pose's own angle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AngleAxis(Enum):
    HEIGHT = "height"
    FACING = "facing"
    TILT = "tilt"
    POV = "pov"


@dataclass(frozen=True)
class Angle:
    axis: AngleAxis
    text: str


ANGLES = {
    "from_below": Angle(AngleAxis.HEIGHT, "(from below:1.3), "),
    "from_above": Angle(AngleAxis.HEIGHT, "(from above:1.35), "),
    "from_front": Angle(AngleAxis.FACING, "(from front:1.3), "),
    "from_side": Angle(AngleAxis.FACING, "(from side:1.2), "),
    "from_behind": Angle(AngleAxis.FACING, "(from behind:1.2), "),
    "dutch": Angle(AngleAxis.TILT, "(dutch angle:1.3), "),
    "pov": Angle(AngleAxis.POV, "(pov:1.2), "),
}

ANGLE_LABELS = {
    "from_below": "あおり",
    "from_above": "俯瞰",
    "from_front": "正面",
    "from_side": "横",
    "from_behind": "背面",
    "dutch": "傾き",
    "pov": "主観",
}


def angle_names(value: str | list[str]) -> tuple[str, ...]:
    names = value.split(",") if isinstance(value, str) else value
    names = tuple(name.strip() for name in names if name.strip())
    if not names:
        raise ValueError("angle は 1 語以上指定してください")
    unknown = [name for name in names if name not in ANGLES]
    if unknown:
        raise ValueError(
            f"angle {unknown} は語彙にありません。使えるのは {sorted(ANGLES)}")
    by_axis: dict[AngleAxis, list[str]] = {}
    for name in names:
        by_axis.setdefault(ANGLES[name].axis, []).append(name)
    clashes = [group for group in by_axis.values() if len(group) > 1]
    if clashes:
        raise ValueError(f"angle は軸ごとに 1 語までです: {clashes}")
    return names


def shows_back(value: str | list[str] | None) -> bool:
    return value is not None and "from_behind" in angle_names(value)


def angle_text(value: str | list[str]) -> str:
    names = angle_names(value)
    return "".join(ANGLES[name].text for axis in AngleAxis
                   for name in names if ANGLES[name].axis is axis)
