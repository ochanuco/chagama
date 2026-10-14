"""Repair mask geometry, in the source render's own pixel space.

Rendering the region union to a mask image is not this module's concern.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# OpenPose-18 body indices.
_KNEE = {"left": 12, "right": 9}
_ANKLE = {"left": 13, "right": 10}
_ELBOW = {"left": 6, "right": 3}
_WRIST = {"left": 7, "right": 4}


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float


def _point(points: Sequence[float], index: int) -> tuple[float, float] | None:
    """A keypoint's (x, y), or None for [0, 0, 0] ('not found')."""
    x, y, c = points[index * 3:index * 3 + 3]
    if x == 0 and y == 0 and c == 0:
        return None
    return (x, y)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length else (0.0, 0.0)


def _foot_circles(body: Sequence[float], pad: float) -> list[Circle]:
    circles = []
    for side in ("left", "right"):
        knee = _point(body, _KNEE[side])
        ankle = _point(body, _ANKLE[side])
        if knee is None or ankle is None:
            continue
        shin = _dist(knee, ankle)
        circles.append(Circle(
            cx=ankle[0], cy=ankle[1] + 0.15 * shin, r=0.30 * shin * pad))
    return circles


def _hand_points(hand: Sequence[float] | None) -> list[tuple[float, float]]:
    if not hand:
        return []
    points = [_point(hand, index) for index in range(21)]
    return [point for point in points if point is not None]


def _hand_circles(body: Sequence[float], hand_left: Sequence[float] | None,
                  hand_right: Sequence[float] | None, pad: float) -> list[Circle]:
    circles = []
    for side, hand in (("left", hand_left), ("right", hand_right)):
        elbow = _point(body, _ELBOW[side])
        wrist = _point(body, _WRIST[side])
        forearm = _dist(elbow, wrist) if elbow is not None and wrist is not None else None
        points = _hand_points(hand)
        if points:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            r = 0.8 * max(x1 - x0, y1 - y0) * pad
            if forearm is not None:
                r = max(r, 0.25 * forearm * pad)
            circles.append(Circle(cx=(x0 + x1) / 2, cy=(y0 + y1) / 2, r=r))
        elif wrist is not None and elbow is not None:
            ux, uy = _unit(elbow, wrist)
            circles.append(Circle(
                cx=wrist[0] + 0.2 * forearm * ux,
                cy=wrist[1] + 0.2 * forearm * uy,
                r=0.35 * forearm * pad))
        # Neither hand keypoints nor a wrist: nothing to draw for this side.
    return circles


def regions_from_pose(pose: Mapping, parts: Sequence[str], pad: float = 1.0) -> list[Circle]:
    """Circles around the requested parts, from the first detected person."""
    people = pose.get("people") or []
    if not people:
        return []
    person = people[0]
    body = person.get("pose_keypoints_2d")
    if not body:
        return []
    circles = []
    if "feet" in parts:
        circles.extend(_foot_circles(body, pad))
    if "hands" in parts:
        circles.extend(_hand_circles(
            body, person.get("hand_left_keypoints_2d"),
            person.get("hand_right_keypoints_2d"), pad))
    return circles


def rects_from_fractions(regions: Sequence[Sequence[float]], width: float,
                         height: float) -> list[Rect]:
    """Caller-given `[x0, y0, x1, y1]` fractions (0..1) as pixel Rects."""
    return [Rect(x0 * width, y0 * height, x1 * width, y1 * height)
            for x0, y0, x1, y1 in regions]
