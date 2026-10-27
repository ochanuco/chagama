"""The skeleton picture the Anima pose adapter reads.

The adapter was trained on rtmlib's `draw_skeleton(..., openpose_skeleton=False)`
over a black BGR canvas written with OpenCV, so this draws the same COCO-WholeBody
points and links (rtmlib's `coco133`, Apache-2.0) the same way and hands the
result back as RGB.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

KEYPOINTS = 133
SCORE_THRESHOLD = 0.3
RADIUS = 2
LINE_WIDTH = 2
PATCH_PIXELS = 16

_BLUE, _GREEN, _ORANGE, _WHITE = (51, 153, 255), (0, 255, 0), (255, 128, 0), (255, 255, 255)
_PINK, _SKY, _RED = (255, 153, 255), (102, 178, 255), (255, 51, 51)
_FINGERS = (_ORANGE, _PINK, _SKY, _RED, _GREEN)


def _hand(root: int) -> tuple[list[tuple[int, int, tuple]], list[tuple]]:
    links, colors = [], []
    for finger, color in enumerate(_FINGERS):
        joints = [root] + [root + 1 + finger * 4 + step for step in range(4)]
        links += [(a, b, color) for a, b in zip(joints, joints[1:])]
        colors += [color] * 4
    return links, colors


_LEFT_HAND_LINKS, _HAND_COLORS = _hand(91)
_RIGHT_HAND_LINKS, _ = _hand(112)

POINT_COLORS: tuple[tuple[int, int, int], ...] = tuple(
    [_BLUE] * 5
    + [_GREEN if index % 2 else _ORANGE for index in range(5, 16)]
    + [_ORANGE] * 7
    + [_WHITE] * 69
    + _HAND_COLORS + [_WHITE] + _HAND_COLORS)

LINKS: tuple[tuple[int, int, tuple[int, int, int]], ...] = tuple(
    [(15, 13, _GREEN), (13, 11, _GREEN), (16, 14, _ORANGE), (14, 12, _ORANGE),
     (11, 12, _BLUE), (5, 11, _BLUE), (6, 12, _BLUE), (5, 6, _BLUE),
     (5, 7, _GREEN), (6, 8, _ORANGE), (7, 9, _GREEN), (8, 10, _ORANGE),
     (1, 2, _BLUE), (0, 1, _BLUE), (0, 2, _BLUE), (1, 3, _BLUE), (2, 4, _BLUE),
     (3, 5, _BLUE), (4, 6, _BLUE),
     (15, 17, _GREEN), (15, 18, _GREEN), (15, 19, _GREEN),
     (16, 20, _ORANGE), (16, 21, _ORANGE), (16, 22, _ORANGE)]
    + _LEFT_HAND_LINKS + _RIGHT_HAND_LINKS)


def frame(width: int, height: int, aspect: float) -> tuple[float, float, float, float]:
    """`(left, top, frame_width, frame_height)`: the source frame fitted
    whole and centred in a `width` x `height` canvas."""
    if width / height > aspect:
        frame_width, frame_height = height * aspect, float(height)
    else:
        frame_width, frame_height = float(width), width / aspect
    return (width - frame_width) / 2, (height - frame_height) / 2, frame_width, frame_height


def draw(points: Sequence[Sequence[float]], aspect: float, width: int,
         height: int) -> np.ndarray:
    """An RGB uint8 `height` x `width` skeleton on black."""
    if len(points) != KEYPOINTS:
        raise ValueError(f"a pose has {KEYPOINTS} points, got {len(points)}")
    left, top, frame_width, frame_height = frame(width, height, aspect)
    pixels = [(int(left + x * frame_width), int(top + y * frame_height))
              for x, y, _ in points]
    visible = [score >= SCORE_THRESHOLD for _, _, score in points]
    canvas = np.zeros((height, width, 3), np.uint8)
    for index, color in enumerate(POINT_COLORS):
        if visible[index]:
            cv2.circle(canvas, pixels[index], RADIUS, color, -1)
    for a, b, color in LINKS:
        if visible[a] and visible[b]:
            cv2.line(canvas, pixels[a], pixels[b], color, thickness=LINE_WIDTH)
    return np.ascontiguousarray(canvas[..., ::-1])


def patch_padded(skeleton: np.ndarray) -> np.ndarray:
    """`skeleton` with black added at the bottom and right up to a multiple of
    `PATCH_PIXELS` on each side."""
    height, width = skeleton.shape[:2]
    return np.pad(skeleton, ((0, -height % PATCH_PIXELS), (0, -width % PATCH_PIXELS), (0, 0)))
