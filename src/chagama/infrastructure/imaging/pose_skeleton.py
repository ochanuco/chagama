"""The pictures the pose adapters read: the skeleton the Anima pose
adapter reads, and the silhouette, lineart or depth canvas for the any-test
LLLite and the VACE depth.

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
HEAD_POINTS = tuple(range(0, 5)) + tuple(range(23, 91))
HEAD_SCALE = 2.2
HEAD_LIFT = 0.2
SHOULDERS, HIPS = (5, 6), (11, 12)
CHEST_DROP = 0.15
CHEST_WIDTH_SCALE = 0.55
CHEST_TORSO_SCALE = 0.32

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


def outline(picture: bytes, aspect: float, width: int, height: int,
            fill: int = 255) -> np.ndarray:
    """An RGB uint8 `height` x `width` canvas: the greyscale `picture` of the
    source frame fitted the way `draw` fits the skeleton, `fill` around it."""
    decoded = cv2.imdecode(np.frombuffer(picture, np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise ValueError("the pose picture is not a readable image")
    left, top, frame_width, frame_height = frame(width, height, aspect)
    x0, y0 = round(left), round(top)
    x1, y1 = round(left + frame_width), round(top + frame_height)
    canvas = np.full((height, width), fill, np.uint8)
    canvas[y0:y1, x0:x1] = cv2.resize(decoded, (x1 - x0, y1 - y0),
                                      interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(np.repeat(canvas[..., None], 3, axis=2))


def free_mask(points: Sequence[Sequence[float]], aspect: float, width: int,
              height: int, head: bool = True, chest: bool = False) -> np.ndarray:
    """A float32 `height` x `width` mask, 1 where the control holds and 0 over
    the discs it leaves to the prompt. The head disc spreads the face and ear
    points by `HEAD_SCALE` and raises the centre by `HEAD_LIFT` of the radius,
    to take the hair above the face in. The chest disc sits `CHEST_DROP` of the
    way from the shoulders to the hips, wide enough for the shoulder span seen
    from the front and the torso seen from the side. A disc whose points are
    not seen is left out."""
    left, top, frame_width, frame_height = frame(width, height, aspect)

    def seen(indices):
        return np.array([(left + points[index][0] * frame_width,
                          top + points[index][1] * frame_height)
                         for index in indices if points[index][2] >= SCORE_THRESHOLD])

    mask = np.ones((height, width), np.float32)
    face = seen(HEAD_POINTS)
    if head and len(face):
        centre = face.mean(axis=0)
        radius = max(np.linalg.norm(face - centre, axis=1).max() * HEAD_SCALE, 1.0)
        cv2.circle(mask, (int(round(centre[0])), int(round(centre[1] - radius * HEAD_LIFT))),
                   int(round(radius)), 0.0, -1)
    shoulders, hips = seen(SHOULDERS), seen(HIPS)
    if chest and len(shoulders) == 2 and len(hips) == 2:
        top_mid, bottom_mid = shoulders.mean(axis=0), hips.mean(axis=0)
        centre = top_mid + (bottom_mid - top_mid) * CHEST_DROP
        radius = max(np.linalg.norm(shoulders[0] - shoulders[1]) * CHEST_WIDTH_SCALE,
                     np.linalg.norm(bottom_mid - top_mid) * CHEST_TORSO_SCALE, 1.0)
        cv2.circle(mask, (int(round(centre[0])), int(round(centre[1]))),
                   int(round(radius)), 0.0, -1)
    return mask


def patch_padded(skeleton: np.ndarray) -> np.ndarray:
    """`skeleton` with black added at the bottom and right up to a multiple of
    `PATCH_PIXELS` on each side."""
    height, width = skeleton.shape[:2]
    return np.pad(skeleton, ((0, -height % PATCH_PIXELS), (0, -width % PATCH_PIXELS), (0, 0)))
