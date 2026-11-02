"""A delivered figure as an acrylic stand standing in a photo.

The figure is printed on a clear plate cut a margin outside its silhouette,
with a tab at the bottom that sits in the slot of a clear base. One light
direction drives every cue: the front rim and the cut edge brighten where
they face it, the far rim glows on the opposite side where light leaves the
plate, the surface glare sweeps across the plate and print alike, and the
base's contact shadow falls away from it. The clear acrylic refracts and
slightly darkens the scene behind it near its edges.

Every length is a share of the printed figure's height, so the stand looks
the same at any photo size. The stand is drawn in focus; the scene behind
is blurred as photo_stage blurs it for a card focused at its base."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Image-plane direction toward the light: upper left.
LIGHT = (-0.55, -0.85)
# Where the plate's back face appears per unit of thickness: seen slightly
# from above and from the left, the edge shows along the top and right.
VIEW = (0.55, -0.85)


@dataclass(frozen=True)
class Geometry:
    margin: float = 0.022
    thickness: float = 0.014
    tab_width: float = 0.18
    tab_height: float = 0.018
    base_width: float = 0.40
    base_top: float = 0.045
    base_front: float = 0.032
    line: float = 0.0025
    bevel: float = 0.012
    refraction: float = 0.010


@dataclass(frozen=True)
class Finish:
    absorb: float = 0.05
    sheen: float = 0.14
    glare: float = 0.42
    shadow: float = 0.45


GEOMETRY = Geometry()
FINISH = Finish()


def stand_height_share(geometry: Geometry = GEOMETRY) -> float:
    """The whole stand's height per unit of printed figure height."""
    return 1 + 2 * geometry.margin + geometry.tab_height + geometry.base_front


def _disc(radius: float) -> np.ndarray:
    r = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR)


def _screen(base: np.ndarray, amount: np.ndarray, colour=(1.0, 1.0, 1.0)) -> np.ndarray:
    a = np.clip(amount, 0.0, 1.0)[..., None]
    return 1 - (1 - base) * (1 - a * np.asarray(colour, np.float32))


def _darken(base: np.ndarray, amount: np.ndarray) -> np.ndarray:
    return base * (1 - np.clip(amount, 0.0, 1.0)[..., None])


def contour(mask: np.ndarray, width: float) -> np.ndarray:
    """An anti-aliased line `width` px wide along the 0.5 level of `mask`."""
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(0.8, width * 0.6))
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3) / 8
    slope = np.hypot(gx, gy) + 1e-6
    distance = np.abs(soft - 0.5) / slope
    return np.clip(1.0 - distance / (width * 0.5 + 0.5), 0.0, 1.0) * (slope > 1e-3)


def _normals(mask: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """The outward unit normal of `mask`'s edge, smoothed over `sigma`."""
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.hypot(gx, gy) + 1e-6
    return -gx / norm, -gy / norm


def plate_shape(alpha: np.ndarray, margin: int) -> np.ndarray:
    """The plate: the figure grown by `margin`, its gaps and holes closed and
    its outline smoothed, as a 0/1 uint8 mask."""
    solid = (alpha > 0.5).astype(np.uint8)
    plate = cv2.dilate(solid, _disc(margin))
    plate = cv2.morphologyEx(plate, cv2.MORPH_CLOSE, _disc(margin * 2.5))
    plate = cv2.GaussianBlur(plate.astype(np.float32), (0, 0), max(1.0, margin * 0.6))
    plate = (plate > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(plate)
    cv2.drawContours(filled, contours, -1, 1, -1)
    return filled


def _clear(image: np.ndarray, mask: np.ndarray, nx: np.ndarray, ny: np.ndarray,
           light: np.ndarray, *, bevel: float, refraction: float,
           absorb: float) -> np.ndarray:
    """`image` seen through clear acrylic inside `mask`: shifted along the
    edge normal near the edge, slightly absorbed and cooled, brighter toward
    the rim on the lit side and darker on the other."""
    height, width = mask.shape
    inside = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    fall = np.exp(-inside / max(bevel, 1.0)) * (mask > 0.5)
    nxs = cv2.GaussianBlur(nx, (0, 0), max(bevel * 0.5, 0.5))
    nys = cv2.GaussianBlur(ny, (0, 0), max(bevel * 0.5, 0.5))
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    seen = cv2.remap(image, xs - nxs * refraction * fall, ys - nys * refraction * fall,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    body = seen * (1 - absorb) + np.array([0.0, 0.006, 0.012], np.float32)
    lit = np.clip(nxs * light[0] + nys * light[1], -1.0, 1.0)
    body = _screen(body, fall ** 2 * (0.10 + 0.25 * np.clip(lit, 0, 1)))
    body = _darken(body, fall ** 2 * 0.18 * np.clip(-lit, 0, 1))
    m = mask[..., None]
    return image * (1 - m) + body * m


def _layout(figure: np.ndarray, size: tuple[int, int], place: tuple[float, float],
            height: float, g: Geometry):
    """The figure scaled so the whole stand is `height` of the photo, its
    base's bottom middle on `place`; returns the placed RGBA 0..1 print and
    the figure's height in pixels."""
    width, photo_height = size
    alpha = figure[..., 3]
    rows, cols = np.nonzero(alpha > 0.02 * 255)
    if rows.size == 0:
        raise ValueError("the delivered picture has no visible figure to stand in the photo")
    crop = figure[rows.min():rows.max() + 1, cols.min():cols.max() + 1] / 255.0
    unit = max(8, round(height * photo_height / stand_height_share(g)))
    scale = unit / crop.shape[0]
    crop = cv2.resize(crop.astype(np.float32), (max(1, round(crop.shape[1] * scale)), unit),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    crop = np.clip(crop, 0.0, 1.0)
    margin = round(g.margin * unit)
    stand = round(unit * stand_height_share(g))
    top = round(place[1] * photo_height) - stand + margin
    left = round(place[0] * width) - crop.shape[1] // 2
    placed = np.zeros((photo_height, width, 4), np.float32)
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + crop.shape[0], photo_height), min(left + crop.shape[1], width)
    if y1 > y0 and x1 > x0:
        placed[y0:y1, x0:x1] = crop[y0 - top:y1 - top, x0 - left:x1 - left]
    return placed, unit


def stand(scene: np.ndarray, figure: np.ndarray, *, place: tuple[float, float],
          height: float, geometry: Geometry = GEOMETRY,
          finish: Finish = FINISH) -> np.ndarray:
    """The acrylic stand of the straight RGBA `figure` (0..255) over the
    already blurred `scene` (HxWx3, 0..1), the middle of its base's bottom
    on `place`, the whole stand `height` of the photo tall. Returns HxWx3
    0..1."""
    g = geometry
    photo_height, width = scene.shape[:2]
    light = np.asarray(LIGHT, np.float32) / np.linalg.norm(LIGHT)
    view = np.asarray(VIEW, np.float32) / np.linalg.norm(VIEW)
    printed, unit = _layout(figure, (width, photo_height), place, height, g)
    alpha = printed[..., 3]
    margin = max(1, round(g.margin * unit))
    thickness = max(1.0, g.thickness * unit)
    line = max(1.0, g.line * unit)
    bevel, refraction = g.bevel * unit, g.refraction * unit
    tab_w, tab_h = round(g.tab_width * unit), round(g.tab_height * unit)
    base_w, base_top = round(g.base_width * unit), round(g.base_top * unit)
    base_front = round(g.base_front * unit)
    cx = round(place[0] * width)

    plate = plate_shape(alpha, margin)
    rows = np.nonzero(plate.any(1))[0]
    if rows.size == 0:
        return scene.copy()
    plate_bottom = int(rows.max())
    for col in range(max(cx - tab_w // 2, 0), min(cx + tab_w // 2, width)):
        filled = np.nonzero(plate[:, col])[0]
        start = int(filled.max()) if filled.size else plate_bottom
        plate[start:plate_bottom + tab_h, col] = 1
    soft = cv2.GaussianBlur(plate.astype(np.float32), (0, 0), 1.6)
    plate_alpha = np.clip((soft - 0.5) * 1.6 + 0.5, 0.0, 1.0)
    nx, ny = _normals(plate, max(2.0, thickness * 0.8))
    facing = np.clip(nx * light[0] + ny * light[1], -1.0, 1.0)

    top_y = plate_bottom + tab_h - base_top // 2
    mid_y, bottom_y = top_y + base_top, top_y + base_top + base_front
    inset = round(base_w * 0.06)
    left_x, right_x = cx - base_w // 2, cx + base_w // 2
    top_face = np.zeros((photo_height, width), np.float32)
    cv2.fillPoly(top_face, [np.array([[left_x + inset, top_y], [right_x - inset, top_y],
                                      [right_x, mid_y], [left_x, mid_y]], np.int32)], 1)
    front_face = np.zeros_like(top_face)
    cv2.rectangle(front_face, (left_x, mid_y), (right_x, bottom_y), 1, -1)
    base = np.maximum(top_face, front_face)
    bnx, bny = _normals(base > 0.5, 2.0)

    image = scene.astype(np.float32).copy()
    shadow = np.zeros_like(top_face)
    offset = -light * base_front * 0.6
    cv2.ellipse(shadow, (round(cx + offset[0]), round(bottom_y + abs(offset[1]) * 0.3)),
                (round(base_w * 0.58), max(2, base_front // 3)), 0, 0, 360, 1, -1)
    image = _darken(image, finish.shadow * cv2.GaussianBlur(shadow, (0, 0), max(1.0, base_front * 0.5)))

    clear = {"bevel": bevel, "refraction": refraction}
    image = _clear(image, cv2.GaussianBlur(front_face, (0, 0), 0.7), bnx, bny, light,
                   absorb=0.16, **clear)
    image = _clear(image, cv2.GaussianBlur(top_face, (0, 0), 0.7), bnx, bny, light,
                   absorb=0.08, **clear)
    image = _screen(image, top_face * 0.10)
    tab_seen = np.zeros_like(top_face)
    cv2.rectangle(tab_seen, (cx - tab_w // 2, top_y + base_top // 3),
                  (cx + tab_w // 2, mid_y + base_front // 3), 1, -1)
    image = _darken(image, 0.10 * cv2.GaussianBlur(tab_seen, (0, 0), 1.0))
    seam = np.zeros_like(top_face)
    cv2.line(seam, (left_x, mid_y), (right_x, mid_y), 1, max(1, int(line)))
    image = _screen(image, cv2.GaussianBlur(seam, (0, 0), 0.6) * 0.75)
    base_facing = np.clip(bnx * light[0] + bny * light[1], -1.0, 1.0)
    image = _screen(image, contour(base, line) * (0.55 + 0.30 * base_facing))
    slot = np.zeros_like(top_face)
    cv2.rectangle(slot, (cx - tab_w // 2 - 2, top_y + base_top // 3),
                  (cx + tab_w // 2 + 2, top_y + base_top * 2 // 3), 1, -1)
    image = _darken(image, 0.35 * cv2.GaussianBlur(slot, (0, 0), 1.0))

    steps = max(2, int(np.ceil(thickness)))
    swept = np.zeros_like(plate_alpha)
    for k in range(1, steps + 1):
        swept = np.maximum(swept, _shift(plate_alpha, *(view * thickness * k / steps)))
    side = np.clip(swept - plate_alpha, 0.0, 1.0)
    through = _clear(image, side, nx, ny, light, absorb=0.25, **clear)
    cut_face = through * 0.78 + np.array([0.02, 0.035, 0.045], np.float32)
    image = image * (1 - side[..., None]) + cut_face * side[..., None]
    middle = np.maximum(_shift(plate_alpha, *(view * thickness * 0.45)),
                        _shift(plate_alpha, *(view * thickness * 0.55)))
    image = _screen(image, contour(middle, line * 0.8) * side
                    * (0.10 + 0.85 * np.clip(facing, 0, 1) ** 1.5))
    back_rim = contour(_shift(plate_alpha, *(view * thickness)), line) * (1 - plate_alpha)
    image = _screen(image, back_rim * (0.20 + 0.55 * np.clip(-facing, 0, 1)),
                    (0.95, 0.98, 1.0))

    image = _clear(image, plate_alpha, nx, ny, light, absorb=finish.absorb, **clear)
    image = image * (1 - alpha[..., None]) + (printed[..., :3] * 0.97 + 0.01) * alpha[..., None]

    ys, xs = np.mgrid[0:photo_height, 0:width].astype(np.float32)
    prow, pcol = np.nonzero(plate)
    centre_x, centre_y = pcol.mean(), prow.mean()
    span = float(max(np.ptp(pcol), np.ptp(prow), 1))
    along = ((xs - centre_x) * light[0] + (ys - centre_y) * light[1]) / span
    across = ((xs - centre_x) * -light[1] + (ys - centre_y) * light[0]) / span
    sheen = np.clip(along + 0.15, 0, 1) ** 1.5 * finish.sheen
    wobble = cv2.GaussianBlur(np.random.default_rng(3).standard_normal(
        (photo_height, width)).astype(np.float32), (0, 0), span * 0.02)
    wobble /= np.abs(wobble).max() + 1e-6
    glare = (np.exp(-((along - 0.10 + 0.01 * wobble) / 0.035) ** 2)
             + 0.6 * np.exp(-((along - 0.01) / 0.010) ** 2)) * finish.glare
    glare *= np.clip(1 - np.abs(across) * 1.3, 0, 1)
    image = _screen(image, plate_alpha * (sheen + glare))

    image = _screen(image, contour(plate_alpha, line)
                    * (0.18 + 0.75 * np.clip(facing, 0, 1) ** 1.2))
    inner = contour(cv2.erode(plate, _disc(line * 1.5)).astype(np.float32), line) * plate_alpha
    image = _darken(image, inner * 0.10 * (0.5 + 0.5 * np.clip(-facing, 0, 1)))
    return np.clip(image, 0.0, 1.0)
