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
    sheen: float = 0.10
    glare: float = 0.30
    shadow: float = 0.45
    reflection: float = 0.14
    # The print as a camera sees it: taken toward the scene's exposure and
    # colour, its blacks lifted to ink on white and its whites rolled off.
    relight: float = 0.8
    ink_black: float = 0.010
    print_white: float = 0.90
    softness: float = 0.45
    grain: float = 0.006
    halation: float = 0.18


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


def _rounded(mask: np.ndarray, radius: float) -> np.ndarray:
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(radius, 0.5))
    return np.clip((soft - 0.5) * 2.5 + 0.5, 0.0, 1.0)


def _variation(shape: tuple[int, int], scale: float, seed: int) -> np.ndarray:
    """Smooth noise in 0..1 whose blobs are about `scale` px across."""
    noise = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), max(scale, 1.0))
    low, high = np.percentile(noise, (2, 98))
    return np.clip((noise - low) / max(high - low, 1e-6), 0.0, 1.0)


def _photographed_print(printed: np.ndarray, scene: np.ndarray, plate: np.ndarray,
                        finish: Finish, softness: float) -> np.ndarray:
    """The print's colour as the camera records it in this scene."""
    from . import relight  # relight reaches photo_stage through stage
    gain = relight.adjustment(relight.ambient(scene * 255.0, plate > 0), finish.relight)
    linear = np.clip(printed[..., :3], 0.0, 1.0) ** 2.2 * gain
    linear = finish.ink_black + (finish.print_white - finish.ink_black) * linear
    linear = linear / (1.0 + 0.25 * linear)
    colour = np.clip(linear / (1.0 / 1.25), 0.0, 1.0) ** (1 / 2.2)
    alpha = printed[..., 3:4]
    blurred = cv2.GaussianBlur(colour * alpha, (0, 0), softness)
    cover = cv2.GaussianBlur(alpha, (0, 0), softness)[..., None]
    return np.where(cover > 1e-3, blurred / np.maximum(cover, 1e-3), colour)


def _cast_shadow(image: np.ndarray, plate: np.ndarray, printed: np.ndarray,
                 base: np.ndarray, floor_y: int, unit: int, scale: float,
                 finish: Finish) -> np.ndarray:
    """The stand's shadow on the surface it stands on, thrown back and to the
    right, away from the light: dark under the print, faint under the clear
    acrylic with light gathered at its edge, its penumbra widening with
    distance from the base."""
    run, rise = 0.32, 0.14
    matrix = np.float32([[1, -run, run * floor_y], [0, rise, floor_y * (1 - rise)]])
    size = (image.shape[1], image.shape[0])
    plate_shadow = cv2.warpAffine(np.maximum(plate, base), matrix, size)
    print_shadow = cv2.warpAffine(printed, matrix, size)
    near = cv2.GaussianBlur(np.stack([plate_shadow, print_shadow], -1), (0, 0), 1.5 * scale)
    far = cv2.GaussianBlur(np.stack([plate_shadow, print_shadow], -1), (0, 0), 10.0 * scale)
    rows = np.arange(image.shape[0], dtype=np.float32)
    reach = np.clip((floor_y - rows) / max(rise * unit, 1.0), 0.0, 1.0)[:, None, None]
    soft = near * (1 - reach) + far * reach
    fade = (1 - 0.6 * reach[..., 0])
    darkness = (0.12 * soft[..., 0] + (finish.shadow - 0.12) * soft[..., 1]) * fade
    caustic = contour(plate_shadow, 2.0 * scale) * (1 - soft[..., 1]) * 0.10 * fade
    image = _darken(image, darkness * (rows <= floor_y + 2)[:, None])
    return _screen(image, caustic * (rows <= floor_y)[:, None])


def _window(along: np.ndarray, across: np.ndarray, span: float) -> np.ndarray:
    """A window's reflection on the flat plate: a pane with a crisp edge
    toward the light and a soft one away from it, crossed by a mullion."""
    def pane(centre, half, crisp, soft):
        d = along - centre
        return (np.clip((half - d) / crisp, 0, 1) * np.clip((d + half) / soft, 0, 1))
    glass = pane(0.30, 0.10, 0.004, 0.08) + 0.5 * pane(0.08, 0.025, 0.003, 0.03)
    mullion = 1 - 0.75 * np.exp(-((across + 0.05) / 0.008) ** 2)
    return np.clip(glass, 0, 1) * mullion


def _glints(edge: np.ndarray, facing: np.ndarray, shape: tuple[int, int],
            span: float, scale: float, seed: int = 11) -> np.ndarray:
    """Short, hard highlights where a polished edge turns to the light."""
    sparkle = _variation(shape, max(span * 0.006, 2.0 * scale), seed)
    sparse = np.clip((sparkle - 0.55) / 0.12, 0.0, 1.0)
    return np.clip(edge * np.clip(facing, 0, 1) ** 4 * sparse * 1.2, 0.0, 1.0)


def _camera(image: np.ndarray, finish: Finish, scale: float, seed: int) -> np.ndarray:
    """One camera over the whole frame: highlights bloom a little and the
    sensor grain covers stand and scene alike."""
    bright = np.clip(image - 0.72, 0.0, 1.0)
    image = image + finish.halation * cv2.GaussianBlur(bright, (0, 0), 6.0 * scale)
    rng = np.random.default_rng(seed)
    luma = rng.standard_normal(image.shape[:2]).astype(np.float32)
    chroma = rng.standard_normal(image.shape).astype(np.float32) * 0.15
    grain = cv2.GaussianBlur(luma[..., None] + chroma, (0, 0), 0.55 * scale)
    grain = grain / (grain.std() + 1e-6)
    return np.clip(image + finish.grain * grain, 0.0, 1.0)


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


def print_mask(figure: np.ndarray, size: tuple[int, int], *, place: tuple[float, float],
               height: float, geometry: Geometry = GEOMETRY) -> np.ndarray:
    """Where the printed figure sits in the photo (HxW 0..1), pulled a pixel
    inside its edge and feathered, so a redrawn frame can take the print back
    without its rim."""
    printed, _ = _layout(figure, size, place, height, geometry)
    inside = cv2.erode(printed[..., 3], _disc(1))
    return np.clip(cv2.GaussianBlur(inside, (0, 0), 1.0), 0.0, 1.0)


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
    corner = max(1.0, base_front * 0.12)
    top_face, front_face = _rounded(top_face, corner), _rounded(front_face, corner)
    base = np.maximum(top_face, front_face)
    bnx, bny = _normals(base > 0.5, 2.0)
    scale = photo_height / 1920
    prow, pcol = np.nonzero(plate)
    span = float(max(np.ptp(pcol), np.ptp(prow), 1))
    vary = 0.45 + 0.55 * _variation((photo_height, width), span * 0.04, 5)

    image = scene.astype(np.float32).copy()
    image = _cast_shadow(image, plate_alpha, alpha, base, bottom_y, unit, scale, finish)

    clear = {"bevel": bevel, "refraction": refraction}
    image = _clear(image, front_face, bnx, bny, light, absorb=0.16, **clear)
    image = _clear(image, top_face, bnx, bny, light, absorb=0.08, **clear)
    image = _screen(image, top_face * 0.06)
    ys, xs = np.mgrid[0:photo_height, 0:width].astype(np.float32)
    mirrored = cv2.remap(printed, xs, 2 * top_y - ys, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    mirrored = cv2.GaussianBlur(mirrored, (0, 0), 1.5 * max(scale, 0.5))
    seen = mirrored[..., 3] * top_face * 0.22
    image = image * (1 - seen[..., None]) + mirrored[..., :3] * seen[..., None]
    tab_seen = np.zeros_like(top_face)
    cv2.rectangle(tab_seen, (cx - tab_w // 2, top_y + base_top // 3),
                  (cx + tab_w // 2, mid_y + base_front // 3), 1, -1)
    image = _darken(image, 0.10 * cv2.GaussianBlur(tab_seen, (0, 0), 1.0))
    seam = np.zeros_like(top_face)
    cv2.line(seam, (left_x, mid_y), (right_x, mid_y), 1, max(1, int(line)))
    image = _screen(image, cv2.GaussianBlur(seam, (0, 0), 0.6) * 0.55 * vary)
    base_facing = np.clip(bnx * light[0] + bny * light[1], -1.0, 1.0)
    image = _screen(image, contour(base, line) * (0.40 + 0.30 * base_facing) * vary)
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
    image = _screen(image, contour(middle, line * 0.8) * side * vary
                    * (0.08 + 0.80 * np.clip(facing, 0, 1) ** 1.5))
    back_rim = contour(_shift(plate_alpha, *(view * thickness)), line) * (1 - plate_alpha)
    image = _screen(image, back_rim * vary * (0.15 + 0.55 * np.clip(-facing, 0, 1)),
                    (0.95, 0.98, 1.0))

    image = _clear(image, plate_alpha, nx, ny, light, absorb=finish.absorb, **clear)
    centre_x, centre_y = pcol.mean(), prow.mean()
    along = ((xs - centre_x) * light[0] + (ys - centre_y) * light[1]) / span
    across = ((xs - centre_x) * -light[1] + (ys - centre_y) * light[0]) / span
    colour = _photographed_print(printed, scene, plate, finish,
                                 finish.softness * max(scale, 0.5))
    colour = colour * (1.0 + 0.06 * np.clip(along, -1, 1))[..., None]
    image = image * (1 - alpha[..., None]) + colour * alpha[..., None]
    environment = cv2.GaussianBlur(scene[::-1], (0, 0), span * 0.03)
    image = _screen(image, plate_alpha * finish.reflection
                    * (0.5 + 0.5 * np.clip(along + 0.3, 0, 1)), environment)
    from . import relight
    light_colour = 0.4 + 0.6 * relight.highlight(scene * 255.0) / 255.0
    image = _screen(image, plate_alpha * _window(along, across, span) * finish.glare,
                    light_colour)

    image = _screen(image, contour(plate_alpha, line) * vary
                    * (0.06 + 0.30 * np.clip(facing, 0, 1) ** 2))
    glints = _glints(contour(plate_alpha, line * 1.2) + contour(middle, line) * side,
                     facing, (photo_height, width), span, scale)
    glints = np.maximum(glints, _glints(contour(base, line), base_facing,
                                        (photo_height, width), span, scale, seed=13))
    image = _screen(image, glints, light_colour)
    image = _screen(image, cv2.GaussianBlur(glints, (0, 0), 3.0 * scale) * 0.8, light_colour)
    inner = contour(cv2.erode(plate, _disc(line * 1.5)).astype(np.float32), line) * plate_alpha
    image = _darken(image, inner * 0.10 * (0.5 + 0.5 * np.clip(-facing, 0, 1)))
    return _camera(np.clip(image, 0.0, 1.0), finish, max(scale, 0.5), 7)
