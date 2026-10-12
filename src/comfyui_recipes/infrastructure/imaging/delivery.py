"""Decode a ComfyUI PNG and apply Yukari's delivery backdrop and outlines."""

from __future__ import annotations

import io
import json
import string
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from ...domain.yukari import delivery_style
from . import backdrops


def image_size(data: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(data)).size


def graph_from_png(data: bytes) -> dict:
    graph = graph_from_png_or_none(data)
    if graph is None:
        raise SystemExit("PNG has no ComfyUI prompt metadata")
    return graph


def graph_from_png_or_none(data: bytes) -> dict | None:
    prompt = Image.open(io.BytesIO(data)).info.get("prompt")
    if prompt is None:
        return None
    try:
        graph = json.loads(prompt)
    except (json.JSONDecodeError, TypeError) as error:
        raise SystemExit("PNG has invalid ComfyUI prompt metadata") from error
    if not isinstance(graph, dict):
        raise SystemExit("PNG has invalid ComfyUI prompt metadata")
    return graph


def parse_color(text: str) -> tuple[int, int, int]:
    value = text.lstrip("#")
    if len(value) != 6 or any(character not in string.hexdigits
                              for character in value):
        raise SystemExit(f"expected a 6-digit hex colour, got {text!r}")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _corner_seed(pixels: np.ndarray) -> np.ndarray:
    """The backdrop colour, as the corner patch's median.

    The single pixel (0, 0) can be a lone grain spike 20+ off the field it
    sits in, and the whole flood dies against it while every corner still
    averages flat -- a delivered picture with no die-cut at all.
    """
    return np.median(pixels[:8, :8].reshape(-1, 3), axis=0)


def background_mask(pixels: np.ndarray, tolerance: int, *,
                    seed: np.ndarray | None = None) -> np.ndarray:
    """Every backdrop region that reaches the frame edge.

    `seed` defaults to the corner patch's own median.
    """
    if seed is None:
        seed = _corner_seed(pixels)
    candidates = np.abs(pixels - seed).max(axis=2) <= tolerance
    structure = ndimage.generate_binary_structure(2, 1)
    labels, _ = ndimage.label(candidates, structure=structure)
    edges = np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    reaching = np.unique(edges[edges > 0])
    return np.isin(labels, reaching)


def enclosed_mask(pixels: np.ndarray, found: np.ndarray, tolerance: int, *,
                  minimum_area: int = 16,
                  seed: np.ndarray | None = None) -> np.ndarray:
    """Backdrop the figure encloses, as regions rather than as pixels.

    Interior linework holds pixels within the tolerance of the backdrop, so
    the colour test alone claims specks along every stroke. Only components
    of at least `minimum_area` survive. `seed` is `background_mask`'s own.
    """
    if seed is None:
        seed = _corner_seed(pixels)
    candidates = (np.abs(pixels - seed).max(axis=2) <= tolerance) & ~found
    labels, count = ndimage.label(
        candidates, ndimage.generate_binary_structure(2, 2))
    if not count:
        return candidates
    sizes = ndimage.sum(candidates, labels, range(1, count + 1))
    return np.isin(labels, 1 + np.nonzero(sizes >= minimum_area)[0])


def corner_spread(data: bytes) -> float:
    """Brightness spread across the four 40px corners of a decoded PNG."""
    pixels = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    c = 40
    corners = [pixels[:c, :c], pixels[:c, -c:], pixels[-c:, :c], pixels[-c:, -c:]]
    means = [corner.reshape(-1, 3).mean() for corner in corners]
    return float(max(means) - min(means))


def down2(pixels: np.ndarray) -> np.ndarray:
    """2x2 box-downsample."""
    height, width = pixels.shape[:2]
    trimmed = pixels[:height - height % 2, :width - width % 2]
    return trimmed.reshape(
        height // 2, 2, width // 2, 2, *pixels.shape[2:]).mean(axis=(1, 3))


def local_backdrop(pixels: np.ndarray, figure: np.ndarray, band: int, *,
                   region: np.ndarray | None = None) -> np.ndarray:
    """The backdrop colour read locally, from well outside `figure`.

    A normalised Gaussian blur of the pixels the matte puts well outside the
    figure, because a bigger redraw shades the backdrop toward the figure and
    a single global sample would claim that shading as figure. `region`
    limits the pixels sampled.
    """
    outside = ~ndimage.binary_dilation(figure, iterations=band * 2)
    if region is not None:
        outside &= region
    sigma = band * 4
    weight = ndimage.gaussian_filter(outside.astype(float), sigma)
    return np.stack(
        [ndimage.gaussian_filter(pixels[..., c] * outside, sigma)
         for c in range(3)], axis=-1) / np.maximum(weight, 1e-6)[..., None]


def refine_matte(pixels: np.ndarray, figure: np.ndarray, band: int,
                 tolerance: int) -> np.ndarray:
    """Retrace the matte's edge by colour, `band` pixels either side of it.

    The matte model loses the thin strands and the hard threshold then
    cuts what it kept into stubs; the redraw's flat backdrop makes the colour
    test exact there, against `local_backdrop`'s own read of it. Islands
    smaller than band*band are the backdrop's own grain and go.
    """
    if band < 1:
        return figure
    local = local_backdrop(pixels, figure, band)
    edge = (ndimage.binary_dilation(figure, iterations=band)
            & ~ndimage.binary_erosion(figure, iterations=band))
    refined = figure.copy()
    refined[edge] = (np.abs(pixels - local).max(axis=2) > tolerance)[edge]
    labels, count = ndimage.label(refined)
    if not count:
        return refined
    sizes = ndimage.sum(refined, labels, range(1, count + 1))
    return np.isin(labels, 1 + np.nonzero(sizes >= band * band)[0])


def soft_clamped(figure: np.ndarray, soft: np.ndarray) -> np.ndarray:
    """`refine_matte`'s figure, bounded by the matte model's own soft output.

    The colour retrace may only add a pixel the model gave any coverage and
    may not drop one it was sure of: a cast shadow drawn against the figure
    fails the colour test as backdrop, and the figure's own light passages
    pass it as figure.
    """
    return ((figure & (soft > delivery_style.MATTE_SOFT_SUPPORT))
            | (soft > delivery_style.MATTE_SOFT_CERTAIN))


def shadow_cut(pixels: np.ndarray, figure: np.ndarray, soft: np.ndarray,
               band: int) -> np.ndarray:
    """`figure` without the cast shadow it throws on the backdrop.

    The matte model reads the floor shadow under a heel or inside a curl of
    hair as figure. A shadow pixel is grey and moderately darker than the
    local backdrop (`delivery_style.SHADOW_*`), not one the model was certain
    of, and connected to the outside through other shadow pixels, so the
    figure's own greys stay. Islands the cut severs go with it.
    """
    if band < 1:
        return figure
    local = local_backdrop(pixels, figure, band)
    darker = local.mean(axis=2) - pixels.mean(axis=2)
    shadow = ((pixels.max(axis=2) - pixels.min(axis=2) < delivery_style.SHADOW_CHROMA)
              & (darker > delivery_style.SHADOW_DARK_NEAR)
              & (darker < delivery_style.SHADOW_DARK_FAR)
              & (soft <= delivery_style.MATTE_SOFT_CERTAIN))
    outside = ~figure
    reaching = ndimage.binary_propagation(outside, mask=outside | shadow)
    cut = figure & ~(shadow & reaching)
    labels, count = ndimage.label(cut)
    if not count:
        return cut
    sizes = ndimage.sum(cut, labels, range(1, count + 1))
    return np.isin(labels, 1 + np.nonzero(sizes >= band * band)[0])


def enclosed_cut(pixels: np.ndarray, figure: np.ndarray,
                 tolerance: int) -> np.ndarray:
    """`figure` without the key-coloured pockets it encloses.

    The matte model fills the gap a loop of hair closes around the backdrop
    and is certain of it, so neither the edge retrace nor `soft_clamped`
    reaches it. A no-op unless the raw backdrop (`_corner_seed`) is a green
    key (`delivery_style.ENCLOSED_KEY_MIN_GREEN_EXCESS`).
    """
    key = _corner_seed(pixels)
    if key[1] - max(key[0], key[2]) < delivery_style.ENCLOSED_KEY_MIN_GREEN_EXCESS:
        key = _pocket_key(pixels, figure)
        if key is None:
            return figure
    return figure & ~(enclosed_mask(pixels, ~figure, tolerance, seed=key)
                      | _shaded_pockets(pixels, figure, tolerance, key))


def _key_shaded(pixels: np.ndarray, key: np.ndarray, tolerance: int) -> np.ndarray:
    """Pixels within `tolerance` of `key` darkened to no less than
    `delivery_style.ENCLOSED_KEY_MIN_SHADE` of itself; the key itself included."""
    shade = np.clip((pixels * key).sum(axis=2) / max((key * key).sum(), 1.0),
                    0.0, 1.0)
    return ((np.abs(pixels - shade[..., None] * key).max(axis=2) <= tolerance)
            & (shade >= delivery_style.ENCLOSED_KEY_MIN_SHADE))


def _shaded_pockets(pixels: np.ndarray, figure: np.ndarray, tolerance: int,
                    key: np.ndarray) -> np.ndarray:
    """The shaded-key regions `figure` encloses on every side.

    The key in shadow also rings the outer edge, where the matte's soft edge
    decides; a region touching the outside is left to it."""
    eight = ndimage.generate_binary_structure(2, 2)
    candidates = figure & _key_shaded(pixels, key, tolerance)
    labels, count = ndimage.label(candidates, eight)
    if not count:
        return candidates
    sizes = ndimage.sum(candidates, labels, range(1, count + 1))
    open_ = np.unique(labels[ndimage.binary_dilation(~figure, eight) & candidates])
    keep = np.setdiff1d(1 + np.nonzero(sizes >= 16)[0], open_)
    return np.isin(labels, keep)


def key_holes(pixels: np.ndarray, figure: np.ndarray, tolerance: int) -> np.ndarray:
    """The pixels of `figure`'s holes that are the green key, shaded or not:
    backdrop for certain, however narrow the hole. Empty unless the raw
    backdrop is a green key."""
    key = _corner_seed(pixels)
    if key[1] - max(key[0], key[2]) < delivery_style.ENCLOSED_KEY_MIN_GREEN_EXCESS:
        return np.zeros(figure.shape, dtype=bool)
    holes = ndimage.binary_fill_holes(figure) & ~figure
    return holes & _key_shaded(pixels, key, tolerance)


def drawn_outline(pixels: np.ndarray, figure: np.ndarray, band: int,
                  key: np.ndarray | None = None,
                  region: np.ndarray | None = None) -> np.ndarray:
    """The white outline the raw drew around `figure`, as figure pixels.

    The outline and the key-tinted halo outside it reach the matte's edge
    through each other -- any blend of the key and white, down to the key
    itself where the matte overshoots by a pixel, or a darker shade of the
    key, the fringe a hires redraw draws around the outline; the figure's
    own line stops them. Specks they leave
    behind, figure islands under band*band pixels, are taken with them, and
    so are the pockets the figure's lines close off inside it, such as a
    gap between fingers: key-tinted regions holding some of the key itself,
    with the stray pixels they surround, under
    `delivery_style.DRAWN_OUTLINE_MAX_POCKET_BANDS` band*band pixels, when
    the key is chromatic. Empty unless they cover `delivery_style.DRAWN_OUTLINE_MIN_EDGE` of the
    edge, not counting where the figure runs off the canvas. `key` is the
    backdrop colour the halo is tinted by, the corner's by default.
    `region` limits the outline and the edge it is measured on.
    """
    empty = np.zeros(figure.shape, dtype=bool)
    if band < 1 or not figure.any():
        return empty
    if key is None:
        key = _corner_seed(pixels)
    span = np.array([255.0, 255.0, 255.0]) - key
    mix = np.clip(((pixels - key) * span).sum(axis=2) / max((span * span).sum(), 1.0),
                  0.0, 1.0)
    shade = np.clip((pixels * key).sum(axis=2) / max((key * key).sum(), 1.0),
                    0.0, 1.0)
    tinted = ((np.linalg.norm(pixels - (key + mix[..., None] * span), axis=2)
               <= delivery_style.DRAWN_OUTLINE_MAX_TINT)
              | ((np.linalg.norm(pixels - shade[..., None] * key, axis=2)
                  <= delivery_style.DRAWN_OUTLINE_MAX_TINT)
                 & (shade >= delivery_style.DRAWN_OUTLINE_MIN_KEY_SHADE)))
    rim = figure_rim(figure, band * delivery_style.DRAWN_OUTLINE_DEPTH_BANDS)
    outside = ~figure
    outline = figure & ndimage.binary_propagation(outside, mask=outside | (tinted & rim))
    edge = figure & ~ndimage.binary_erosion(figure, border_value=1)
    if region is not None:
        outline &= region
        edge &= region
    if (outline & edge).sum() < delivery_style.DRAWN_OUTLINE_MIN_EDGE * edge.sum():
        return empty
    eight = ndimage.generate_binary_structure(2, 2)
    rest = figure & ~outline
    labels, count = ndimage.label(rest, eight)
    sizes = ndimage.sum(rest, labels, range(1, count + 1))
    outline |= np.isin(labels, 1 + np.nonzero(sizes < band * band)[0])
    if _key_excess(key) < delivery_style.KEY_DESPILL_MIN_EXCESS:
        return outline
    pockets = tinted & figure & ~outline
    if region is not None:
        pockets &= region
    labels, count = ndimage.label(pockets, eight)
    if not count:
        return outline
    sizes = ndimage.sum(pockets, labels, range(1, count + 1))
    keyed = ndimage.maximum(pockets & (mix < 0.5), labels, range(1, count + 1))
    limit = band * band * delivery_style.DRAWN_OUTLINE_MAX_POCKET_BANDS
    taken = np.isin(labels, 1 + np.nonzero((sizes < limit) & (keyed > 0))[0])
    dominant, others = _key_channels(key)
    excess = pixels[..., dominant] - np.maximum(pixels[..., others[0]],
                                                pixels[..., others[1]])
    keyish = figure & (excess >= _key_excess(key) / 2)
    taken = ndimage.binary_propagation(taken, eight, mask=taken | keyish)
    return outline | (ndimage.binary_fill_holes(taken) & figure)


def _pocket_key(pixels: np.ndarray, region: np.ndarray) -> np.ndarray | None:
    """The green key found inside `region` when the corners are not it.

    A drawn frame line closes the raw's green off from the white outside it,
    so the corners read as a white backdrop while the matte keeps the whole
    green pocket as figure. The key is the median of the region's green
    pixels, once there are `ENCLOSED_POCKET_MIN_AREA` of them.
    """
    excess = pixels[..., 1] - np.maximum(pixels[..., 0], pixels[..., 2])
    inside = region & (excess >= delivery_style.ENCLOSED_KEY_MIN_GREEN_EXCESS)
    if int(inside.sum()) < delivery_style.ENCLOSED_POCKET_MIN_AREA:
        return None
    return np.median(pixels[inside], axis=0)


def frame_window(pixels: np.ndarray, figure: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray] | None:
    """The inside of a drawn frame and its key colour, else None.

    A frame leaves the canvas border almost free of key colour (under
    `delivery_style.FRAME_WINDOW_MAX_BORDER_GREEN`), unlike a green screen.
    The window green is every 8-connected key-coloured component of at
    least `delivery_style.ENCLOSED_POCKET_MIN_AREA` pixels that reaches
    `figure` dilated by `delivery_style.FRAME_WINDOW_FIGURE_REACH_PX` (any
    component without a `figure`), and must cover at least
    `delivery_style.FRAME_WINDOW_MIN_AREA_PCT` of the canvas. The window is
    that green's filled convex hull -- a tilted frame is a quad, and the
    figure can split the green -- and the key is its median colour.
    """
    green = (pixels[..., 1] - np.maximum(pixels[..., 0], pixels[..., 2])
             >= delivery_style.ENCLOSED_KEY_MIN_GREEN_EXCESS)
    border = np.concatenate([green[0], green[-1], green[:, 0], green[:, -1]])
    if border.mean() >= delivery_style.FRAME_WINDOW_MAX_BORDER_GREEN:
        return None
    labels, count = ndimage.label(green, ndimage.generate_binary_structure(2, 2))
    if not count:
        return None
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= delivery_style.ENCLOSED_POCKET_MIN_AREA
    if figure is not None:
        reach = ndimage.binary_dilation(
            figure, iterations=delivery_style.FRAME_WINDOW_FIGURE_REACH_PX)
        keep &= np.bincount(labels[reach], minlength=count + 1) > 0
    keep[0] = False
    union = keep[labels]
    if union.sum() < labels.size * delivery_style.FRAME_WINDOW_MIN_AREA_PCT / 100:
        return None
    rows, cols = np.nonzero(union)
    hull = cv2.convexHull(np.stack([cols, rows], axis=1).astype(np.int32))
    window = np.zeros(union.shape, dtype=np.uint8)
    cv2.fillConvexPoly(window, hull, 1)
    return window.astype(bool), np.median(pixels[union], axis=0)


def keyed_coverage(pixels: np.ndarray, figure: np.ndarray, local: np.ndarray,
                   band: int, tolerance: int) -> np.ndarray:
    """Figure coverage as a ramp on the figure's outermost pixel ring.

    1 inside the ring-eroded figure, 0 outside the figure entirely; the ring
    itself ramps by each pixel's own colour distance from the local
    backdrop, reaching 1 at `delivery_style.KEY_EDGE_RAMP` tolerances out.
    """
    if band < 1:
        return figure.astype(float)
    inside = ndimage.binary_erosion(
        figure, iterations=delivery_style.KEY_EDGE_RING_PX)
    distance = np.abs(pixels - local).max(axis=2)
    ramp = np.clip(distance / (delivery_style.KEY_EDGE_RAMP * tolerance), 0.0, 1.0)
    coverage = np.where(inside, 1.0, ramp)
    coverage[~figure] = 0.0
    return coverage


def unpremultiply(pixels: np.ndarray, local: np.ndarray,
                  coverage: np.ndarray) -> np.ndarray:
    """Solve for the figure's own colour under a fractional edge coverage.

    Where 0 < coverage < 1 the pixel is figure blended into the local
    backdrop at that ratio; dividing the blend's departure from the backdrop
    by coverage recovers the figure colour the blend was mixed from.
    """
    soft = (coverage > 0) & (coverage < 1)
    safe = np.where(soft, coverage, 1.0)[..., None]
    solved = np.clip(local + (pixels - local) / safe, 0, 255)
    return np.where(soft[..., None], solved, pixels)


def _key_channels(key: np.ndarray) -> tuple[int, list[int]]:
    """`key`'s dominant channel index and the other two, in channel order."""
    dominant = int(np.argmax(key))
    return dominant, [channel for channel in range(3) if channel != dominant]


def _key_excess(key: np.ndarray) -> float:
    dominant, others = _key_channels(key)
    return float(key[dominant] - max(key[others[0]], key[others[1]]))


def figure_rim(figure: np.ndarray, band: int) -> np.ndarray:
    """`figure`'s outermost `band` pixels, the only place the key can spill.

    A drawn figure has no bounced light, so the key's tint exists only where
    the edge blends into the backdrop. Despilling past that takes the key's
    chroma out of the figure's own colours: a yellow-green key turns the
    skin pink.
    """
    if band < 1:
        return figure
    return figure & ~ndimage.binary_erosion(figure, iterations=band)


def despill(pixels: np.ndarray, region: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Remove the key colour's own chroma from every `region` pixel.

    A no-op unless `key` is a chromatic key: its dominant channel must
    clear the larger of the other two by `delivery_style.KEY_DESPILL_MIN_EXCESS`.
    Each region pixel's chroma is projected onto the key's chroma direction
    and the positive part subtracted, so e.g. a teal key leaves neither
    green nor cyan behind.
    """
    if _key_excess(key) < delivery_style.KEY_DESPILL_MIN_EXCESS:
        return pixels
    direction = key - key.mean()
    direction = direction / np.linalg.norm(direction)
    chroma = pixels - pixels.mean(axis=2, keepdims=True)
    along = np.clip((chroma * direction).sum(axis=2), 0.0, None)
    cleared = np.clip(pixels - along[..., None] * direction, 0, 255)
    return np.where(region[..., None], cleared, pixels)


def stroke_alpha(mask: np.ndarray, gap: float, width: float,
                 edge_smooth: float) -> np.ndarray:
    """Coverage of the band, gap..gap+width pixels out into the backdrop.

    Distance is blurred by `edge_smooth` before ramping, so a jagged `mask`
    does not produce a jagged band. The inner ramp does nothing at gap 0
    and keeps the stroke from stepping when it is pushed away from the
    figure.
    """
    distance = ndimage.distance_transform_edt(mask)
    smoothed = ndimage.gaussian_filter(distance, edge_smooth)
    outer = np.clip(gap + width + 0.5 - smoothed, 0.0, 1.0)
    inner = np.clip(smoothed - gap + 0.5, 0.0, 1.0)
    alpha = outer * inner
    alpha[~mask] = 0.0
    return alpha


def directional_stroke_alpha(mask: np.ndarray, gap: float, w_min: float,
                             w_max: float, light: tuple[float, float],
                             smooth: float, edge_smooth: float) -> np.ndarray:
    """Coverage of a band whose width follows the outline's own normal.

    Thin where the outward normal faces `light` (image coords, x right, y
    down), thick on the opposite side. The normal comes from a
    Gaussian-blurred distance field so a hair strand or notch does not flip
    the width pixel to pixel; the edge itself ramps on a separate, smaller
    `edge_smooth` blur.
    """
    distance = ndimage.distance_transform_edt(mask)
    field = ndimage.gaussian_filter(distance, smooth)
    ny, nx = np.gradient(field)
    norm = np.hypot(nx, ny)
    norm[norm == 0] = 1.0
    facing = (nx / norm) * light[0] + (ny / norm) * light[1]
    k = (1.0 - facing) / 2.0
    k = k * k * (3 - 2 * k)
    width = w_min + (w_max - w_min) * k
    width = ndimage.gaussian_filter(width, smooth / 2)
    smoothed = ndimage.gaussian_filter(distance, edge_smooth)
    outer = np.clip(gap + width + 0.5 - smoothed, 0.0, 1.0)
    inner = np.clip(smoothed - gap + 0.5, 0.0, 1.0)
    alpha = outer * inner
    alpha[~mask] = 0.0
    return alpha


def _polygon_coverage(region: np.ndarray, eps_pct: float,
                      supersample: int = 2) -> np.ndarray:
    """Coverage of `region`'s outer shape, polygon-simplified -- the
    shape-simplifying counterpart to `STROKE_EDGE_SMOOTH`'s ramp blur.

    Holes (background the shape encloses, e.g. between an arm and the body)
    survive: only contours with a parent in the hierarchy are cleared.
    """
    height, width = region.shape
    scaled = cv2.resize(region.astype(np.uint8) * 255,
                        (width * supersample, height * supersample),
                        interpolation=cv2.INTER_LINEAR)
    contours, hierarchy = cv2.findContours(
        (scaled > 127).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    canvas = np.zeros_like(scaled)
    if contours:
        eps = max(height, width) * supersample * eps_pct / 100
        polygons = [cv2.approxPolyDP(contour, eps, True) for contour in contours]
        for index, polygon in enumerate(polygons):
            if hierarchy[0][index][3] == -1:
                cv2.fillPoly(canvas, [polygon], 255, lineType=cv2.LINE_AA)
        for index, polygon in enumerate(polygons):
            if hierarchy[0][index][3] != -1:
                cv2.fillPoly(canvas, [polygon], 0, lineType=cv2.LINE_AA)
    return cv2.resize(canvas, (width, height),
                      interpolation=cv2.INTER_AREA).astype(float) / 255


def _extruded_region(mask: np.ndarray, w_min: float, w_max: float,
                     light: tuple[float, float]) -> np.ndarray:
    """The uniform-width band at `w_min` around `mask`, extruded away from
    `light` out to `w_max`.

    The union of that band shifted 1 px at a time along `(-light)`, so the
    side facing the light keeps the `w_min` edge while the opposite side
    reads as a straight-walled extrusion -- `_polygon_coverage` is the edge
    treatment for this region, not a gaussian ramp.
    """
    base = ndimage.distance_transform_edt(~mask) <= w_min
    region = base.copy()
    length = w_max - w_min
    dx, dy = -light[0], -light[1]
    full_steps = int(np.floor(length))
    for step in range(1, full_steps + 1):
        region |= ndimage.shift(base, (dy * step, dx * step),
                                order=0, mode="constant", cval=False)
    if length > full_steps:
        region |= ndimage.shift(base, (dy * length, dx * length),
                                order=0, mode="constant", cval=False)
    return region


def _shadow_shift(light: tuple[float, float],
                  outer_w: float) -> tuple[float, float]:
    """The `(dy, dx)` an `ndimage.shift` throws a sticker's own shape by:
    straight away from `light` (image coords)."""
    off = outer_w * delivery_style.STICKER_SHADOW_OFFSET
    return -light[1] * off, -light[0] * off


def _shadow_coverage(region: np.ndarray, eps_pct: float) -> np.ndarray:
    if eps_pct <= 0:
        return region.astype(float)
    return _polygon_coverage(region, eps_pct)


class Layers(NamedTuple):
    """A delivery split into what it was composited from, back to front.

    Colours are straight (not premultiplied) 0..255 floats, alphas 0..1;
    `backdrop` is None for a transparent delivery.
    """
    picture: np.ndarray
    figure: np.ndarray
    figure_alpha: np.ndarray
    outline: np.ndarray
    outline_alpha: np.ndarray
    backdrop: np.ndarray | None
    tag: str


def keep_scene(data: bytes, matte: bytes) -> tuple[bytes, str]:
    """Deliver the redraw as drawn, background included."""
    return data, "scene"


def _outlines(outlines) -> list[dict]:
    return list(delivery_style.OUTLINES if outlines is None else outlines)


def _band_widths(height: int, width: int, outlines=None) -> list[float]:
    longest = max(height, width)
    return [longest * float(outline["width"]) / 100 for outline in _outlines(outlines)]


def _band_colours(outlines=None) -> list[np.ndarray]:
    return [np.array(parse_color(outline["color"]), dtype=float)
            for outline in _outlines(outlines)]


def band_alphas(figure: np.ndarray, outlines=None, light: str | None = None,
                eps_pct: float | None = None) -> list[np.ndarray]:
    """Coverage of each outline band outside `figure`, innermost first.

    `outlines` (default `delivery_style.OUTLINES`) lists `{"color", "width"}`
    bands; each starts at the previous band's outer edge. `eps_pct` (default
    `delivery_style.STROKE_CUT_EPS_PCT`) is the Douglas-Peucker epsilon, as a
    percent of the longest side, each band's outer outline is simplified to,
    for a hand-cut rather than die-cut edge; 0 reproduces the smooth-ramp
    geometry. `light`, one of `delivery_style.STROKE_LIGHTS`' keys, shades
    the outermost band's width by direction; `STROKE_EVEN` (or None) keeps
    it uniform. The innermost band also runs under the figure's own edge
    pixels, so a soft matting alpha there blends into the band, not the
    backdrop.
    """
    height, width = figure.shape
    widths = _band_widths(height, width, outlines)
    if light is not None and light not in delivery_style.STROKE_CHOICES:
        valid = ", ".join(repr(key) for key in delivery_style.STROKE_CHOICES)
        raise ValueError(f"light must be null or one of {valid}, got {light!r}")
    light_vec = delivery_style.STROKE_LIGHTS.get(light)
    last = len(widths) - 1

    eps = delivery_style.STROKE_CUT_EPS_PCT if eps_pct is None else eps_pct
    if eps <= 0:
        bg2 = ~(np.array(Image.fromarray(figure)
                         .resize((width * 2, height * 2), Image.NEAREST)))
        alphas, gap = [], 0.0
        for index, band_w in enumerate(widths):
            if index == last and light_vec is not None:
                alpha = directional_stroke_alpha(
                    bg2, gap * 2,
                    band_w * 2 * delivery_style.STROKE_LIGHT_THIN,
                    band_w * 2 * delivery_style.STROKE_LIGHT_THICK,
                    light_vec, delivery_style.STROKE_LIGHT_SMOOTH * band_w * 2,
                    delivery_style.STROKE_EDGE_SMOOTH)
            else:
                alpha = stroke_alpha(bg2, gap * 2, band_w * 2,
                                     delivery_style.STROKE_EDGE_SMOOTH)
            alphas.append(down2(alpha))
            gap += band_w
        return alphas

    alphas, inner = [], figure
    for index, band_w in enumerate(widths):
        if index == last and light_vec is not None:
            region = _extruded_region(
                inner, band_w * delivery_style.STROKE_LIGHT_THIN,
                band_w * delivery_style.STROKE_LIGHT_THICK, light_vec)
        else:
            region = ndimage.distance_transform_edt(~inner) <= band_w
        alpha = _polygon_coverage(region, eps)
        alphas.append(alpha)
        inner = alpha >= 0.5
    if not alphas:
        return alphas
    first = np.where(figure, 1.0, alphas[0])
    return ([np.where(ndimage.binary_erosion(figure), 0.0, first)]
            + [np.where(figure, 0.0, alpha) for alpha in alphas[1:]])


def _bands_over(alphas: list[np.ndarray], colours: list[np.ndarray],
                flat: np.ndarray) -> np.ndarray:
    bands = flat
    for alpha, colour in zip(reversed(alphas), reversed(colours)):
        bands = bands + alpha[..., None] * (colour - bands)
    return bands


def _sticker_layers(figure: np.ndarray, backdrop_rgb, shape: tuple[int, ...],
                    light: str | None, clip: np.ndarray | None, shadow: bool,
                    scene: str | None, shadow_from: str | None, outlines
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The backdrop, the bands over it, and those bands (with the shadow) as
    one straight-alpha layer whose composite over the backdrop is exactly
    the bands."""
    alphas = band_alphas(figure, outlines, light)
    if clip is not None:
        alphas = [np.where(clip, alpha, 0.0) for alpha in alphas]
    # backdrop_rgb may be a 3-vector or a full (H, W, 3) pattern; either
    # broadcasts onto the picture's shape unchanged.
    flat = np.broadcast_to(np.array(backdrop_rgb, dtype=float), shape).copy()
    shaded = flat
    shade = np.ones(shape)
    if shadow_from is None and light in delivery_style.STROKE_LIGHTS:
        shadow_from = light
    if shadow and shadow_from is not None and alphas:
        outer_w = _band_widths(*figure.shape, outlines)[-1]
        region = figure.copy()
        for alpha in alphas:
            region |= alpha >= 0.5
        shift = _shadow_shift(delivery_style.STROKE_LIGHTS[shadow_from], outer_w)
        shifted = ndimage.shift(region, shift, order=0, mode="constant", cval=False)
        shadow_a = _shadow_coverage(shifted, delivery_style.STROKE_CUT_EPS_PCT) \
            * (1 - np.clip(sum(alphas), 0.0, 1.0))
        if clip is not None:
            shadow_a = np.where(clip, shadow_a, 0.0)
        if scene is None:
            darken = 1 - shadow_a[..., None] * delivery_style.STICKER_SHADOW_DARKEN
            shaded = flat * darken
            shade = np.broadcast_to(darken, shape)
        else:
            cast = np.array(delivery_style.LIGHT_SCENES[scene]["cast"])
            weight = delivery_style.LIGHT_CAST_STRENGTH * shadow_a[..., None]
            shaded = flat * (1 - weight) + flat * cast * weight
            shade = (1 - weight) + cast * weight
    bands = _bands_over(alphas, _band_colours(outlines), shaded)
    transmission = np.ones(figure.shape)
    for alpha in alphas:
        transmission = transmission * (1 - alpha)
    outline_alpha = 1 - shade.min(axis=2) * transmission
    outline = ((bands - flat * (1 - outline_alpha)[..., None])
               / np.maximum(outline_alpha, 1e-6)[..., None])
    outline[outline_alpha <= 0.0] = 0.0
    return flat, bands, np.clip(outline, 0, 255), outline_alpha


def sticker(px: np.ndarray, figure: np.ndarray, coverage: np.ndarray,
            backdrop_rgb, light: str | None = None,
            clip: np.ndarray | None = None,
            shadow: bool = False, scene: str | None = None,
            shadow_from: str | None = None, outlines=None) -> np.ndarray:
    """Frame `figure` on `backdrop_rgb` with the `outlines` bands outside it.

    `coverage` is the figure's own per-pixel alpha in 0..1; the composite is
    coverage * px + (1 - coverage) * (the bands over the backdrop).
    `figure` alone decides where the bands sit -- coverage may be soft at
    the edge the bands are drawn from a hard boundary. `clip`, a frame
    window, keeps the bands and the shadow inside it. `shadow` throws a
    translucent drop shadow of the sticker's own shape onto the backdrop,
    away from `shadow_from`, or from `light` when that is a direction.
    """
    _, bands, _, _ = _sticker_layers(figure, backdrop_rgb, px.shape, light,
                                     clip, shadow, scene, shadow_from, outlines)
    return bands + coverage[..., None] * (px - bands)


def _backdrop_tag_suffix(backdrop: str | None) -> str:
    if not backdrop:
        return ""
    name = backdrop if backdrop in backdrops.PATTERNS else backdrop.lstrip("#")
    return f"-bg-{name}"


def _bands_tag(height: int, width: int, outlines) -> str:
    widths = _band_widths(height, width, outlines)
    return ("o" + "+".join(f"{band_w:.0f}" for band_w in widths)
            if widths else "nooutline")


def _light_tag_suffix(light: str | None, shadow: bool = False) -> str:
    if light in delivery_style.STROKE_LIGHTS:
        return f"-light-{light}" + ("-shadow" if shadow else "")
    return ""


def _cut_tag_suffix() -> str:
    """Marks a delivery's tag with the rim eps it was cut at, if any."""
    eps = delivery_style.STROKE_CUT_EPS_PCT
    return f"-cut{eps:g}" if eps > 0 else ""


def scene_backdrop(backdrop_rgb: np.ndarray, light: str | None,
                   scene: str) -> np.ndarray:
    """Tint `backdrop_rgb` to `scene` and brighten it toward the light."""
    colour, amount, value = delivery_style.LIGHT_SCENES[scene]["backdrop"]
    tinted = backdrop_rgb * ((1 - amount) + amount * np.array(colour) / 255.0) * value
    if light not in delivery_style.STROKE_LIGHTS:
        return tinted
    lx, ly = delivery_style.STROKE_LIGHTS[light]
    height, width = tinted.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    away = -(xx * lx + yy * ly)
    span = away.max() - away.min()
    far = np.clip((away - away.min()) / span, 0, 1) if span else np.zeros_like(away)
    base, slope = delivery_style.LIGHT_BACKDROP_BRIGHTEN
    return tinted * (base - slope * far)[..., None]


def fill_unkeyed_holes(px: np.ndarray, soft: np.ndarray) -> np.ndarray:
    """`soft` with the enclosed holes it punched through the figure filled
    back in where under `delivery_style.MATTE_HOLE_MIN_KEY_SHARE` of the
    hole is key-coloured. A no-op unless the raw backdrop is a green key."""
    key = _corner_seed(px)
    if key[1] - max(key[0], key[2]) < delivery_style.ENCLOSED_KEY_MIN_GREEN_EXCESS:
        return soft
    labels, count = ndimage.label(soft <= 127)
    if not count:
        return soft
    border = np.unique(np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keyed = np.abs(px - key).max(axis=2) <= delivery_style.MATTE_EDGE_TOLERANCE
    share = ndimage.mean(keyed, labels, range(1, count + 1))
    holes = 1 + np.nonzero(share < delivery_style.MATTE_HOLE_MIN_KEY_SHARE)[0]
    fill = np.isin(labels, holes) & ~np.isin(labels, border)
    return np.where(fill, 255, soft).astype(soft.dtype)


def cut_figure(px: np.ndarray, soft: np.ndarray) -> np.ndarray:
    """The silhouette the matte model's soft output and the colour retrace
    agree on, without cast shadow or enclosed key pockets."""
    soft = fill_unkeyed_holes(px, soft)
    height, width = px.shape[:2]
    band = int(max(height, width) * delivery_style.MATTE_EDGE_BAND_PCT / 100)
    tolerance = delivery_style.MATTE_EDGE_TOLERANCE
    figure = soft_clamped(refine_matte(px, soft > 127, band, tolerance), soft)
    figure = shadow_cut(px, figure, soft, band)
    return enclosed_cut(px, figure, tolerance)


def _png(array: np.ndarray, mode: str) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array, mode).save(output, "PNG")
    return output.getvalue()


def clean_layers(data: bytes, matte: bytes, light: str | None = None,
                 backdrop: str | None = None, scene: str | None = None,
                 light_from: str | None = None, matted: bool = False,
                 outlines=None) -> Layers:
    """Frame the figure the matte cuts out, in the delivery's own colours.

    `light_from` is the scene light's direction: it tints the backdrop and
    throws the drop shadow whatever `light` does to the outermost band.

    The matte is the authority on the silhouette, not colour: repin moves
    the figure's own colours, and pale hair lands inside the backdrop's
    tolerance once it has. A chromatic raw backdrop (a green screen) is
    despilled from the figure's rim.

    `matted`: `matte` is the matting stage's alpha and `data` its
    foreground, already cut and despilled; the alpha is the coverage as is.
    """
    px = np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(float)
    soft = np.array(Image.open(io.BytesIO(matte)).convert("L"))
    height, width = px.shape[:2]
    band = int(max(height, width) * delivery_style.MATTE_EDGE_BAND_PCT / 100)
    tolerance = delivery_style.MATTE_EDGE_TOLERANCE
    figure = soft >= 128 if matted else cut_figure(px, soft)
    frame = frame_window(px, figure)
    window, key = frame if frame is not None else (None, _corner_seed(px))
    inner = None if window is None else ndimage.binary_erosion(
        window, iterations=delivery_style.FRAME_WINDOW_EDGE_PX)
    raw = px
    if matted:
        outline_drawn = np.zeros(figure.shape, dtype=bool)
        coverage = soft / 255.0
    else:
        outline_drawn = drawn_outline(px, figure, band, key, inner)
        local = local_backdrop(px, figure, band, region=window)
        coverage = keyed_coverage(px, figure, local, band, tolerance)
        px = despill(unpremultiply(px, local, coverage),
                     figure_rim(figure, band), key)
        px[outline_drawn] = 255.0
        coverage[outline_drawn] = 1.0
    backdrop_rgb = backdrops.render(backdrop, height, width)
    if scene is not None:
        backdrop_rgb = scene_backdrop(backdrop_rgb, light_from or light, scene)
    flat, bands, outline, outline_alpha = _sticker_layers(
        figure if window is None else figure & inner, backdrop_rgb, px.shape,
        light, window, True, scene, light_from, outlines)
    composite = bands + coverage[..., None] * (px - bands)
    if window is not None:
        excess = raw[..., 1] - np.maximum(raw[..., 0], raw[..., 2])
        half = delivery_style.ENCLOSED_KEY_MIN_GREEN_EXCESS / 2
        green = np.clip((excess - half) / half, 0.0, 1.0)[..., None]
        outer = raw + green * (backdrop_rgb - raw)
        composite = np.where(window[..., None], composite, outer)
        flat = np.where(window[..., None], flat, outer)
        coverage = np.where(window, coverage, 0.0)

    keyed = not matted and _key_excess(key) >= delivery_style.KEY_DESPILL_MIN_EXCESS
    tag = ("clean-" + _bands_tag(height, width, outlines)
           + _backdrop_tag_suffix(backdrop) + _cut_tag_suffix()
           + ("-key" if keyed else "") + ("-matted" if matted else "")
           + ("-outline" if outline_drawn.any() else ""))
    return Layers(composite, px, coverage, outline, outline_alpha, flat,
                  tag + _light_tag_suffix(light, shadow=True))


def clean_background(data: bytes, matte: bytes, light: str | None = None,
                     backdrop: str | None = None,
                     scene: str | None = None,
                     light_from: str | None = None,
                     matted: bool = False, outlines=None) -> tuple[bytes, str]:
    """`clean_layers`' composite as a PNG, with its tag."""
    layers = clean_layers(data, matte, light, backdrop, scene, light_from,
                          matted, outlines)
    return (_png(np.clip(layers.picture, 0, 255).astype(np.uint8), "RGB"),
            layers.tag)


def transparent_layers(data: bytes, matte: bytes, light: str | None = None,
                       matted: bool = False, outlines=None) -> Layers:
    """Cut the figure out and frame it with the sticker bands on alpha 0.

    Same silhouette authority as `clean_layers`. The figure gets a
    sub-pixel ramp so retraced strands keep partial coverage, and the soft
    matte only adds coverage inside a 1-px ring around it. Outside the
    bands, alpha is 0 instead of the backdrop colour.
    `matted` takes `matte` as the matting stage's alpha, as `clean_layers`.
    """
    px = np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(np.uint8)
    soft = np.array(Image.open(io.BytesIO(matte)).convert("L"))
    height, width = px.shape[:2]
    if matted:
        return _transparent_over_bands(px, soft >= 128, soft / 255.0, light,
                                       "-matted", outlines)
    band = int(max(height, width) * delivery_style.MATTE_EDGE_BAND_PCT / 100)
    soft = fill_unkeyed_holes(px.astype(float), soft)
    figure = refine_matte(
        px.astype(float), soft > 127, band, delivery_style.MATTE_EDGE_TOLERANCE)
    figure = shadow_cut(px.astype(float), soft_clamped(figure, soft), soft, band)
    opened = enclosed_cut(
        px.astype(float), figure, delivery_style.MATTE_EDGE_TOLERANCE)
    soft = np.where(figure & ~opened, 0, soft)
    figure = opened
    outline_drawn = drawn_outline(px.astype(float), figure, band)
    px[outline_drawn] = 255

    halo = ndimage.binary_dilation(figure, iterations=1)
    ramp = ndimage.gaussian_filter(figure.astype(float), 0.6)
    coverage = np.maximum(ramp, soft / 255.0)
    coverage[~halo] = 0.0
    coverage[ndimage.binary_erosion(figure, iterations=1)] = 1.0
    coverage[outline_drawn] = 1.0
    return _transparent_over_bands(
        px, figure, coverage, light, "-outline" if outline_drawn.any() else "",
        outlines)


def transparent(data: bytes, matte: bytes, light: str | None = None,
                matted: bool = False, outlines=None) -> tuple[bytes, str]:
    """`transparent_layers`' composite as an RGBA PNG, with its tag."""
    layers = transparent_layers(data, matte, light, matted, outlines)
    return _png(layers.picture, "RGBA"), layers.tag


def _transparent_over_bands(px: np.ndarray, figure: np.ndarray,
                            coverage: np.ndarray, light: str | None,
                            suffix: str, outlines) -> Layers:
    height, width = px.shape[:2]
    alphas = band_alphas(figure, outlines, light)
    band_alpha = np.clip(sum(alphas, np.zeros(figure.shape)), 0.0, 1.0)
    band_rgb = _bands_over(alphas, _band_colours(outlines), np.zeros(px.shape))
    band_rgb = band_rgb / np.maximum(band_alpha, 1e-6)[..., None]
    # Figure over bands, straight alpha out.
    alpha = coverage + (1.0 - coverage) * band_alpha
    premultiplied = (coverage[..., None] * px
                     + ((1.0 - coverage) * band_alpha)[..., None] * band_rgb)
    rgb = premultiplied / np.maximum(alpha, 1e-6)[..., None]
    rgb[alpha <= 0.0] = 0.0

    rgba = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8),
                      np.clip(alpha * 255, 0, 255).astype(np.uint8)])
    tag = ("transparent-" + _bands_tag(height, width, outlines)
           + _cut_tag_suffix() + suffix)
    return Layers(rgba, px.astype(float), coverage, np.clip(band_rgb, 0, 255),
                  band_alpha, None, tag + _light_tag_suffix(light))


def _rgba(colour: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.dstack([np.clip(np.rint(colour), 0, 255).astype(np.uint8),
                      np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8)])


def deliver_pngs(data: bytes, matte: bytes, *, keep_scene: bool = False,
                 transparent: bool = False, light: str | None = None,
                 backdrop: str | None = None, scene: str | None = None,
                 light_from: str | None = None, matted: bool = False,
                 outlines=None) -> tuple[bytes, bytes, bytes, bytes | None, str]:
    """The delivered picture and its `figure`, `outline` and `backdrop`
    layers as PNGs, and its tag. A kept scene is its own backdrop with the
    matte's figure over it and no outline; a transparent delivery has no
    backdrop."""
    if keep_scene:
        scene_rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        alpha = np.array(Image.open(io.BytesIO(matte)).convert("L"))
        empty = np.zeros((*alpha.shape, 4), np.uint8)
        return (data, _png(np.dstack([scene_rgb, alpha]), "RGBA"),
                _png(empty, "RGBA"), _png(scene_rgb, "RGB"), "scene")
    if transparent:
        layers = transparent_layers(data, matte, light, matted, outlines)
        picture = _png(layers.picture, "RGBA")
        backdrop_png = None
    else:
        layers = clean_layers(data, matte, light, backdrop, scene, light_from,
                              matted, outlines)
        picture = _png(np.clip(layers.picture, 0, 255).astype(np.uint8), "RGB")
        backdrop_png = _png(np.clip(np.rint(layers.backdrop), 0, 255)
                            .astype(np.uint8), "RGB")
    return (picture,
            _png(_rgba(layers.figure, layers.figure_alpha), "RGBA"),
            _png(_rgba(layers.outline, layers.outline_alpha), "RGBA"),
            backdrop_png, layers.tag)
