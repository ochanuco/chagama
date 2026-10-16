"""Per-pixel alpha for the delivery cut: a trimap from the retraced matte,
solved by a matting model, see-through clothing unmixed from the green key,
and the figure's own colour recovered under that alpha.

The model and the foreground estimator are passed in, so this module stays
numpy-only; the worker's node pack supplies both.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image
from scipy import ndimage

from ...domain.yukari import delivery_style
from . import delivery

# (rgb uint8 HxWx3, trimap uint8 HxW of 0/128/255) -> alpha float HxW in 0..1
Predict = Callable[[np.ndarray, np.ndarray], np.ndarray]
# (rgb float HxWx3 in 0..1, alpha float HxW) -> foreground float HxWx3 in 0..1
Foreground = Callable[[np.ndarray, np.ndarray], np.ndarray]

KNOWN_FIGURE = 255
UNKNOWN = 128
KNOWN_BACKDROP = 0


def trimap(figure: np.ndarray, width: int) -> np.ndarray:
    known = np.full(figure.shape, UNKNOWN, dtype=np.uint8)
    known[ndimage.binary_erosion(figure, iterations=width)] = KNOWN_FIGURE
    known[~ndimage.binary_dilation(figure, iterations=width)] = KNOWN_BACKDROP
    return known


def _starts(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, step))
    return starts + [length - tile]


def _feather(size: int, overlap: int) -> np.ndarray:
    index = np.arange(size, dtype=float)
    return np.clip(np.minimum(index + 1, size - index) / max(overlap, 1), 0.0, 1.0)


def tiled(predict: Predict, rgb: np.ndarray, known: np.ndarray, tile: int,
          overlap: int) -> np.ndarray:
    """`predict` over the tiles that hold unknown pixels, feathered together.
    Pixels no tile visits keep the trimap's own 0 or 1."""
    height, width = known.shape
    unknown = known == UNKNOWN
    total = np.zeros(known.shape)
    weight = np.zeros(known.shape)
    step = tile - overlap
    for top in _starts(height, tile, step):
        for left in _starts(width, tile, step):
            window = (slice(top, top + tile), slice(left, left + tile))
            if not unknown[window].any():
                continue
            alpha = predict(rgb[window], known[window])
            feather = np.outer(_feather(alpha.shape[0], overlap),
                               _feather(alpha.shape[1], overlap))
            total[window] += alpha * feather
            weight[window] += feather
    alpha = (known == KNOWN_FIGURE).astype(float)
    solved = unknown & (weight > 0)
    alpha[solved] = total[solved] / weight[solved]
    return alpha


def settle(alpha: np.ndarray, known: np.ndarray) -> np.ndarray:
    """`alpha` clipped to the unit range with the trimap's known regions
    forced onto it."""
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[known == KNOWN_FIGURE] = 1.0
    alpha[known == KNOWN_BACKDROP] = 0.0
    return alpha


def green_key(px: np.ndarray) -> np.ndarray | None:
    """The raw backdrop's colour when it is a green key, else None."""
    key = delivery._corner_seed(px)
    if key[1] - max(key[0], key[2]) < delivery_style.ENCLOSED_KEY_MIN_GREEN_EXCESS:
        return None
    return key


def key_share(px: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Each pixel's green excess as a share of the key's own: 1 on the key,
    0 or less on a neutral or on any colour green does not lead."""
    excess = px[..., 1] - np.maximum(px[..., 0], px[..., 2])
    return excess / max(float(key[1] - max(key[0], key[2])), 1.0)


def translucent(px: np.ndarray, alpha: np.ndarray, known: np.ndarray) -> np.ndarray:
    """The figure pixels the green key shows through: key-tinted components
    that reach the trimap's known figure. Empty off a green key."""
    key = green_key(px)
    if key is None:
        return np.zeros(alpha.shape, dtype=bool)
    tinted = (alpha > 0) & (key_share(px, key) >= delivery_style.TRANSLUCENT_MIN_KEY_SHARE)
    labels, count = ndimage.label(tinted, ndimage.generate_binary_structure(2, 2))
    if not count:
        return tinted
    index = range(1, count + 1)
    sizes = ndimage.sum(tinted, labels, index)
    reach = ndimage.sum(known == KNOWN_FIGURE, labels, index)
    keep = 1 + np.nonzero((sizes >= delivery_style.TRANSLUCENT_MIN_AREA) & (reach > 0))[0]
    return np.isin(labels, keep)


def see_through(px: np.ndarray, alpha: np.ndarray, known: np.ndarray) -> np.ndarray:
    """`alpha` capped at 1 - key share on the `translucent` pixels: a
    material green does not lead, laid over the key at that coverage."""
    region = translucent(px, alpha, known)
    if not region.any():
        return alpha
    cap = np.clip(1.0 - key_share(px, green_key(px)), 0.0, 1.0)
    return np.where(region, np.minimum(alpha, cap), alpha)


def clearer(px: np.ndarray, alpha: np.ndarray, gain: float) -> np.ndarray:
    """`alpha` with the key showing through the see-through inside scaled by
    `gain`: 1 - gain * (1 - alpha) there. The soft edge is left as is."""
    if gain == 1.0:
        return alpha
    inside = ndimage.binary_erosion(alpha > 0, iterations=delivery_style.MATTING_TRIMAP_PX)
    known = np.where(inside, KNOWN_FIGURE, UNKNOWN).astype(np.uint8)
    region = translucent(px, alpha, known) & (alpha < 1.0)
    return np.where(region, np.clip(1.0 - gain * (1.0 - alpha), 0.0, 1.0), alpha)


def unmixed(px: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Where the green key shows through the figure's inside, the colour
    the key was blended with at `alpha`; elsewhere NaN. The soft edge, within
    the trimap band of the cut, is left to the foreground estimate."""
    solved = np.full(px.shape, np.nan)
    key = green_key(px)
    if key is None:
        return solved
    inside = ndimage.binary_erosion(alpha > 0, iterations=delivery_style.MATTING_TRIMAP_PX)
    region = (inside & (alpha < 1.0)
              & (alpha >= delivery_style.TRANSLUCENT_MIN_SOLVE_ALPHA)
              & (key_share(px, key) >= delivery_style.TRANSLUCENT_MIN_KEY_SHARE))
    safe = np.where(region, alpha, 1.0)[..., None]
    colour = np.clip(key + (px - key) / safe, 0.0, 255.0)
    return np.where(region[..., None], colour, solved)


def figure_colour(px: np.ndarray, alpha: np.ndarray,
                  foreground: Foreground) -> np.ndarray:
    """The figure's colour under `alpha`: estimated where alpha > 0 (unmixed
    from the key where it shows through) and despilled of the raw key on the
    soft edge, the raw pixel where alpha is 0."""
    colour = np.clip(foreground(px / 255.0, alpha), 0.0, 1.0) * 255.0
    solved = unmixed(px, alpha)
    colour = np.where(np.isnan(solved), colour, solved)
    edge = (alpha > 0) & ~ndimage.binary_erosion(alpha >= 1.0)
    colour = delivery.despill(colour, ndimage.binary_dilation(edge, iterations=2),
                              delivery._corner_seed(px))
    return np.where((alpha > 0)[..., None], colour, px)


def _png(array: np.ndarray, mode: str | None = None) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode).save(
        output, "PNG")
    return output.getvalue()


def known_regions(px: np.ndarray, soft: np.ndarray) -> np.ndarray:
    """The trimap of the retraced matte, with the key the figure encloses
    known backdrop: a gap narrower than the unknown band is otherwise left
    to the model, which reads it as figure."""
    figure = delivery.cut_figure(px, soft)
    known = trimap(figure, delivery_style.MATTING_TRIMAP_PX)
    known[delivery.key_holes(px, figure, delivery_style.MATTE_EDGE_TOLERANCE)] = (
        KNOWN_BACKDROP)
    return known


def alpha_png(data: bytes, matte: bytes, predict: Predict) -> bytes:
    """The alpha (L PNG) of the figure in `data`, from the matte model's soft
    output `matte`."""
    px = np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(float)
    soft = np.array(Image.open(io.BytesIO(matte)).convert("L"))
    known = known_regions(px, soft)
    alpha = tiled(predict, px.astype(np.uint8), known,
                  delivery_style.MATTING_TILE_PX,
                  delivery_style.MATTING_TILE_OVERLAP_PX)
    return _png(see_through(px, settle(alpha, known), known) * 255, "L")


def clearer_png(data: bytes, alpha: bytes, gain: float) -> bytes:
    """`clearer` over the alpha (L PNG) of the figure in `data`."""
    px = np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(float)
    solved = np.array(Image.open(io.BytesIO(alpha)).convert("L")).astype(float) / 255.0
    return _png(clearer(px, solved, gain) * 255, "L")


def cutout_png(foreground: bytes, alpha: bytes) -> bytes:
    """The figure's colour (RGB PNG) with the alpha (L PNG) as a straight
    RGBA PNG."""
    rgba = Image.open(io.BytesIO(foreground)).convert("RGB")
    rgba.putalpha(Image.open(io.BytesIO(alpha)).convert("L"))
    output = io.BytesIO()
    rgba.save(output, "PNG")
    return output.getvalue()


def foreground_png(data: bytes, alpha: bytes, foreground: Foreground) -> bytes:
    """The figure's colour (RGB PNG) in `data` under the alpha (L PNG)."""
    px = np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(float)
    solved = np.array(Image.open(io.BytesIO(alpha)).convert("L")).astype(float) / 255.0
    return _png(figure_colour(px, solved, foreground))
