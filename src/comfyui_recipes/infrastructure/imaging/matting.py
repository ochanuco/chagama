"""Per-pixel alpha for the delivery cut: a trimap from the retraced matte,
solved by a matting model, and the figure's own colour recovered under that
alpha.

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


def figure_colour(px: np.ndarray, alpha: np.ndarray,
                  foreground: Foreground) -> np.ndarray:
    """The figure's colour under `alpha`: estimated where alpha > 0 and
    despilled of the raw key on the soft edge, the raw pixel where alpha is 0."""
    colour = np.clip(foreground(px / 255.0, alpha), 0.0, 1.0) * 255.0
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
    return _png(settle(alpha, known) * 255, "L")


def foreground_png(data: bytes, alpha: bytes, foreground: Foreground) -> bytes:
    """The figure's colour (RGB PNG) in `data` under the alpha (L PNG)."""
    px = np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(float)
    solved = np.array(Image.open(io.BytesIO(alpha)).convert("L")).astype(float) / 255.0
    return _png(figure_colour(px, solved, foreground))
