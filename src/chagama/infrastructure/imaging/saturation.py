"""Compress a delivery's layers' saturation on repin's curve.

Each layer is read through its own alpha: the pixels with any coverage are
the range the curve works on, and the rest stay as they are. The layers are
then put back together as a delivery sets them."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from . import stage
from .palette import repin


def _decode(data: bytes, mode: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert(mode), dtype=np.float32)


def _png(array: np.ndarray, mode: str) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode).save(output, "PNG")
    return output.getvalue()


def saturate_layer(layer: np.ndarray, strength: float) -> np.ndarray:
    """A straight HxWx4 0..255 `layer` with `repin`'s curve applied where
    its alpha is above 0; its alpha is untouched."""
    covered = layer[..., 3] > 0
    if strength <= 0.0 or not covered.any():
        return layer.copy()
    rgb, _ = repin(np.clip(np.rint(layer[..., :3]), 0, 255).astype(np.uint8),
                   figure=covered, strength=strength)
    out = layer.copy()
    out[..., :3] = np.where(covered[..., None], rgb, layer[..., :3])
    return out


def recompose(figure: np.ndarray, outline: np.ndarray,
              backdrop: np.ndarray | None) -> np.ndarray:
    """The outline over the `backdrop` and the figure over both: RGB when
    there is a backdrop, else the straight RGBA of the two layers alone."""
    figure_alpha, outline_alpha = figure[..., 3] / 255.0, outline[..., 3] / 255.0
    if backdrop is not None:
        return stage.composite(backdrop.astype(float), figure[..., :3].astype(float),
                               figure_alpha, outline[..., :3].astype(float), outline_alpha)
    colour, alpha = stage.straight(stage.premultiplied(figure) + stage.premultiplied(outline)
                                   * (1.0 - figure_alpha)[..., None])
    return np.dstack([colour, alpha * 255.0])


def saturation_pngs(figure_png: bytes, outline_png: bytes, backdrop_png: bytes | None,
                    *, strength: float) -> tuple[bytes, bytes, bytes]:
    """The delivery's picture, figure and outline as PNGs after the curve;
    the picture is RGBA when there is no backdrop."""
    figure = saturate_layer(_decode(figure_png, "RGBA"), strength)
    outline = saturate_layer(_decode(outline_png, "RGBA"), strength)
    backdrop = _decode(backdrop_png, "RGB") if backdrop_png is not None else None
    picture = recompose(figure, outline, backdrop)
    return (_png(picture, "RGB" if backdrop is not None else "RGBA"),
            _png(figure, "RGBA"), _png(outline, "RGBA"))
