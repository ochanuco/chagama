"""Set a delivery's figure and outline layers against a backdrop of their own.

A delivery cut out on alpha (or framed on its own backdrop) is set either on a
procedural pattern or flat colour at its own canvas, optionally throwing the
sticker's drop shadow onto it, or on a photo: the figure and its outline bands
are scaled together, their base on a point of the photo. Nothing is blurred
here; the layers come back as they now stand, canvas-sized, so the light and
the depth of field can work on them after."""

from __future__ import annotations

import io
from typing import NamedTuple

import numpy as np
from PIL import Image
from scipy import ndimage

from ...domain.yukari import delivery_style
from . import backdrops, photo_stage
from .delivery import _band_widths, _shadow_coverage, _shadow_shift, scene_backdrop

# The shadow's offset when the delivery has no outline bands to measure it by,
# as a share of the longest side.
BARE_SHADOW_WIDTH = 0.01


class Staged(NamedTuple):
    """Straight 0..255 colours, 0..1 alphas, all at the staged canvas."""
    picture: np.ndarray
    figure: np.ndarray
    figure_alpha: np.ndarray
    outline: np.ndarray
    outline_alpha: np.ndarray
    backdrop: np.ndarray


def premultiplied(layer: np.ndarray) -> np.ndarray:
    """A straight HxWx4 0..255 layer as premultiplied 0..1 RGBA."""
    alpha = layer[..., 3:4] / 255.0
    return np.concatenate([layer[..., :3] / 255.0 * alpha, alpha], axis=-1).astype(np.float32)


def straight(layer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A premultiplied 0..1 RGBA layer as straight 0..255 colour and alpha."""
    alpha = np.clip(layer[..., 3], 0.0, 1.0)
    colour = np.where(alpha[..., None] > 1e-6,
                      layer[..., :3] / np.maximum(alpha, 1e-6)[..., None], 0.0)
    return np.clip(colour * 255.0, 0.0, 255.0), alpha


def composite(backdrop: np.ndarray, figure: np.ndarray, figure_alpha: np.ndarray,
              outline: np.ndarray, outline_alpha: np.ndarray) -> np.ndarray:
    """The outline over the backdrop and the figure over both, 0..255."""
    picture = backdrop + outline_alpha[..., None] * (outline - backdrop)
    return picture + figure_alpha[..., None] * (figure - picture)


def _shadow(outline_alpha: np.ndarray, figure_alpha: np.ndarray, shadow_from: str,
            outlines) -> np.ndarray:
    """The coverage (0..1) of the sticker's drop shadow, thrown away from
    `shadow_from`, outside the outline bands."""
    height, width = figure_alpha.shape
    widths = _band_widths(height, width, outlines)
    outer_w = widths[-1] if widths else max(height, width) * BARE_SHADOW_WIDTH
    region = (figure_alpha >= 0.5) | (outline_alpha >= 0.5)
    shift = _shadow_shift(delivery_style.STROKE_LIGHTS[shadow_from], outer_w)
    shifted = ndimage.shift(region, shift, order=0, mode="constant", cval=False)
    return (_shadow_coverage(shifted, delivery_style.STROKE_CUT_EPS_PCT)
            * (1 - outline_alpha))


def _shadowed(outline: np.ndarray, outline_alpha: np.ndarray,
              figure_alpha: np.ndarray, shadow_from: str, outlines
              ) -> tuple[np.ndarray, np.ndarray]:
    """The outline layer with the sticker's drop shadow, thrown away from
    `shadow_from`, merged under it as translucent black."""
    shadow = (_shadow(outline_alpha, figure_alpha, shadow_from, outlines)
              * delivery_style.STICKER_SHADOW_DARKEN)
    alpha = outline_alpha + shadow * (1 - outline_alpha)
    colour = (outline * outline_alpha[..., None]) / np.maximum(alpha, 1e-6)[..., None]
    colour[alpha <= 0.0] = 0.0
    return colour, alpha


def on_pattern(figure: np.ndarray, outline: np.ndarray, backdrop: str | None,
               shadow_from: str | None = None, outlines=None, scene: str | None = None,
               light_from: str | None = None) -> Staged:
    """`figure` and `outline` (straight HxWx4 0..255) on the `backdrop`
    pattern or #RRGGBB colour (the delivery default when None) at their own
    canvas, with a drop shadow away from `shadow_from` when it names a
    direction. A light `scene` tints the backdrop toward its colour and
    brightens it toward `light_from`, as a delivery in that scene does, and
    its shadow takes the scene's cast colour, painted into the backdrop."""
    height, width = figure.shape[:2]
    figure_alpha = figure[..., 3] / 255.0
    outline_colour, outline_alpha = outline[..., :3].astype(float), outline[..., 3] / 255.0
    backdrop_rgb = backdrops.render(backdrop, height, width)
    if scene is not None:
        backdrop_rgb = scene_backdrop(backdrop_rgb, light_from, scene)
        if shadow_from is not None:
            cast = np.array(delivery_style.LIGHT_SCENES[scene]["cast"])
            weight = (delivery_style.LIGHT_CAST_STRENGTH
                      * _shadow(outline_alpha, figure_alpha, shadow_from, outlines))[..., None]
            backdrop_rgb = backdrop_rgb * (1 - weight) + backdrop_rgb * cast * weight
    elif shadow_from is not None:
        outline_colour, outline_alpha = _shadowed(
            outline_colour, outline_alpha, figure_alpha, shadow_from, outlines)
    figure_colour = figure[..., :3].astype(float)
    return Staged(composite(backdrop_rgb, figure_colour, figure_alpha,
                            outline_colour, outline_alpha),
                  figure_colour, figure_alpha, outline_colour, outline_alpha,
                  backdrop_rgb)


def on_photo(photo: np.ndarray, figure: np.ndarray, outline: np.ndarray, *,
             place: tuple[float, float], height: float) -> Staged:
    """`figure` and `outline` scaled together so their visible card is
    `height` (a share of the photo height) tall, the middle of its bottom
    edge on `place`, at the photo's canvas over `photo` (HxWx3 0..255)."""
    front, back = premultiplied(figure), premultiplied(outline)
    stacked = front[..., 3] + back[..., 3] * (1 - front[..., 3])
    rows, cols = np.nonzero(stacked > photo_stage.CARD_EDGE_ALPHA)
    if rows.size == 0:
        raise ValueError("the delivered picture has no visible figure to set in the photo")
    box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    size = photo.shape[1::-1]
    figure_colour, figure_alpha = straight(
        photo_stage.place_card(front[box], size, place, height))
    outline_colour, outline_alpha = straight(
        photo_stage.place_card(back[box], size, place, height))
    backdrop = photo[..., :3].astype(float)
    return Staged(composite(backdrop, figure_colour, figure_alpha,
                            outline_colour, outline_alpha),
                  figure_colour, figure_alpha, outline_colour, outline_alpha, backdrop)


def blur_on_photo(backdrop: np.ndarray, depth: np.ndarray, figure: np.ndarray,
                  outline: np.ndarray, *, place: tuple[float, float], height: float,
                  **blur) -> np.ndarray:
    """A photo-set delivery's layers (straight HxWx4 0..255, already placed
    by `on_photo`) blurred by the photo's depth, as `photo_stage.blur_card`
    does: the figure over its outline is the card."""
    front, back = premultiplied(figure), premultiplied(outline)
    placed = front + back * (1.0 - front[..., 3:4])
    return photo_stage.blur_card(backdrop, depth, placed, place=place, height=height,
                                 **blur)


def _decode(data: bytes, mode: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert(mode), dtype=np.float32)


def _png(array: np.ndarray, mode: str) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode).save(output, "PNG")
    return output.getvalue()


def layer_pngs(staged: Staged) -> tuple[bytes, bytes, bytes, bytes]:
    """The staged picture (RGB), figure and outline (RGBA) and backdrop
    (RGB) as PNGs."""
    return (_png(staged.picture, "RGB"),
            _png(np.dstack([staged.figure, staged.figure_alpha * 255.0]), "RGBA"),
            _png(np.dstack([staged.outline, staged.outline_alpha * 255.0]), "RGBA"),
            _png(staged.backdrop, "RGB"))


def stage_pngs(figure_png: bytes, outline_png: bytes, *, photo_png: bytes | None = None,
               backdrop: str | None = None, shadow_from: str | None = None,
               outlines=None, place: tuple[float, float] = (0.5, 1.0),
               height: float = 1.0, scene: str | None = None,
               light_from: str | None = None) -> tuple[bytes, bytes, bytes, bytes]:
    """`on_photo` when `photo_png` is given, `on_pattern` otherwise, over
    PNG bytes; returns `layer_pngs`."""
    figure, outline = _decode(figure_png, "RGBA"), _decode(outline_png, "RGBA")
    if photo_png is not None:
        staged = on_photo(_decode(photo_png, "RGB"), figure, outline,
                          place=place, height=height)
    else:
        staged = on_pattern(figure, outline, backdrop, shadow_from, outlines, scene,
                            light_from)
    return layer_pngs(staged)


def blur_on_photo_png(backdrop_png: bytes, depth_png: bytes, figure_png: bytes,
                      outline_png: bytes, **options) -> bytes:
    """`blur_on_photo` over PNG bytes; an RGB PNG at the backdrop's size."""
    rgb = blur_on_photo(_decode(backdrop_png, "RGB"), _decode(depth_png, "L"),
                        _decode(figure_png, "RGBA"), _decode(outline_png, "RGBA"),
                        **options)
    return _png(rgb, "RGB")
