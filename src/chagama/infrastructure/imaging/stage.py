"""Set a delivery's figure and outline layers against a backdrop of their own.

A delivery cut out on alpha (or framed on its own backdrop) is set either on a
procedural pattern or flat colour at its own canvas, optionally throwing the
sticker's drop shadow onto it, or in a frame: a panel of that pattern on a
paper canvas, the figure cut along the panel's sides or let out over them. Or
on a photo: the figure and its outline bands are scaled together, their base
on a point of the photo. Nothing is blurred here; the layers come back as they now stand, canvas-sized, so the light and
the depth of field can work on them after."""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ...domain.yukari import delivery_style
from . import backdrops, photo_stage
from .delivery import (
    _band_widths,
    _shadow_coverage,
    _shadow_shift,
    parse_color,
    scene_backdrop,
)

# The shadow's offset when the delivery has no outline bands to measure it by,
# as a share of the longest side.
BARE_SHADOW_WIDTH = 0.01
# A frame's pen strokes are drawn this many times finer and box-filtered down.
STROKE_SUPERSAMPLE = 4
# The alpha a figure pixel must reach to count as visible: 1 once quantised.
VISIBLE_ALPHA = 0.5 / 255


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
    return _on_backdrop(figure[..., :3].astype(float), figure[..., 3] / 255.0,
                        outline[..., :3].astype(float), outline[..., 3] / 255.0,
                        backdrops.render(backdrop, height, width), shadow_from, outlines,
                        scene, light_from)


def _on_backdrop(figure_colour: np.ndarray, figure_alpha: np.ndarray,
                 outline_colour: np.ndarray, outline_alpha: np.ndarray,
                 backdrop_rgb: np.ndarray, shadow_from: str | None, outlines,
                 scene: str | None, light_from: str | None) -> Staged:
    """The figure and outline over `backdrop_rgb`, in the light `scene` and
    with the drop shadow as `on_pattern` sets them."""
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
    return Staged(composite(backdrop_rgb, figure_colour, figure_alpha,
                            outline_colour, outline_alpha),
                  figure_colour, figure_alpha, outline_colour, outline_alpha,
                  backdrop_rgb)


def _inside(count: int, start: float | None, stop: float | None) -> np.ndarray:
    """Whether each of `count` pixel centres lies in [start, stop); None is unbounded."""
    centre = np.arange(count) + 0.5
    inside = np.ones(count, bool)
    if start is not None:
        inside &= centre >= start
    if stop is not None:
        inside &= centre < stop
    return inside


def _rect(shape: tuple[int, int], x0: float | None, y0: float | None,
          x1: float | None, y1: float | None) -> np.ndarray:
    """The pixels whose centres lie in [x0, x1) x [y0, y1)."""
    height, width = shape
    return _inside(height, y0, y1)[:, None] & _inside(width, x0, x1)[None, :]


def _paste(target: np.ndarray, source: np.ndarray, x: int, y: int) -> None:
    """Copy `source` into `target` with its top-left corner at (x, y), clipped."""
    height, width = target.shape[:2]
    left, top = max(x, 0), max(y, 0)
    right = min(x + source.shape[1], width)
    bottom = min(y + source.shape[0], height)
    if right > left and bottom > top:
        target[top:bottom, left:right] = source[top - y:bottom - y, left - x:right - x]


def _share_px(rect, width: int, height: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    return x0 * width, y0 * height, x1 * width, y1 * height


def _width_px(band: Mapping | None, longest: int) -> float:
    return longest * band["width"] / 100 if band else 0.0


def _border_half(frame: Mapping, longest: int) -> float:
    """Half the frame's border: its line and the rim edging both sides of it."""
    return _width_px(frame.get("line"), longest) / 2 + _width_px(frame.get("rim"), longest)


def _stroke_cover(shape: tuple[int, int], points: np.ndarray, diameter: float) -> np.ndarray:
    """The anti-aliased 0..1 coverage of a round pen `diameter` px across
    drawn through `points` (Nx2 px, x right / y down)."""
    height, width = shape
    radius = diameter / 2
    left = max(int(np.floor(points[:, 0].min() - radius)) - 1, 0)
    top = max(int(np.floor(points[:, 1].min() - radius)) - 1, 0)
    right = min(int(np.ceil(points[:, 0].max() + radius)) + 1, width)
    bottom = min(int(np.ceil(points[:, 1].max() + radius)) + 1, height)
    cover = np.zeros(shape, np.float32)
    if right <= left or bottom <= top:
        return cover
    scale = STROKE_SUPERSAMPLE
    pen = Image.new("L", ((right - left) * scale, (bottom - top) * scale), 0)
    draw = ImageDraw.Draw(pen)
    path = [((x - left) * scale - 0.5, (y - top) * scale - 0.5) for x, y in points]
    if len(path) > 1:
        draw.line(path, fill=255, width=max(1, round(diameter * scale)), joint="curve")
    r = radius * scale
    for x, y in (path[0], path[-1]):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    small = pen.resize((right - left, bottom - top), Image.Resampling.BOX)
    cover[top:bottom, left:right] = np.asarray(small, np.float32) / 255.0
    return cover


def frame_keep(shape: tuple[int, int], frame: Mapping) -> np.ndarray:
    """The 0..1 share of each pixel of a delivery canvas `shape` that the
    `frame` keeps: nothing past a cut side's border inner edge, then each
    stroke in order, an erase taking its coverage away and a restore
    giving it back."""
    height, width = shape
    longest = max(height, width)
    x0, y0, x1, y1 = _share_px(frame["box"], width, height)
    half = _border_half(frame, longest)
    sides = frame["sides"]
    keep = _rect(shape,
                 x0 + half if sides["left"] == "cut" else None,
                 y0 + half if sides["top"] == "cut" else None,
                 x1 - half if sides["right"] == "cut" else None,
                 y1 - half if sides["bottom"] == "cut" else None).astype(np.float32)
    for stroke in frame.get("strokes") or ():
        points = np.asarray(stroke["points"], float) * (width, height)
        cover = _stroke_cover(shape, points, longest * stroke["width"] / 100)
        if stroke["mode"] == "erase":
            keep *= 1 - cover
        else:
            keep += (1 - keep) * cover
    return keep


def frame_canvas(frame: Mapping, figure_alpha: np.ndarray,
                 outline_alpha: np.ndarray) -> tuple[int, int, int, int]:
    """The framed canvas as (left, top, right, bottom) px of the delivery
    canvas: the frame's own `canvas`, else the box out to its border's outer
    edge together with the visible figure and outline, `FRAME_MARGIN_PCT`
    around, rounded outward and kept within `FRAME_COORD_RANGE`."""
    height, width = figure_alpha.shape
    if frame.get("canvas") is not None:
        return tuple(int(round(v)) for v in _share_px(frame["canvas"], width, height))
    longest = max(height, width)
    half = _border_half(frame, longest)
    x0, y0, x1, y1 = _share_px(frame["box"], width, height)
    x0, y0, x1, y1 = x0 - half, y0 - half, x1 + half, y1 + half
    rows, cols = np.nonzero((figure_alpha > VISIBLE_ALPHA) | (outline_alpha > VISIBLE_ALPHA))
    if rows.size:
        x0, y0 = min(x0, cols.min()), min(y0, rows.min())
        x1, y1 = max(x1, cols.max() + 1), max(y1, rows.max() + 1)
    margin = longest * delivery_style.FRAME_MARGIN_PCT / 100
    low, high = delivery_style.FRAME_COORD_RANGE
    return (max(int(np.floor(x0 - margin)), int(np.ceil(low * width))),
            max(int(np.floor(y0 - margin)), int(np.ceil(low * height))),
            min(int(np.ceil(x1 + margin)), int(np.floor(high * width))),
            min(int(np.ceil(y1 + margin)), int(np.floor(high * height))))


def frame_canvas_share(figure_png: bytes, outline_png: bytes,
                       frame: Mapping) -> list[float]:
    """`frame_canvas` for the layer PNGs, as shares of the delivery canvas."""
    figure, outline = _decode(figure_png, "RGBA"), _decode(outline_png, "RGBA")
    height, width = figure.shape[:2]
    keep = frame_keep((height, width), frame)
    left, top, right, bottom = frame_canvas(
        frame, figure[..., 3] / 255.0 * keep, outline[..., 3] / 255.0 * keep)
    return [left / width, top / height, right / width, bottom / height]


def in_frame(figure: np.ndarray, outline: np.ndarray, frame: Mapping,
             backdrop: str | None, shadow_from: str | None = None, outlines=None,
             scene: str | None = None, light_from: str | None = None) -> Staged:
    """`figure` and `outline` (straight HxWx4 0..255) in the `frame`: on
    the framed canvas painted `paper`, the box filled with the `backdrop`
    pattern rendered at the box's size, its line straddling the box's sides
    with the rim along both its edges, and the figure and outline kept by
    `frame_keep` over all of it; the light `scene` and drop shadow as
    `on_pattern` sets them."""
    height, width = figure.shape[:2]
    keep = frame_keep((height, width), frame)
    figure_alpha = figure[..., 3] / 255.0 * keep
    outline_alpha = outline[..., 3] / 255.0 * keep
    left, top, right, bottom = frame_canvas(frame, figure_alpha, outline_alpha)
    if right <= left or bottom <= top:
        raise ValueError("the frame's canvas is empty")
    size = (bottom - top, right - left)
    layers = []
    for colour, alpha in ((figure[..., :3], figure_alpha), (outline[..., :3], outline_alpha)):
        placed = np.zeros((*size, 4), float)
        _paste(placed, np.dstack([colour, alpha]), -left, -top)
        layers.append((placed[..., :3], placed[..., 3]))
    x0, y0, x1, y1 = np.subtract(_share_px(frame["box"], width, height),
                                 (left, top, left, top))
    backdrop_rgb = backdrops.render(frame["paper"], *size)
    panel_x0, panel_y0 = int(np.ceil(x0 - 0.5)), int(np.ceil(y0 - 0.5))
    panel_x1, panel_y1 = int(np.ceil(x1 - 0.5)), int(np.ceil(y1 - 0.5))
    if panel_x1 > panel_x0 and panel_y1 > panel_y0:
        _paste(backdrop_rgb, backdrops.render(backdrop, panel_y1 - panel_y0,
                                              panel_x1 - panel_x0), panel_x0, panel_y0)
    longest = max(height, width)
    for band, half in ((frame.get("rim"), _border_half(frame, longest)),
                       (frame.get("line"), _width_px(frame.get("line"), longest) / 2)):
        if band and half > 0:
            ring = (_rect(size, x0 - half, y0 - half, x1 + half, y1 + half)
                    & ~_rect(size, x0 + half, y0 + half, x1 - half, y1 - half))
            backdrop_rgb[ring] = parse_color(band["color"])
    (figure_colour, figure_alpha), (outline_colour, outline_alpha) = layers
    return _on_backdrop(figure_colour, figure_alpha, outline_colour, outline_alpha,
                        backdrop_rgb, shadow_from, outlines, scene, light_from)


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
               light_from: str | None = None,
               frame: Mapping | None = None) -> tuple[bytes, bytes, bytes, bytes]:
    """`on_photo` when `photo_png` is given, `in_frame` when a `frame` is,
    `on_pattern` otherwise, over PNG bytes; returns `layer_pngs`."""
    figure, outline = _decode(figure_png, "RGBA"), _decode(outline_png, "RGBA")
    if photo_png is not None:
        staged = on_photo(_decode(photo_png, "RGB"), figure, outline,
                          place=place, height=height)
    elif frame is not None:
        staged = in_frame(figure, outline, frame, backdrop, shadow_from, outlines, scene,
                          light_from)
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
