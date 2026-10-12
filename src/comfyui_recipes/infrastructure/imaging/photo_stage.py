"""Stand a delivered figure inside a photo, like an acrylic stand shot on
location, and blur the scene by depth.

The figure and its outline bands form one flat card standing at the photo's
depth where its base is, pulled toward the camera by `nearer`. Upright, the
whole card sits at that one depth; `tilt` leans it, its top that much farther
than its base, so a focus on the face lets the body fall out of focus below.
The photo keeps its own per-pixel depth. Both are blurred
by the same lens-disc circle of confusion as the dof, so whatever shares the
focus depth stays sharp (a signboard the card stands beside) and the rest
falls out of focus."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from .depth_blur import (
    ALPHA_EPSILON,
    DEPTH_PERCENTILES,
    K_FRACTION,
    _focus_depth,
    _gather,
    _to_linear,
    _to_srgb,
)

# Card pixels at or below this alpha are cut from its bounding box.
CARD_EDGE_ALPHA = 0.02
SUBJECT = "subject"


def normalised_depth(depth: np.ndarray) -> np.ndarray:
    """`depth` (higher = nearer) stretched to 0..1 between its percentiles."""
    low, high = np.percentile(depth, DEPTH_PERCENTILES)
    if high - low < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip((depth - low) / (high - low), 0.0, 1.0).astype(np.float32)


def card(figure: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """The figure over its outline bands (both HxWx4 straight 0..255),
    cropped to what is visible, as premultiplied 0..1 RGBA."""
    def premultiplied(layer: np.ndarray) -> np.ndarray:
        alpha = layer[..., 3:4] / 255.0
        return np.concatenate([layer[..., :3] / 255.0 * alpha, alpha], axis=-1)

    front, back = premultiplied(figure), premultiplied(outline)
    stacked = front + back * (1.0 - front[..., 3:4])
    rows, cols = np.nonzero(stacked[..., 3] > CARD_EDGE_ALPHA)
    if rows.size == 0:
        raise ValueError("the delivered picture has no visible figure to stand in the photo")
    return stacked[rows.min():rows.max() + 1, cols.min():cols.max() + 1].astype(np.float32)


def card_rows(size: tuple[int, int], place: tuple[float, float],
              height: float) -> tuple[int, int]:
    """The canvas row just below the placed card's bottom edge and its
    height in pixels."""
    photo_height = size[1]
    return round(place[1] * photo_height), max(1, round(height * photo_height))


def place_card(premultiplied: np.ndarray, size: tuple[int, int],
               place: tuple[float, float], height: float) -> np.ndarray:
    """The card scaled to `height` (a share of the photo height) with the
    middle of its bottom edge on `place`, on a transparent HxWx4 canvas."""
    width, photo_height = size
    bottom, card_height = card_rows(size, place, height)
    scale = card_height / premultiplied.shape[0]
    card_width = max(1, round(premultiplied.shape[1] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = np.clip(cv2.resize(premultiplied, (card_width, card_height),
                                 interpolation=interpolation), 0.0, 1.0)
    resized[..., :3] = np.minimum(resized[..., :3], resized[..., 3:4])
    left = round(place[0] * width - card_width / 2)
    top = bottom - card_height
    canvas = np.zeros((photo_height, width, 4), np.float32)
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + card_width, width), min(top + card_height, photo_height)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = resized[y0 - top:y1 - top, x0 - left:x1 - left]
    return canvas


def stage(photo: np.ndarray, depth: np.ndarray, figure: np.ndarray,
          outline: np.ndarray, *, place: tuple[float, float], height: float,
          stand_at: tuple[float, float] | None, nearer: float,
          focus: tuple[float, float] | str, f_number: float,
          tilt: float = 0.0) -> np.ndarray:
    """`photo` HxWx3 0..255 with `depth` HxW (higher = nearer); `figure` and
    `outline` are a delivery's straight RGBA layers. `stand_at` names the
    photo point whose depth the card's base stands at (the base itself when
    None); `tilt` puts the card's top that much farther (negative: nearer),
    in normalised scene depth. `focus` is a point of the picture (on the
    card, the card's depth there) or SUBJECT for the card's middle.
    Returns the HxWx3 composite, 0..255."""
    height_px, width_px = photo.shape[:2]
    scene = normalised_depth(cv2.resize(depth.astype(np.float32), (width_px, height_px),
                                        interpolation=cv2.INTER_LINEAR))
    placed = place_card(card(figure, outline), (width_px, height_px), place, height)
    card_alpha = placed[..., 3]
    base = stand_at if stand_at is not None else (place[0], min(place[1], 1.0 - 1e-6))
    base_depth = _focus_depth(scene, *base) + nearer
    bottom, card_height = card_rows((width_px, height_px), place, height)
    rise = np.clip((bottom - np.arange(height_px, dtype=np.float32)) / card_height, 0.0, 1.0)
    row_depth = np.clip(base_depth - tilt * rise, 0.0, 1.0)
    if focus == SUBJECT:
        focus_depth = float(np.clip(base_depth - tilt * 0.5, 0.0, 1.0))
    else:
        fx = min(width_px - 1, int(focus[0] * width_px))
        fy = min(height_px - 1, int(focus[1] * height_px))
        focus_depth = (float(row_depth[fy]) if card_alpha[fy, fx] > 0.5
                       else _focus_depth(scene, focus[0], focus[1]))

    scale = K_FRACTION * max(width_px, height_px) / f_number
    photo_colour, _ = _gather(_to_linear(photo.astype(np.float32)).astype(np.float32),
                              np.ones((height_px, width_px), np.float32),
                              (scale * np.abs(scene - focus_depth)).astype(np.float32))
    straight = np.where(card_alpha[..., None] > ALPHA_EPSILON,
                        placed[..., :3] / np.maximum(card_alpha, ALPHA_EPSILON)[..., None],
                        0.0)
    card_radius = np.broadcast_to(
        (scale * np.abs(row_depth - focus_depth))[:, None], (height_px, width_px)
    ).astype(np.float32)
    card_colour, card_cover = _gather(_to_linear(straight * 255.0).astype(np.float32),
                                      card_alpha, card_radius)
    cover = np.clip(card_cover, 0.0, 1.0)[..., None]
    linear = card_colour * cover + photo_colour * (1.0 - cover)
    return _to_srgb(linear)


def stage_png(photo_png: bytes, depth_png: bytes, figure_png: bytes,
              outline_png: bytes, **options) -> bytes:
    """`stage` over PNG bytes; the result is an RGB PNG at the photo's size."""
    def decode(data: bytes, mode: str) -> np.ndarray:
        return np.asarray(Image.open(io.BytesIO(data)).convert(mode), dtype=np.float32)

    rgb = stage(decode(photo_png, "RGB"), decode(depth_png, "L"),
                decode(figure_png, "RGBA"), decode(outline_png, "RGBA"), **options)
    output = io.BytesIO()
    Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8), "RGB").save(output, "PNG")
    return output.getvalue()
