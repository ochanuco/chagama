"""Render a repair region mask (circles + rects) to a PNG ComfyUI can load."""

from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...domain.repair.regions import Circle, Rect


def render_mask_png(width: int, height: int, circles: Sequence[Circle],
                    rects: Sequence[Rect]) -> bytes:
    """White shapes on black, RGB -- `ImageToMask` reads the red channel."""
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    union = np.zeros((height, width), dtype=bool)
    for circle in circles:
        union |= (xx - circle.cx) ** 2 + (yy - circle.cy) ** 2 <= circle.r ** 2
    for rect in rects:
        union |= ((xx >= rect.x0) & (xx <= rect.x1)
                  & (yy >= rect.y0) & (yy <= rect.y1))
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[union] = 255
    output = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(output, "PNG")
    return output.getvalue()


def render_soft_mask_png(width: int, height: int, rects: Sequence[Rect],
                         inside: float, feather: float) -> bytes:
    """Grey-on-white PNG for `SetLatentNoiseMask`: 1.0 (white) redraws, so the
    canvas starts white and each rect is painted at `inside` then blurred by
    `feather` pixels, feathering the redraw back in at the rect edges.
    """
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)
    level = round(max(0.0, min(1.0, inside)) * 255)
    for rect in rects:
        draw.rectangle([rect.x0, rect.y0, rect.x1, rect.y1], fill=level)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    output = io.BytesIO()
    mask.convert("RGB").save(output, "PNG")
    return output.getvalue()


def mask_bbox_fraction(width: int, height: int, circles: Sequence[Circle],
                       rects: Sequence[Rect]) -> tuple[float, float, float, float]:
    """The union's bounding box, clipped to the canvas, as 0..1 fractions."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    for circle in circles:
        xs0.append(circle.cx - circle.r)
        ys0.append(circle.cy - circle.r)
        xs1.append(circle.cx + circle.r)
        ys1.append(circle.cy + circle.r)
    for rect in rects:
        xs0.append(rect.x0)
        ys0.append(rect.y0)
        xs1.append(rect.x1)
        ys1.append(rect.y1)
    if not xs0:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = max(0.0, min(xs0)) / width
    y0 = max(0.0, min(ys0)) / height
    x1 = min(float(width), max(xs1)) / width
    y1 = min(float(height), max(ys1)) / height
    return (x0, y0, x1, y1)
