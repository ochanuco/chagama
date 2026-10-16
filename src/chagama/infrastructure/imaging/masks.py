"""Render a redraw region mask to a PNG ComfyUI can load."""

from __future__ import annotations

import io
from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFilter

from ...domain.regions import Rect


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
