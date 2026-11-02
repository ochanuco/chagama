"""A staged delivery's outline as a clear acrylic plate: the guide an image
edit model re-renders as real acrylic.

The plate is the delivery's own cut outline (figure and outline bands
together), drawn clear instead of white and purple: the backdrop shows
through it, its cut edge is swept away from the light, and its rim catches
the light. The figure is printed on it as delivered. This only has to tell
the model where the acrylic is and how thick; the model makes it look real,
and the figure is laid back over its result through `print_mask`."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from ...domain.yukari.delivery_style import STROKE_LIGHTS
from . import acrylic_stand as stand

# Thickness of the plate's cut edge, as a share of the figure's height.
THICKNESS = 0.014


def plate(backdrop: np.ndarray, figure: np.ndarray, outline_alpha: np.ndarray,
          direction: str) -> tuple[np.ndarray, np.ndarray]:
    """`backdrop` HxWx3 and `figure` HxWx4 straight 0..255 with the outline
    bands' 0..1 alpha. Returns the guide picture (HxWx3 0..255) and the
    print mask (HxW 0..1)."""
    light = np.asarray(STROKE_LIGHTS[direction], np.float32)
    figure_alpha = figure[..., 3] / 255.0
    cut = (np.maximum(figure_alpha, outline_alpha) > 0.5).astype(np.uint8)
    rows = np.nonzero(cut.any(1))[0]
    if rows.size == 0:
        raise ValueError("the delivered picture has no outline to make acrylic")
    unit = int(rows.max() - rows.min())
    thickness = max(2.0, THICKNESS * unit)
    line = max(1.0, 0.0025 * unit)
    soft = cv2.GaussianBlur(cut.astype(np.float32), (0, 0), 1.6)
    plate_alpha = np.clip((soft - 0.5) * 1.6 + 0.5, 0.0, 1.0)
    nx, ny = stand._normals(cut, max(2.0, thickness * 0.8))
    facing = np.clip(nx * light[0] + ny * light[1], -1.0, 1.0)

    image = (backdrop / 255.0).astype(np.float32)
    away = -light
    steps = max(2, int(np.ceil(thickness)))
    swept = np.zeros_like(plate_alpha)
    for k in range(1, steps + 1):
        swept = np.maximum(swept, stand._shift(plate_alpha, *(away * thickness * k / steps)))
    side = np.clip(swept - plate_alpha, 0.0, 1.0)
    clear = {"bevel": 0.015 * unit, "refraction": 0.015 * unit}
    through = stand._clear(image, side, nx, ny, light, absorb=0.25, **clear)
    image = image * (1 - side[..., None]) + through * 0.85 * side[..., None]
    image = stand._clear(image, plate_alpha, nx, ny, light, absorb=0.05, **clear)
    image = image * (1 - figure_alpha[..., None]) + figure[..., :3] / 255.0 * figure_alpha[..., None]
    image = stand._screen(image, stand.contour(plate_alpha, line)
                          * (0.2 + 0.6 * np.clip(facing, 0, 1)))

    inside = cv2.erode(figure_alpha.astype(np.float32), stand._disc(1))
    mask = np.clip(cv2.GaussianBlur(inside, (0, 0), 1.0), 0.0, 1.0)
    return np.clip(image, 0.0, 1.0) * 255.0, mask


def plate_pngs(backdrop_png: bytes, figure_png: bytes, outline_png: bytes,
               direction: str) -> tuple[bytes, bytes]:
    """`plate` over PNG bytes: the guide (RGB) and the print mask (L)."""
    def decode(data: bytes, mode: str) -> np.ndarray:
        return np.asarray(Image.open(io.BytesIO(data)).convert(mode), dtype=np.float32)

    def png(array: np.ndarray, mode: str) -> bytes:
        output = io.BytesIO()
        Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode).save(
            output, "PNG")
        return output.getvalue()

    guide, mask = plate(decode(backdrop_png, "RGB"), decode(figure_png, "RGBA"),
                        decode(outline_png, "RGBA")[..., 3] / 255.0, direction)
    return png(guide, "RGB"), png(mask * 255.0, "L")
