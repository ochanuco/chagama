"""Lens distortion over a delivered picture: a fisheye that swells the frame
around a centre and squeezes its rim, or its inverse that shrinks the centre
and stretches the rim. The picture is taken as a rectilinear (or, inverted, an
equidistant fisheye) projection and remapped to the other one, so the
distortion follows a real lens instead of an arbitrary polynomial.

The radius runs from the centre to the farthest corner. The fisheye keeps that
corner in place and pulls everything else inward, so the frame stays full; the
inverse would reach past the nearer edges, so it zooms in just enough to keep
every sample inside the picture."""

from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image

# The half angle of view, at the farthest corner, of strength ±1.
MAX_HALF_ANGLE = math.radians(75)
# Each output pixel averages SUPERSAMPLE² samples, so a squeezed rim does not
# alias.
SUPERSAMPLE = 3
STRIP_ROWS = 128


def _half_angle(strength: float) -> float:
    return abs(strength) * MAX_HALF_ANGLE


def source_radius(radius: np.ndarray, strength: float) -> np.ndarray:
    """The source radius an output radius samples, both as a share of the
    reach: tan for the fisheye (strength > 0), atan for its inverse."""
    theta = _half_angle(strength)
    if theta < 1e-6:
        return radius
    if strength > 0:
        return np.tan(radius * theta) / math.tan(theta)
    return np.arctan(radius * math.tan(theta)) / theta


def _gain(radius: np.ndarray, strength: float) -> np.ndarray:
    """source_radius / radius, with its limit at the centre."""
    safe = np.maximum(radius, 1e-6)
    return source_radius(safe, strength) / safe


def _zoom(width: int, height: int, cx: float, cy: float, reach: float,
          strength: float) -> float:
    """How far the inverse fisheye zooms in so that every boundary pixel, and
    so every pixel on its ray, samples inside the picture."""
    if strength >= 0:
        return 1.0
    xs, ys = np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    edge_x = np.concatenate([xs, xs, np.zeros(height), np.full(height, width - 1)])
    edge_y = np.concatenate([np.zeros(width), np.full(width, height - 1), ys, ys])
    radius = np.hypot(edge_x - cx, edge_y - cy) / reach
    return max(1.0, float(_gain(radius, strength).max()))


def distort(image: np.ndarray, center: tuple[float, float],
            strength: float) -> np.ndarray:
    """`image` (H, W, C float32, premultiplied when it has alpha) seen
    through the lens centred at `center` (0..1 of the width and height)."""
    height, width = image.shape[:2]
    if abs(strength) < 1e-6:
        return image.copy()
    cx, cy = center[0] * (width - 1), center[1] * (height - 1)
    reach = max(math.hypot(x - cx, y - cy)
                for x in (0, width - 1) for y in (0, height - 1))
    zoom = _zoom(width, height, cx, cy, reach, strength)
    s = SUPERSAMPLE
    xs = (np.arange(width * s, dtype=np.float64) + 0.5) / s - 0.5
    out = np.empty_like(image)
    for top in range(0, height, STRIP_ROWS):
        rows = min(STRIP_ROWS, height - top)
        ys = top + (np.arange(rows * s, dtype=np.float64) + 0.5) / s - 0.5
        dx, dy = xs[None, :] - cx, ys[:, None] - cy
        gain = _gain(np.hypot(dx, dy) / reach, strength) / zoom
        map_x = (cx + dx * gain).astype(np.float32)
        map_y = (cy + dy * gain).astype(np.float32)
        sampled = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        reduced = cv2.resize(sampled, (width, rows), interpolation=cv2.INTER_AREA)
        out[top:top + rows] = reduced.reshape(rows, width, -1)
    return out


def lens_png(png: bytes, center: tuple[float, float],
             strength: float) -> tuple[bytes, str]:
    """The distorted picture and its mode: RGBA when the source has any
    transparency, RGB otherwise."""
    rgba = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"),
                      dtype=np.float32) / 255.0
    alpha = rgba[..., 3:]
    transparent = bool((alpha < 1.0).any())
    if transparent:
        layered = np.concatenate([rgba[..., :3] * alpha, alpha], axis=-1)
    else:
        layered = np.ascontiguousarray(rgba[..., :3])
    warped = distort(layered, center, strength)
    if transparent:
        alpha = np.clip(warped[..., 3:], 0.0, 1.0)
        colour = np.where(alpha > 1e-4, warped[..., :3] / np.maximum(alpha, 1e-4), 0.0)
        warped = np.concatenate([colour, alpha], axis=-1)
    mode = "RGBA" if transparent else "RGB"
    array = np.clip(np.round(warped * 255.0), 0, 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(array, mode).save(output, "PNG")
    return output.getvalue(), mode
