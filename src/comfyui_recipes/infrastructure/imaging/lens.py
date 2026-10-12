"""Lens distortion over a delivered picture, in one of two shapes.

`fisheye` bends the whole frame: a fisheye that swells the frame around a
centre and squeezes its rim, or its inverse that shrinks the centre and
stretches the rim. The picture is taken as a rectilinear (or, inverted, an
equidistant fisheye) projection and remapped to the other one, so the
distortion follows a real lens instead of an arbitrary polynomial. The radius
runs from the centre to the farthest corner. The fisheye keeps that corner in
place and pulls everything else inward, so the frame stays full; the inverse
would reach past the nearer edges, so it zooms in just enough to keep every
sample inside the picture.

`sphere` bends only a disc around the centre, like a spherize filter: the
disc is wrapped over a hemisphere seen head-on, so it bulges like a ball
(strength > 0) or is pinched in (strength < 0), and everything outside the
disc stays where it was."""

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
SHAPES = ("fisheye", "sphere")


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


def sphere_radius(radius: np.ndarray, strength: float) -> np.ndarray:
    """The source radius an output radius samples inside the sphere's disc,
    both as a share of the disc's radius: the hemisphere's arc (asin) for a
    bulge, its inverse (sin) for a pinch, blended with the identity by
    |strength|. Outside the disc nothing moves."""
    inside = np.clip(radius, 0.0, 1.0)
    curve = (np.arcsin(inside) * (2 / math.pi) if strength > 0
             else np.sin(inside * (math.pi / 2)))
    bent = (1 - abs(strength)) * inside + abs(strength) * curve
    return np.where(radius < 1.0, bent, radius)


def _gain(radius: np.ndarray, strength: float, shape: str = "fisheye") -> np.ndarray:
    """source radius / radius, with its limit at the centre."""
    safe = np.maximum(radius, 1e-6)
    bend = sphere_radius if shape == "sphere" else source_radius
    return bend(safe, strength) / safe


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


def distort(image: np.ndarray, center: tuple[float, float], strength: float,
            shape: str = "fisheye", radius: float | None = None) -> np.ndarray:
    """`image` (H, W, C float32, premultiplied when it has alpha) seen
    through the lens centred at `center` (0..1 of the width and height).
    A `sphere` bends the disc of `radius` (a share of the shorter side)."""
    height, width = image.shape[:2]
    if abs(strength) < 1e-6:
        return image.copy()
    cx, cy = center[0] * (width - 1), center[1] * (height - 1)
    if shape == "sphere":
        if radius is None:
            raise ValueError("a sphere lens needs a radius")
        reach, zoom = radius * min(width, height), 1.0
    else:
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
        gain = _gain(np.hypot(dx, dy) / reach, strength, shape) / zoom
        map_x = (cx + dx * gain).astype(np.float32)
        map_y = (cy + dy * gain).astype(np.float32)
        sampled = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        reduced = cv2.resize(sampled, (width, rows), interpolation=cv2.INTER_AREA)
        out[top:top + rows] = reduced.reshape(rows, width, -1)
    if shape == "sphere":
        # The supersampled identity outside the disc still averages neighbours;
        # pixels well clear of it keep the source exactly.
        ys, xs = np.mgrid[:height, :width]
        outside = np.hypot(xs - cx, ys - cy) > reach + 1
        out[outside] = image[outside]
    return out


def lens_png(png: bytes, center: tuple[float, float], strength: float,
             shape: str = "fisheye", radius: float | None = None) -> tuple[bytes, str]:
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
    warped = distort(layered, center, strength, shape, radius)
    if transparent:
        alpha = np.clip(warped[..., 3:], 0.0, 1.0)
        colour = np.where(alpha > 1e-4, warped[..., :3] / np.maximum(alpha, 1e-4), 0.0)
        warped = np.concatenate([colour, alpha], axis=-1)
    mode = "RGBA" if transparent else "RGB"
    array = np.clip(np.round(warped * 255.0), 0, 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(array, mode).save(output, "PNG")
    return output.getvalue(), mode
