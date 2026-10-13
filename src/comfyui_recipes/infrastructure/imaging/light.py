"""Scene lighting underpaint: shade, tint and rim painted onto the figure from
one light direction, read off a depth map and the silhouette. The figure is
re-sampled over it, so the result only has to say where light and shadow go."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from ...domain.yukari import delivery_style
from . import palette

GAMMA = 2.2
# Lit-side tint and the far-side tint multiplier push the colour past 1.0 in
# linear light so a tint reads at full strength on mid tones.
TINT_GAIN = 1.6
DEPTH_RELIEF = 0.15
DEPTH_BLUR = 0.006
INFLATE_BLUR = 0.004
RIM_BAND = 0.006
RIM_FACING_POWER = 1.5
FAR_POWER = 1.3
FAR_PERCENTILES = (5, 95)


def normals(depth: np.ndarray, matte: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    """Unit surface normals (y down) from the silhouette's inflated height
    plus the depth relief, and the distance of every pixel inside the matte
    from its edge."""
    height, width = matte.shape
    longest = max(height, width)
    inside = matte > 0.5
    dist = ndimage.distance_transform_edt(inside)
    reach = longest * delivery_style.LIGHT_REACH
    ramp = np.clip(dist / reach, 0, 1)
    inflate = cv2.GaussianBlur(np.sqrt(ramp * (2 - ramp)), (0, 0),
                               longest * INFLATE_BLUR) * reach
    relief = cv2.GaussianBlur(depth / 255.0, (0, 0), longest * DEPTH_BLUR) \
        * longest * DEPTH_RELIEF
    gy, gx = np.gradient(inflate + relief)
    n = np.dstack([-gx, -gy, np.ones_like(gx)])
    return n / np.linalg.norm(n, axis=2, keepdims=True), dist


def _light_figure(rgb: np.ndarray, matte: np.ndarray, depth: np.ndarray,
                  direction: str, scene: dict, form: bool = True) -> np.ndarray:
    """`rgb` lit from `direction` in `scene`. `form` shades by the surface
    normals; without a depth map they come from the silhouette alone, which
    darkens whatever happens to face away from the nearest edge (a jaw by a
    gap under the hair), so `form=False` keeps only the far-side gradient,
    the tints and the rim."""
    elevation = delivery_style.LIGHT_ELEVATION
    lx, ly = delivery_style.STROKE_LIGHTS[direction]
    flat = np.sqrt(1 - elevation ** 2)
    light = np.array([lx * flat, ly * flat, elevation])
    n, dist = normals(depth, matte)
    ndl = np.clip((n * light).sum(axis=2), -1, 1)
    inside = matte > 0.5
    lin = (rgb / 255.0) ** GAMMA
    color = (np.array(scene["color"]) / 255.0) ** GAMMA
    shadow = np.array(scene["shadow"]) ** GAMMA
    skin = palette.skin_mask(np.clip(rgb, 0, 255).astype(np.uint8))
    keep = (1 - scene["skin_keep"] * skin.astype(float))[..., None]
    shade_k = delivery_style.LIGHT_SHADE * keep
    grad_k = delivery_style.LIGHT_GRADIENT * keep
    tint_k = scene["tint"] * keep

    out = lin
    if form:
        away = np.clip((elevation - ndl) / (1 + elevation), 0, 1)
        out = lin * (1 - shade_k * away[..., None] * (1 - shadow))

    height, width = inside.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    proj = -(xx * lx + yy * ly)
    low, high = np.percentile(proj[inside], FAR_PERCENTILES)
    far = (np.clip((proj - low) / (high - low), 0, 1) ** FAR_POWER)[..., None]
    out = out * (1 - grad_k * far * (1 - shadow))
    near = 1 - far
    out = out * (1 - tint_k * near) + out * color * TINT_GAIN * tint_k * near

    if form:
        lit = np.clip((ndl - elevation * 0.5) / (1 - elevation * 0.5), 0, 1)[..., None]
        out = out * (1 - tint_k * lit) + out * color * TINT_GAIN * tint_k * lit

    band = max(height, width) * RIM_BAND
    edge = np.exp(-(dist / band) ** 2) * inside
    nxy = np.sqrt(n[..., 0] ** 2 + n[..., 1] ** 2) + 1e-6
    facing = np.clip((n[..., 0] * lx + n[..., 1] * ly) / nxy, 0, 1) ** RIM_FACING_POWER
    rim = delivery_style.LIGHT_RIM * edge * facing
    out = 1 - (1 - out) * (1 - color * rim[..., None])
    out = np.where(inside[..., None], out, lin)
    return np.clip(out, 0, 1) ** (1 / GAMMA) * 255


def underpaint_png(image_png: bytes, depth_png: bytes, matte_png: bytes,
                   direction: str, scene: str) -> bytes:
    rgb = np.array(Image.open(io.BytesIO(image_png)).convert("RGB")).astype(float)
    height, width = rgb.shape[:2]
    depth = np.array(Image.open(io.BytesIO(depth_png)).convert("L"),
                     dtype=np.float32)
    matte = np.array(Image.open(io.BytesIO(matte_png)).convert("L"),
                     dtype=np.float32) / 255.0
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    matte = cv2.resize(matte, (width, height), interpolation=cv2.INTER_LINEAR)
    inside = matte > 0.5
    lit = rgb
    if inside.any():
        lit = np.where(inside[..., None], _light_figure(
            rgb, matte, depth.astype(float), direction,
            delivery_style.LIGHT_SCENES[scene]), rgb)
    output = io.BytesIO()
    Image.fromarray(np.clip(np.rint(lit), 0, 255).astype(np.uint8), "RGB").save(
        output, "PNG")
    return output.getvalue()
