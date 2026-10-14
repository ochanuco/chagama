"""Depth-of-field blur over a delivery's layers: a lens-disc circle of
confusion that grows with distance from a focus point, applied in linear
light. The figure keeps its own depth, the outline bands and shadow take the
depth of the nearest figure pixel, the backdrop sits on the far plane; each
layer is blurred on its own and the layers are composited back to front, so
an out-of-focus figure spreads a soft alpha over what stands behind it."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

LEVELS = 8
K_FRACTION = 0.024
DEPTH_PERCENTILES = (2, 98)
FOCUS_WINDOW_FRACTION = 0.015
ALPHA_EPSILON = 1e-4
MATTE_THRESHOLD = 0.5
# A blur radius up to this share of the long side still reads as sharp.
SHARP_FRACTION = 0.001


def sharp_reach_per_f() -> float:
    """How far from the focus depth, in normalised depth per unit of
    f-number, the blur stays within SHARP_FRACTION."""
    return SHARP_FRACTION / K_FRACTION


def _normalised_depth(depth: np.ndarray, inside: np.ndarray) -> np.ndarray | None:
    sample = depth[inside] if inside.any() else depth.ravel()
    low, high = np.percentile(sample, DEPTH_PERCENTILES)
    if high - low < 1e-6:
        return None
    return np.clip((depth - low) / (high - low), 0.0, 1.0)


def _focus_depth(depth: np.ndarray, focus_x: float, focus_y: float) -> float:
    height, width = depth.shape
    radius = max(1, round(FOCUS_WINDOW_FRACTION * max(width, height)))
    cx = min(width - 1, int(focus_x * width))
    cy = min(height - 1, int(focus_y * height))
    window = depth[max(0, cy - radius):cy + radius + 1,
                   max(0, cx - radius):cx + radius + 1]
    return float(np.median(window))


def _disc(radius: float) -> np.ndarray:
    reach = int(np.ceil(radius))
    y, x = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    kernel = np.clip(radius + 0.5 - np.hypot(x, y), 0.0, 1.0).astype(np.float32)
    return kernel / kernel.sum()


def _to_linear(rgb: np.ndarray) -> np.ndarray:
    c = rgb / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(linear: np.ndarray) -> np.ndarray:
    c = np.clip(linear, 0.0, 1.0)
    return 255.0 * np.where(c <= 0.0031308, c * 12.92,
                            1.055 * c ** (1 / 2.4) - 0.055)


def _spread_outward(radius: np.ndarray, inside: np.ndarray) -> np.ndarray:
    _, (rows, cols) = ndimage.distance_transform_edt(~inside, return_indices=True)
    return radius[rows, cols]


def _gather(source: np.ndarray, alpha: np.ndarray, radius: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    """`source` (linear, straight) and `alpha` blurred by a per-pixel disc of
    `radius`, as LEVELS fixed discs blended by each pixel's own radius.
    Returns the straight colour and its cover."""
    r_max = float(radius.max())
    if r_max < 0.5:
        return source, alpha
    position = np.minimum(radius / r_max, 1.0) * (LEVELS - 1)
    colour = np.zeros_like(source)
    cover = np.zeros_like(alpha)
    for level in range(LEVELS):
        weight = np.clip(1.0 - np.abs(position - level), 0.0, 1.0)
        if not weight.any():
            continue
        level_radius = r_max * level / (LEVELS - 1)
        if level_radius < 0.5:
            layer, layer_cover = source, alpha
        else:
            kernel = _disc(level_radius)
            contributes = alpha * np.clip(
                (radius - 0.5 * level_radius) / (0.5 * level_radius), 0.0, 1.0)
            premultiplied = np.concatenate(
                [source * contributes[..., None], contributes[..., None]],
                axis=-1).astype(np.float32)
            gathered = cv2.filter2D(premultiplied, -1, kernel,
                                    borderType=cv2.BORDER_REPLICATE)
            weight_sum = gathered[..., 3:4]
            layer = np.where(weight_sum > ALPHA_EPSILON,
                             gathered[..., :3] / np.maximum(weight_sum, ALPHA_EPSILON),
                             source)
            layer_cover = cv2.filter2D(alpha, -1, kernel,
                                       borderType=cv2.BORDER_REPLICATE)
        colour += layer * weight[..., None]
        cover += layer_cover * weight
    return colour, np.clip(cover, 0.0, 1.0)


def blur_layers(figure: np.ndarray, figure_alpha: np.ndarray,
                outline: np.ndarray, outline_alpha: np.ndarray,
                backdrop: np.ndarray | None, depth: np.ndarray,
                focus_x: float, focus_y: float, f_number: float,
                scope: dict) -> tuple[np.ndarray, np.ndarray]:
    """Colours are HxWx3 straight 0..255, alphas HxW 0..1, `depth` HxW with
    higher = nearer. `scope` names the layers (`figure`, `outline`,
    `backdrop`) to blur; the others are composited sharp, as delivered.
    Returns the composite's straight colour and alpha, opaque over a
    backdrop."""
    height, width = figure_alpha.shape
    inside = figure_alpha > MATTE_THRESHOLD
    normalised = _normalised_depth(depth, inside) if inside.any() else None
    if normalised is None:
        radius = np.zeros((height, width), np.float32)
        far_radius = 0.0
    else:
        near = _spread_outward(normalised, inside)
        picture_depth = np.where(
            inside, normalised, np.where(outline_alpha > MATTE_THRESHOLD, near, 0.0))
        d_focus = _focus_depth(picture_depth, focus_x, focus_y)
        scale = K_FRACTION * max(width, height) / f_number
        radius = (scale * np.abs(near - d_focus)).astype(np.float32)
        far_radius = scale * d_focus

    def blurred(colour: np.ndarray, alpha: np.ndarray, key: str):
        if not scope.get(key):
            return colour, alpha
        gathered, cover = _gather(_to_linear(colour).astype(np.float32),
                                  alpha.astype(np.float32), radius)
        return _to_srgb(gathered), cover

    layers = []
    if backdrop is not None:
        colour = backdrop.astype(np.float32)
        if scope.get("backdrop") and far_radius >= 0.5:
            colour = _to_srgb(cv2.filter2D(
                _to_linear(colour).astype(np.float32), -1, _disc(far_radius),
                borderType=cv2.BORDER_REPLICATE))
        layers.append((colour, np.ones((height, width), np.float32)))
    layers.append(blurred(outline, outline_alpha, "outline"))
    layers.append(blurred(figure, figure_alpha, "figure"))

    premultiplied = np.zeros((height, width, 3), np.float64)
    cover = np.zeros((height, width), np.float64)
    for colour, alpha in layers:
        premultiplied = colour * alpha[..., None] + premultiplied * (1 - alpha[..., None])
        cover = alpha + cover * (1 - alpha)
    straight = premultiplied / np.maximum(cover, ALPHA_EPSILON)[..., None]
    straight[cover <= 0] = 0.0
    return straight, cover


def _decode(data: bytes, mode: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert(mode), dtype=np.float32)


def blur_layers_png(figure_png: bytes, outline_png: bytes,
                    backdrop_png: bytes | None, depth_png: bytes,
                    focus_x: float, focus_y: float, f_number: float,
                    scope: dict) -> bytes:
    """`blur_layers` over a delivery's layer PNGs; RGB over a backdrop, RGBA
    without one."""
    figure = _decode(figure_png, "RGBA")
    outline = _decode(outline_png, "RGBA")
    height, width = figure.shape[:2]
    backdrop = _decode(backdrop_png, "RGB") if backdrop_png is not None else None
    depth = cv2.resize(_decode(depth_png, "L"), (width, height),
                       interpolation=cv2.INTER_LINEAR)
    colour, alpha = blur_layers(
        figure[..., :3], figure[..., 3] / 255.0, outline[..., :3],
        outline[..., 3] / 255.0, backdrop, depth, focus_x, focus_y, f_number,
        scope)
    rgb = np.clip(np.rint(colour), 0, 255).astype(np.uint8)
    output = io.BytesIO()
    if backdrop is not None:
        Image.fromarray(rgb, "RGB").save(output, "PNG")
    else:
        rgba = np.dstack([rgb, np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(output, "PNG")
    return output.getvalue()
