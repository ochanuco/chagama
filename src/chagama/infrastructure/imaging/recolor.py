"""Repaint a render's fills from a measured palette, keeping its own linework.

`repin` only ever nudges the render's own saturation, so a washed-out black
stays washed out -- V is untouched by design. `recolor` instead detects the
render's own linework, labels the fills it encloses, classifies each fill as
a material, and asserts that material's colour outright: it fixes value as
well as hue and saturation, at the cost of trusting the classifier instead
of the render. Targets live in `domain.yukari.delivery_style`; this module
only applies them.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage

from .palette import figure_mask
from ...domain.yukari.delivery_style import (
    RECOLOR_ACCENT_ERODE, RECOLOR_ACCENT_S, RECOLOR_DARK_V,
    RECOLOR_HAIR_S, RECOLOR_KEEP_HS, RECOLOR_KEEP_V,
    RECOLOR_LEG_CY, RECOLOR_LEG_MIN_AREA, RECOLOR_LEG_STOPS, RECOLOR_LINE_MAX,
    RECOLOR_LINE_RELIEF, RECOLOR_LINE_WINDOW, RECOLOR_SKIN_HUE,
    RECOLOR_H_SPREAD, RECOLOR_S_CEILING, RECOLOR_TARGETS, RECOLOR_WHITE_S,
)


def lines(im: np.ndarray) -> np.ndarray:
    """The render's own linework: dark, and darker than what it bounds.

    Darkness is the brightest channel, not luminance. A magenta stroke is
    dark by luminance while its red channel is at full -- and some renders
    draw the creases between fingers, the mouth and the collarbone exactly
    that way, so a luminance test calls them linework and preserves them.
    """
    fig = figure_mask(im)
    dark = im.astype(float).max(axis=-1)
    relief = ndimage.maximum_filter(dark, RECOLOR_LINE_WINDOW) - dark
    return fig & (dark < RECOLOR_LINE_MAX) & (relief > RECOLOR_LINE_RELIEF)


def classify(h: float, s: float, v: float, cy: float, area: float) -> str | None:
    lo, hi = RECOLOR_SKIN_HUE
    warm = h < lo or h > hi
    if v < RECOLOR_DARK_V:
        return "tights" if cy > RECOLOR_LEG_CY else "hoodie"
    if cy > RECOLOR_LEG_CY and area >= RECOLOR_LEG_MIN_AREA and not warm:
        return "tights"
    if s > RECOLOR_ACCENT_S:
        return None
    if warm:
        return "skin"
    if s < RECOLOR_WHITE_S:
        return "white"
    if s < RECOLOR_HAIR_S:
        return "hair"
    return "dress"


def _around(values: np.ndarray, median: float, target: float,
            keep: float, wrap: bool = False,
            ceiling: float | None = None) -> np.ndarray:
    """The target, with the region's own deviation kept around it."""
    deviation = values - median
    if wrap:
        deviation = (deviation + 128.0) % 256.0 - 128.0
    if wrap:
        deviation = np.clip(deviation, -RECOLOR_H_SPREAD, RECOLOR_H_SPREAD)
    moved = target + deviation * keep
    if ceiling is not None:
        moved = np.minimum(moved, target + ceiling)
    return moved % 256.0 if wrap else moved


def _paint_tights(region: np.ndarray, out_h, out_s, out_v, V, top: int,
                  height: int) -> None:
    rows = np.where(region)[0]
    cy = (rows - top) / height
    stops = RECOLOR_LEG_STOPS
    xs = [stop[0] for stop in stops]
    target_h = np.interp(cy, xs, [stop[1][0] for stop in stops])
    target_s = np.interp(cy, xs, [stop[1][1] for stop in stops])
    target_v = np.interp(cy, xs, [stop[1][2] for stop in stops])
    hs = RECOLOR_KEEP_HS
    out_h[region] = _around(out_h[region], float(np.median(out_h[region])),
                            target_h, hs, wrap=True)
    out_s[region] = _around(out_s[region], float(np.median(out_s[region])),
                            target_s, hs, ceiling=RECOLOR_S_CEILING)
    out_v[region] = np.clip(
        target_v + (V[region] - np.median(V[region]))
        * RECOLOR_KEEP_V["tights"], 0, 255)


def _stray_stroke(region: np.ndarray) -> bool:
    """A saturated fill too thin to be an iris or a hair pin."""
    return not ndimage.binary_erosion(
        region, iterations=RECOLOR_ACCENT_ERODE).any()


def _paint_surrounding(region: np.ndarray, out_h, out_s, out_v) -> bool:
    """Give a stray stroke the colour of the material it sits in."""
    ring = ndimage.binary_dilation(region, iterations=3) & ~region
    if not ring.any():
        return False
    for channel in (out_h, out_s, out_v):
        channel[region] = np.median(channel[ring])
    return True


def recolor(im: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """RGB array in, RGB array out, plus a report of each material's share.

    Regions are the fills the render's own linework encloses -- figure_mask
    finds the figure, `lines` finds the linework, ndimage.label enumerates
    what is left. Each region is classified once from its median HSV and its
    centroid height, normalised to the FIGURE's own bounding box rather than
    the canvas, since where the figure sits on the canvas is not part of its
    shape. Painting keeps a share of the region's own V against the target,
    so the render's own shading still reads under the asserted colour.
    """
    hsv = np.array(Image.fromarray(im).convert("HSV")).astype(float)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    fig = figure_mask(im)
    line = lines(im)
    labels, count = ndimage.label(fig & ~line)

    rows = np.where(fig.any(axis=1))[0]
    top, bottom = int(rows.min()), int(rows.max())
    height = max(bottom - top, 1)

    out_h, out_s, out_v = H.copy(), S.copy(), V.copy()
    total = int(fig.sum())
    claimed: dict[str, int] = {}
    strays = []
    for index in range(1, count + 1):
        region = labels == index
        h = float(np.median(H[region]))
        s = float(np.median(S[region]))
        v = float(np.median(V[region]))
        cy = (float(np.mean(np.where(region)[0])) - top) / height
        material = classify(h, s, v, cy, int(region.sum()) / total)
        if material is None:
            if _stray_stroke(region):
                strays.append(region)
            continue
        if material == "tights":
            _paint_tights(region, out_h, out_s, out_v, V, top, height)
        else:
            t_h, t_s, t_v = RECOLOR_TARGETS[material]
            hs = RECOLOR_KEEP_HS
            out_h[region] = _around(H[region], h, t_h, hs, wrap=True)
            out_s[region] = _around(S[region], s, t_s, hs,
                                    ceiling=RECOLOR_S_CEILING)
            out_v[region] = np.clip(
                t_v + (V[region] - v) * RECOLOR_KEEP_V[material], 0, 255)
        claimed[material] = claimed.get(material, 0) + int(region.sum())

    cleared = sum(_paint_surrounding(region, out_h, out_s, out_v)
                  for region in strays)

    out = np.stack([out_h, out_s.clip(0, 255), out_v.clip(0, 255)], axis=-1)
    rgb = np.array(Image.fromarray(out.clip(0, 255).astype(np.uint8), "HSV")
                   .convert("RGB"))
    rgb[line] = im[line]

    share = ", ".join(f"{material} {count_ / total * 100:.1f}%"
                      for material, count_ in
                      sorted(claimed.items(), key=lambda kv: -kv[1]))
    report = [share] if share else []
    if cleared:
        report.append(f"{cleared} stray strokes taken into their surround")
    return rgb, report


def recolor_png(data: bytes) -> tuple[bytes, list[str]]:
    im = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    rgb, report = recolor(im)
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    return output.getvalue(), report
