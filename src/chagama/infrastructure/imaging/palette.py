"""Measure and pin a render's colour against Yukari's delivery palette.

`repin` scales each material window's saturation toward its own per-band
target (V untouched, scaling only ever down) so materials that drift by
different amounts are corrected separately instead of by one global factor.
`measure`/`verdict` score a render against the acceptance bands. Values live
in `domain.yukari.delivery_style`; this module only applies them.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage

from .delivery import background_mask, enclosed_mask
from ...domain.yukari.delivery_style import (
    ACCENT_KEEP, ACCENT_RAMP, ACCENT_VALUE_RAMP, BACKDROP_SPREAD_MAX, BG_SAT_MAX,
    FIGURE_LIGHT_SAT_TARGET, FIGURE_LIGHT_V, FIGURE_MIDTONE_V,
    FIGURE_SAT_MEAN_MAX, FIGURE_SAT_P90_MAX, PALETTE_WINDOWS, REPIN_CHROMA_WINDOWS,
    REPIN_DARK, REPIN_DARK_EXEMPT, REPIN_LIGHT, REPIN_MID, REPIN_SAT_ONLY_V_MAX,
    REPIN_SAT_ONLY_WINDOWS, REPIN_SKIN_CAP, REPIN_SKIN_CAP_V,
    REPIN_SKIN_WINDOW,
    SAT_BAND, SKIN_PIN_MIN_AREA, SKIN_SOURCE_S_MIN, SKIN_SOURCE_V_MIN,
)

H_TARGET_BLEND = 0.7   # hue is eased toward the target, not snapped


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def window_w(H, lo, hi, feather=10.0):
    return smoothstep((H - (lo - feather)) / feather) * \
        smoothstep(((hi + feather) - H) / feather)


def band_w(V, feather=30.0):
    w = smoothstep((V - (FIGURE_LIGHT_V - feather)) / (2 * feather))
    return w, 1.0 - w


def palette_window(name: str) -> dict:
    return next(w for w in PALETTE_WINDOWS if w["name"] == name)


def figure_mask(im: np.ndarray) -> np.ndarray:
    mask = background_mask(im.astype(int), 18)
    mask |= enclosed_mask(im.astype(int), mask, 4)
    return ~mask


def repin(im: np.ndarray, protect: np.ndarray | None = None,
          figure: np.ndarray | None = None,
          strength: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """RGB array in, RGB array out, plus a report of what moved.

    Saturation is compressed per V band toward fixed knees rather than
    scaled by a measured factor, so the curve is the same for every picture
    and a pale render passes through untouched. `protect` marks pixels kept
    verbatim; the correction fades to zero over a feathered edge. `figure`
    is the boolean pixel range the curve applies to, estimated from the
    picture when None; `strength` scales the correction (0 leaves the
    picture as it is).
    """
    hsv = np.array(Image.fromarray(im).convert("HSV")).astype(float)
    fig = figure_mask(im) if figure is None else figure
    keep = (ndimage.gaussian_filter(protect.astype(float), 5)
            if protect is not None else 0.0)
    if protect is not None:
        keep = np.maximum(keep, protect.astype(float))
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    a_start, a_width = ACCENT_RAMP
    v_start, v_width = ACCENT_VALUE_RAMP
    accent = smoothstep((S - a_start) / a_width) * \
        smoothstep((V - v_start) / v_width)

    def compress(knee_ratio):
        knee, ratio = knee_ratio
        kept = ratio + (ACCENT_KEEP - ratio) * accent
        return np.where(S > knee, knee + (S - knee) * kept, S)

    w_base = fig * smoothstep((S - 10) / 20) * (1 - keep) * strength
    wl, _ = band_w(V)
    wd = smoothstep((90 - V) / 30)
    target_c = wl * compress(REPIN_LIGHT) + (1 - wl) * compress(REPIN_MID)
    target_d = compress(REPIN_DARK)

    w_sat = np.zeros_like(S)
    ease_sum = np.zeros_like(H)
    ease_target = np.zeros_like(H)
    for name in REPIN_CHROMA_WINDOWS:
        window = palette_window(name)
        w_chroma_i = window_w(H, *window["hue"]) * w_base * (1 - wd)
        w_sat = np.maximum(w_sat, w_chroma_i)
        ease_i = w_chroma_i * H_TARGET_BLEND * (1 - accent)
        ease_sum = ease_sum + ease_i
        ease_target = ease_target + ease_i * window["hue_target"]
    below = smoothstep((REPIN_SAT_ONLY_V_MAX - V) / 20.0)
    for lo, hi in REPIN_SAT_ONLY_WINDOWS:
        w_sat = np.maximum(w_sat, window_w(H, lo, hi) * below * w_base * (1 - wd))
    s_delta = w_sat * (target_c - S)

    w_dark = w_base * wd
    for lo, hi in REPIN_DARK_EXEMPT:
        w_dark = w_dark * (1 - window_w(H, lo, hi))

    w_skin = np.zeros_like(S)
    for lo, hi in (palette_window(REPIN_SKIN_WINDOW)["hue"],
                   *REPIN_SAT_ONLY_WINDOWS):
        w_skin = np.maximum(w_skin, window_w(H, lo, hi))
    w_skin = w_skin * fig * smoothstep((V - REPIN_SKIN_CAP_V) / 20.0)
    s_delta = s_delta + w_skin * np.minimum(compress(REPIN_SKIN_CAP) - S - s_delta, 0)

    new_S = S + s_delta + w_dark * (target_d - S)
    new_H = H * (1 - ease_sum) + ease_target

    moved = np.abs(new_S - S) > 2
    report = [f"compressed {moved.mean() * 100:.1f}% of frame "
              f"(accent kept {float((accent > 0.5).mean()) * 100:.1f}%)"]
    out = np.stack([new_H, new_S.clip(0, 255), V], axis=-1)
    rgb = np.array(Image.fromarray(out.clip(0, 255).astype(np.uint8), "HSV")
                   .convert("RGB"))
    return rgb, report


def skin_mask(source: np.ndarray) -> np.ndarray:
    """Where the render that fed the redraw drew skin.

    Read off the source because the redraw no longer knows: the hue it put
    there is the same purple as the hair. Opening drops the speckle along
    every warm line. Nothing closes the holes: the lips sit in one, and a
    closed mouth region pins them to the cheek's hue.
    """
    hsv = np.array(Image.fromarray(source).convert("HSV")).astype(float)
    lo, hi = palette_window(REPIN_SKIN_WINDOW)["hue"]
    mask = ((hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
            & (hsv[..., 1] > SKIN_SOURCE_S_MIN)
            & (hsv[..., 2] >= SKIN_SOURCE_V_MIN)
            & figure_mask(source))
    mask = ndimage.binary_opening(mask, np.ones((5, 5)))
    labels, count = ndimage.label(mask)
    if not count:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return np.isin(labels, 1 + np.nonzero(sizes >= SKIN_PIN_MIN_AREA)[0])


def measure(data: bytes) -> dict:
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "LA"):
        rgba = img.convert("RGBA")
        backdrop = Image.new("RGB", rgba.size, (200, 200, 200))
        backdrop.paste(rgba, mask=rgba.getchannel("A"))
        img = backdrop
    im = np.array(img.convert("RGB"))
    hsv = np.array(Image.fromarray(im).convert("HSV")).astype(float)
    edge = np.concatenate([im[:30].reshape(-1, 3), im[-30:].reshape(-1, 3),
                           im[:, :30].reshape(-1, 3), im[:, -30:].reshape(-1, 3)])
    bg = np.median(edge, axis=0).astype(int)
    bg_hsv = np.array(Image.fromarray(bg[None, None].astype(np.uint8))
                      .convert("HSV")).astype(float)[0, 0]
    mask = background_mask(im.astype(int), 18)
    mask |= enclosed_mask(im.astype(int), mask, 4)
    c = 40
    corners = [im[:c, :c], im[:c, -c:], im[-c:, :c], im[-c:, -c:]]
    corner_means = [x.reshape(-1, 3).mean() for x in corners]
    mid = ~mask & (hsv[..., 2] >= FIGURE_MIDTONE_V)
    fig_s = hsv[..., 1][mid] if mid.any() else np.zeros(1)
    light = ~mask & (hsv[..., 2] >= FIGURE_LIGHT_V)
    light_s = float(hsv[..., 1][light].mean()) if light.any() else 0.0
    return {"bg": tuple(bg), "bg_sat": float(bg_hsv[1]),
            "sat": float(hsv[..., 1].mean()),
            "fig_sat_mean": float(fig_s.mean()),
            "fig_sat_p90": float(np.percentile(fig_s, 90)),
            "light_sat": light_s,
            "norm_factor": min(1.0, FIGURE_LIGHT_SAT_TARGET / light_s)
            if light_s else 1.0,
            "corner_spread": float(max(corner_means) - min(corner_means))}


def verdict(m: dict) -> list[str]:
    fails = []
    if not SAT_BAND[0] <= m["sat"] <= SAT_BAND[1]:
        fails.append(f"mean saturation {m['sat']:.1f} outside {SAT_BAND}")
    if m["bg_sat"] > BG_SAT_MAX:
        fails.append(f"background saturation {m['bg_sat']:.1f} > {BG_SAT_MAX}")
    if m["fig_sat_mean"] > FIGURE_SAT_MEAN_MAX:
        fails.append(f"figure midtone saturation {m['fig_sat_mean']:.1f} > "
                     f"{FIGURE_SAT_MEAN_MAX}")
    if m["fig_sat_p90"] > FIGURE_SAT_P90_MAX:
        fails.append(f"figure midtone p90 saturation {m['fig_sat_p90']:.1f} > "
                     f"{FIGURE_SAT_P90_MAX}")
    if m["corner_spread"] > BACKDROP_SPREAD_MAX:
        fails.append(f"backdrop not flat: corner spread "
                     f"{m['corner_spread']:.1f} > {BACKDROP_SPREAD_MAX} "
                     f"(gradient backdrop starves the flood mask; every "
                     f"figure number above is suspect)")
    return fails


def summarize(data: bytes) -> dict:
    """`measure` plus its `verdict`, under one key -- one injection point."""
    m = measure(data)
    m["fails"] = verdict(m)
    return m
