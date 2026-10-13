"""Light a set figure by the backdrop it now stands on.

A delivered figure keeps the exposure and white balance it was drawn in, so on
a night photo or a coloured pattern it floats: too bright, the wrong colour.
Here the backdrop around the card is sampled for its ambient light. The figure
and its outline bands are pulled toward that exposure and colour, by
`strength`; with a direction, the figure is also shaded from that side, its
lit side tinted by the colour of the backdrop's brightest light. A light
scene (sunset, moon) shades and tints the figure as that scene's light redraw
underpaint does instead, and tints a pattern backdrop to the scene as a
delivery in it; a photo backdrop is left as it is."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage

from ...domain.yukari import delivery_style
from .delivery import scene_backdrop
from .light import _light_figure
from .stage import composite

LUMA = np.array([0.2126, 0.7152, 0.0722])
# The ambient is read from the backdrop in this ring around the card, as a
# share of the longest side; the whole backdrop when the ring is empty.
RING = 0.06
# The ambient luminance (linear) a delivery looks right against: the figure
# is darkened toward a dimmer ambient (or lifted toward a brighter one) by
# (ambient / REFERENCE) ** (EXPOSURE_POWER * strength), within the gains.
REFERENCE_LUMINANCE = 0.35
EXPOSURE_POWER = 0.4
GAIN_RANGE = (0.35, 1.15)
# The share of the ambient's colour cast taken on at strength 1, and the
# bounds a channel's cast is held within.
WHITE_BALANCE = 0.5
CAST_RANGE = (0.5, 2.0)
# The backdrop's brightest light is the mean of its pixels above this
# luminance percentile.
HIGHLIGHT_PERCENTILE = 98
# The direction shading's scene at strength 1.
LIT_TINT = 0.4
SHADOW_DEPTH = 0.55
SKIN_KEEP = 0.5


def _linear(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 255) / 255.0) ** 2.2


def _srgb(linear: np.ndarray) -> np.ndarray:
    return np.clip(linear, 0.0, 1.0) ** (1 / 2.2) * 255.0


def ambient(backdrop: np.ndarray, card: np.ndarray) -> np.ndarray:
    """The mean linear colour of `backdrop` in the ring around `card`."""
    height, width = card.shape
    distance = ndimage.distance_transform_edt(~card)
    ring = (distance > 0) & (distance <= RING * max(height, width))
    sample = backdrop[ring] if ring.any() else backdrop.reshape(-1, 3)
    return _linear(sample).mean(axis=0)


def highlight(backdrop: np.ndarray) -> np.ndarray:
    """The colour (0..255, brightest channel at 255) of the backdrop's
    brightest light."""
    linear = _linear(backdrop.reshape(-1, 3))
    luminance = linear @ LUMA
    bright = linear[luminance >= np.percentile(luminance, HIGHLIGHT_PERCENTILE)].mean(axis=0)
    return _srgb(bright / max(float(bright.max()), 1e-6))


def adjustment(ambient_rgb: np.ndarray, strength: float) -> np.ndarray:
    """The per-channel linear gain that takes the figure toward the ambient:
    exposure times white balance."""
    ambient_luminance = max(float(ambient_rgb @ LUMA), 1e-6)
    exposure = (ambient_luminance / REFERENCE_LUMINANCE) ** (EXPOSURE_POWER * strength)
    exposure = float(np.clip(exposure, *GAIN_RANGE))
    cast = np.clip(ambient_rgb / ambient_luminance, *CAST_RANGE)
    balance = 1.0 + WHITE_BALANCE * strength * (cast - 1.0)
    balance = balance / float(balance @ LUMA)
    return exposure * balance


def relight(figure: np.ndarray, figure_alpha: np.ndarray, outline: np.ndarray,
            outline_alpha: np.ndarray, backdrop: np.ndarray, *,
            direction: str | None, strength: float, scene: str | None = None,
            tint_backdrop: bool = False
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Straight 0..255 colours and 0..1 alphas in, at one canvas. Returns
    the relit figure and outline colours, the backdrop (tinted to `scene`
    when `tint_backdrop`) and the recomposited picture."""
    card = (figure_alpha > 0.5) | (outline_alpha > 0.5)
    inside = figure_alpha > 0.5
    figure_colour = figure.astype(float)
    backdrop = backdrop.astype(float)
    if scene is not None:
        direction = direction or delivery_style.LIGHT_FROM_DEFAULT
        if tint_backdrop:
            backdrop = scene_backdrop(backdrop, direction, scene)
        if inside.any():
            figure_colour = np.where(inside[..., None], _light_figure(
                figure_colour, figure_alpha, np.zeros(figure_alpha.shape), direction,
                delivery_style.LIGHT_SCENES[scene]), figure_colour)
    elif direction is not None and inside.any() and strength > 0:
        scene = {"color": tuple(highlight(backdrop)), "tint": LIT_TINT * strength,
                 "shadow": tuple(1.0 - SHADOW_DEPTH * strength * np.ones(3)),
                 "skin_keep": SKIN_KEEP}
        figure_colour = np.where(inside[..., None], _light_figure(
            figure_colour, figure_alpha, np.zeros(figure_alpha.shape), direction, scene),
            figure_colour)
    gain = adjustment(ambient(backdrop, card), strength)
    lit_figure = _srgb(_linear(figure_colour) * gain)
    lit_outline = _srgb(_linear(outline) * gain)
    return lit_figure, lit_outline, backdrop, composite(backdrop, lit_figure, figure_alpha,
                                                        lit_outline, outline_alpha)


def relight_pngs(figure_png: bytes, outline_png: bytes, backdrop_png: bytes, *,
                 direction: str | None, strength: float, scene: str | None = None,
                 tint_backdrop: bool = False) -> tuple[bytes, bytes, bytes, bytes]:
    """`relight` over a staged delivery's layer PNGs: the picture (RGB), the
    relit figure and outline layers (RGBA) and the backdrop layer (RGB)."""
    def decode(data: bytes, mode: str) -> np.ndarray:
        return np.asarray(Image.open(io.BytesIO(data)).convert(mode), dtype=np.float32)

    def png(array: np.ndarray, mode: str) -> bytes:
        output = io.BytesIO()
        Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode).save(
            output, "PNG")
        return output.getvalue()

    figure, outline = decode(figure_png, "RGBA"), decode(outline_png, "RGBA")
    backdrop = decode(backdrop_png, "RGB")
    figure_alpha, outline_alpha = figure[..., 3] / 255.0, outline[..., 3] / 255.0
    lit_figure, lit_outline, lit_backdrop, picture = relight(
        figure[..., :3], figure_alpha, outline[..., :3], outline_alpha,
        backdrop.astype(float), direction=direction, strength=strength, scene=scene,
        tint_backdrop=tint_backdrop)
    return (png(picture, "RGB"), png(np.dstack([lit_figure, figure[..., 3]]), "RGBA"),
            png(np.dstack([lit_outline, outline[..., 3]]), "RGBA"), png(lit_backdrop, "RGB"))
