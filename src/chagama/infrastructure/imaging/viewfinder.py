"""Mirrorless-camera viewfinder overlay: a rule-of-thirds grid, a focus frame
at the focus point, and a bottom bar reading the exposure settings. Sizes
scale with the shorter side so the overlay looks the same at any resolution."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path(__file__).parents[4] / "assets" / "fonts" / "BarlowCondensed-SemiBold.ttf"
REFERENCE_SIDE = 1280
SHUTTER = "1/100"
WHITE = (255, 255, 255, 235)
FOCUS = (255, 92, 40, 255)
AMBER = (255, 176, 40, 255)
GRID = (255, 255, 255, 150)
BAR = (0, 0, 0, 150)
OUTLINE = (0, 0, 0, 140)


def _centred(draw: ImageDraw.ImageDraw, centre_x: float, baseline: float,
             font: ImageFont.FreeTypeFont,
             segments: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    x = centre_x - draw.textlength("".join(text for text, _ in segments), font=font) / 2
    for text, fill in segments:
        draw.text((x, baseline), text, font=font, fill=fill, anchor="ls",
                  stroke_width=max(1, font.size // 22), stroke_fill=OUTLINE)
        x += draw.textlength(text, font=font)


def viewfinder_png(png: bytes, focus: tuple[float, float], f_number: float) -> bytes:
    base = Image.open(io.BytesIO(png))
    has_alpha = "A" in base.getbands()
    base = base.convert("RGBA")
    w, h = base.size
    u = min(w, h) / REFERENCE_SIDE
    layer = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(layer)

    line = max(1, round(1.5 * u))
    for i in (1, 2):
        draw.line([(w * i / 3, 0), (w * i / 3, h)], fill=GRID, width=line)
        draw.line([(0, h * i / 3), (w, h * i / 3)], fill=GRID, width=line)

    cx, cy = focus[0] * w, focus[1] * h
    half, arm, thickness = 40 * u, 13 * u, round(3 * u)
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * half, cy + sy * half
            draw.line([(x, y), (x - sx * arm, y)], fill=FOCUS, width=thickness)
            draw.line([(x, y), (x, y - sy * arm)], fill=FOCUS, width=thickness)
    tip = 6 * u
    gap = half + 11 * u
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        px, py = cx + dx * gap, cy + dy * gap
        if dx:
            points = [(px + dx * tip, py), (px, py - tip), (px, py + tip)]
        else:
            points = [(px, py + dy * tip), (px - tip, py), (px + tip, py)]
        draw.polygon(points, fill=WHITE)

    bar = 96 * u
    draw.rectangle([0, h - bar, w, h], fill=BAR)
    font = ImageFont.truetype(str(FONT_PATH), round(62 * u))
    baseline = h - bar / 2 + 22 * u
    pitch = w / 3
    _centred(draw, w / 2 - pitch, baseline, font, [(SHUTTER, WHITE)])
    _centred(draw, w / 2, baseline, font, [(f"F{f_number:g}", WHITE)])
    _centred(draw, w / 2 + pitch, baseline, font, [("ISO ", WHITE), ("AUTO", AMBER)])

    out = Image.alpha_composite(base, layer)
    if not has_alpha:
        out = out.convert("RGB")
    buffer = io.BytesIO()
    out.save(buffer, format="PNG")
    return buffer.getvalue()
