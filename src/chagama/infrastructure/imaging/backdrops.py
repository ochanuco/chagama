"""Named procedural backdrops for the sticker and compose delivery paths."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ...domain.yukari import delivery_style


def _grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates normalised by the longest side, x right / y down."""
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    longest = max(height, width)
    return xx / longest, yy / longest


def _palette() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stripes' own lavender/light/white triad, shared by every pattern."""
    from .delivery import parse_color

    base = np.array(parse_color(delivery_style.STRIPES_BASE), dtype=float)
    white = np.array([255.0, 255.0, 255.0])
    light = (base * (1 - delivery_style.STRIPES_CONTRAST)
            + white * delivery_style.STRIPES_CONTRAST)
    return base, light, white


def _edge(v: np.ndarray, soft: float = 0.004) -> np.ndarray:
    """A soft step at `v == 0`, ramped over `soft` widths."""
    return np.clip(v / soft + 0.5, 0.0, 1.0)


def _paint(img: np.ndarray, colour: np.ndarray, t: np.ndarray) -> np.ndarray:
    return img * (1 - t[..., None]) + colour[None, None, :] * t[..., None]


def _flat(height: int, width: int, colour: np.ndarray) -> np.ndarray:
    return np.broadcast_to(colour, (height, width, 3)).astype(float).copy()


def _deep() -> np.ndarray:
    """Lavender pulled toward the purple stroke, for the pencil/paper motifs."""
    from .delivery import parse_color

    base, _light, _white = _palette()
    stroke = np.array(parse_color(delivery_style.OUTLINES[-1]["color"]), dtype=float)
    return (base * delivery_style.BACKDROP_DEEP_LAVENDER_SHARE
           + stroke * delivery_style.BACKDROP_DEEP_STROKE_SHARE)


def _pink() -> np.ndarray:
    """The pink accent shared by the ear/sticker/ornament motifs."""
    from .delivery import parse_color

    return np.array(parse_color(delivery_style.BACKDROP_PINK), dtype=float)


def _value_noise(x: np.ndarray, y: np.ndarray, cells: int, seed: int) -> np.ndarray:
    """Smoothstep-interpolated value noise on a `cells`x`cells` seeded lattice."""
    lattice = np.random.default_rng(seed).random((cells + 3, cells + 3))
    u, v = x * cells, y * cells
    i, j = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i, v - j
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    a, b = lattice[j, i], lattice[j, i + 1]
    c, d = lattice[j + 1, i], lattice[j + 1, i + 1]
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def _burst(img: np.ndarray, x: np.ndarray, y: np.ndarray, strength: float) -> np.ndarray:
    """White radial burst in stripes' own geometry, blended in at `strength`."""
    cx = x.max() * delivery_style.STRIPES_BURST_CENTER[0]
    cy = y.max() * delivery_style.STRIPES_BURST_CENTER[1]
    ang = np.arctan2(y - cy, x - cx)
    rays = np.clip(
        ((ang * delivery_style.STRIPES_BURST_RAYS / (2 * np.pi)) % 1.0 - 0.5) / 0.03 + 0.5,
        0.0, 1.0)
    dist = np.hypot(x - cx, y - cy)
    fade = np.clip(1 - dist / delivery_style.STRIPES_BURST_REACH, 0.0, 1.0) ** 1.5
    t = rays * fade * strength
    white = np.array([255.0, 255.0, 255.0])
    return img * (1 - t[..., None]) + white[None, None, :] * t[..., None]


def stripes(height: int, width: int) -> np.ndarray:
    """Diagonal lavender bands with a white radial burst behind the figure."""
    x, y = _grid(height, width)
    base, light, _white = _palette()

    band = ((x + y) / delivery_style.STRIPES_PITCH) % 1.0
    stripe = _edge(band - 0.5, 0.01)
    img = (light[None, None, :] * (1 - stripe[..., None])
          + base[None, None, :] * stripe[..., None])

    img = _burst(img, x, y, delivery_style.STRIPES_BURST)
    return np.clip(img, 0, 255)


def waveform(height: int, width: int) -> np.ndarray:
    """Voice-meter bars, loud behind the figure, quiet toward the edges."""
    x, y = _grid(height, width)
    base, light, _white = _palette()
    img = _flat(height, width, base)
    row_h = delivery_style.WAVEFORM_ROW_HEIGHT
    pitch = delivery_style.WAVEFORM_PITCH
    bar = delivery_style.WAVEFORM_BAR
    row = np.floor(y / row_h)
    cy = (row + 0.5) * row_h
    i = np.floor(x / pitch)
    bx = (x % pitch) - pitch / 2
    s = np.abs(0.6 * np.sin(i * 0.37 + row * 1.3) + 0.4 * np.sin(i * 0.113 + row * 2.1))
    centre = np.exp(-((x - x.max() / 2) / delivery_style.WAVEFORM_CENTER_SPREAD) ** 2)
    amp = (delivery_style.WAVEFORM_AMP_MIN
          + delivery_style.WAVEFORM_AMP_GAIN * s * (0.3 + 0.7 * centre))
    half = np.maximum(amp - bar / 2, 0)
    d = np.hypot(bx, np.maximum(np.abs(y - cy) - half, 0)) - bar / 2
    img = _paint(img, light, _edge(-d, delivery_style.WAVEFORM_EDGE_SOFT))
    return np.clip(img, 0, 255)


def ears(height: int, width: int) -> np.ndarray:
    """Rabbit-hood ears on a staggered lattice, pink inner ear, alternating lean."""
    x, y = _grid(height, width)
    base, light, _white = _palette()
    pink = _pink()
    img = _flat(height, width, base)
    cell = delivery_style.EARS_CELL
    row = np.floor(y / cell)
    col = np.floor((x + cell / 2 * (row % 2)) / cell)
    gx = ((x + cell / 2 * (row % 2)) % cell) - cell / 2
    gy = (y % cell) - cell / 2
    lean = np.where((row + col) % 2 == 0, 1.0, -1.0) * delivery_style.EARS_LEAN
    for side in (-1, 1):
        ang = side * delivery_style.EARS_TILT + lean
        ox = gx - side * delivery_style.EARS_OFFSET_X
        oy = gy + delivery_style.EARS_OFFSET_Y
        u = ox * np.cos(ang) + oy * np.sin(ang)
        v = -ox * np.sin(ang) + oy * np.cos(ang)
        outer = np.sqrt((u / delivery_style.EARS_OUTER_RX) ** 2
                        + (v / delivery_style.EARS_OUTER_RY) ** 2)
        inner = np.sqrt((u / delivery_style.EARS_INNER_RX) ** 2
                        + ((v - delivery_style.EARS_INNER_OFFSET_Y)
                           / delivery_style.EARS_INNER_RY) ** 2)
        img = _paint(img, light, _edge((1 - outer) * delivery_style.EARS_OUTER_RX,
                                       delivery_style.EARS_EDGE_SOFT))
        img = _paint(img, pink, delivery_style.EARS_INNER_ALPHA
                     * _edge((1 - inner) * delivery_style.EARS_INNER_RX,
                            delivery_style.EARS_EDGE_SOFT))
    return np.clip(img, 0, 255)


def phases(height: int, width: int) -> np.ndarray:
    """Rows of moon phases waxing and waning across the canvas, discs outlined."""
    x, y = _grid(height, width)
    base, light, white = _palette()
    img = _flat(height, width, base)
    cellx = delivery_style.PHASES_CELL_X
    celly = delivery_style.PHASES_CELL_Y
    r = delivery_style.PHASES_RADIUS
    row = np.floor(y / celly)
    col = np.floor((x + cellx / 2 * (row % 2)) / cellx)
    gx = ((x + cellx / 2 * (row % 2)) % cellx) - cellx / 2
    gy = (y % celly) - celly / 2
    k = (col + row * 3) % 8
    shift = r * 2 * np.abs(k / 4 - 1)
    shift = np.where(k < 4, shift, -shift)
    disc = _edge(r - np.hypot(gx, gy), delivery_style.PHASES_EDGE_SOFT)
    shadow = _edge(r - np.hypot(gx - shift, gy), delivery_style.PHASES_EDGE_SOFT)
    lit = disc * (1 - shadow)
    lit = np.where(k == 4, disc, lit)
    ring = _edge(delivery_style.PHASES_RING_WIDTH - np.abs(np.hypot(gx, gy) - r),
                delivery_style.PHASES_EDGE_SOFT)
    img = _paint(img, white, delivery_style.PHASES_RING_ALPHA * ring)
    img = _paint(img, light, lit)
    return np.clip(img, 0, 255)


def hatching(height: int, width: int) -> np.ndarray:
    """Hand-drawn pencil hatching in dashed patches, echoing the sketch line."""
    x, y = _grid(height, width)
    lavender, light, _white = _palette()
    deep = _deep()
    colours = {"lavender": lavender, "deep": deep}
    img = _flat(height, width, light)
    pitch = delivery_style.HATCHING_PITCH
    for k, layer in enumerate(delivery_style.HATCHING_LAYERS):
        dx, dy = layer["direction"]
        p = (x * dx + y * dy) / np.sqrt(2)
        q = (x * dy - y * dx) / np.sqrt(2)
        wobble = ((_value_noise(x, y, delivery_style.HATCHING_WOBBLE_CELLS,
                                delivery_style.HATCHING_WOBBLE_SEED + k) - 0.5)
                 * delivery_style.HATCHING_WOBBLE_AMPLITUDE)
        line = _edge(delivery_style.HATCHING_LINE_HALF_WIDTH
                     - np.abs(((p + wobble) % pitch) - pitch / 2),
                     delivery_style.HATCHING_LINE_EDGE_SOFT)
        stroke_id = np.floor((p + wobble) / pitch)
        dash = _edge(np.sin(q * delivery_style.HATCHING_DASH_FREQ
                            + stroke_id * delivery_style.HATCHING_DASH_STROKE_FREQ)
                    + delivery_style.HATCHING_DASH_BIAS,
                    delivery_style.HATCHING_DASH_EDGE_SOFT)
        patch = _edge(_value_noise(x, y, delivery_style.HATCHING_PATCH_CELLS,
                                   delivery_style.HATCHING_PATCH_SEED + k)
                     - layer["threshold"], delivery_style.HATCHING_PATCH_EDGE_SOFT)
        img = _paint(img, colours[layer["colour"]], layer["alpha"] * line * dash * patch)
    return np.clip(img, 0, 255)


def torn(height: int, width: int) -> np.ndarray:
    """Hand-cut paper layers behind the figure, white paper edges like the rim."""
    x, y = _grid(height, width)
    base, light, white = _palette()
    deep = _deep()
    img = _flat(height, width, deep)
    cx = x.max() * delivery_style.TORN_CENTER[0]
    cy = y.max() * delivery_style.TORN_CENTER[1]
    d = np.hypot(x - cx, y - cy)
    ang = np.arctan2(y - cy, x - cx)
    rng = np.random.default_rng(delivery_style.TORN_SEED)
    colours = {"lavender": base, "light": light}
    n = delivery_style.TORN_VERTICES
    for radius, colour_name in delivery_style.TORN_LAYERS:
        colour = colours[colour_name]
        verts = radius * (1 + rng.uniform(-delivery_style.TORN_JITTER,
                                          delivery_style.TORN_JITTER, n))
        verts = np.append(verts, verts[0])
        t = (ang + np.pi) / (2 * np.pi) * n
        i = np.floor(t).astype(int) % n
        f = t - np.floor(t)
        rr = verts[i] * (1 - f) + verts[i + 1] * f
        img = _paint(img, white, _edge(rr + delivery_style.TORN_PAPER_LIP - d,
                                       delivery_style.TORN_EDGE_SOFT))
        img = _paint(img, colour, _edge(rr - d, delivery_style.TORN_EDGE_SOFT))
    return np.clip(img, 0, 255)


def stickers(height: int, width: int) -> np.ndarray:
    """Scattered crescents, sparkles and hood ears, seeded, sparse."""
    x, y = _grid(height, width)
    base, light, _white = _palette()
    pink = _pink()
    img = _flat(height, width, base)
    rng = np.random.default_rng(delivery_style.STICKERS_SEED)
    longest = max(height, width)
    cell = delivery_style.STICKERS_CELL
    jitter = delivery_style.STICKERS_JITTER
    scale_lo, scale_hi = delivery_style.STICKERS_SCALE_RANGE
    angle_lo, angle_hi = delivery_style.STICKERS_ANGLE_RANGE
    radius = int(delivery_style.STICKERS_MOTIF_RADIUS_SHARE * longest)
    soft = delivery_style.STICKERS_EDGE_SOFT
    for gy0 in np.arange(0, y.max() + cell, cell):
        for gx0 in np.arange(0, x.max() + cell, cell):
            px = gx0 + cell / 2 + rng.uniform(-jitter, jitter)
            py = gy0 + cell / 2 + rng.uniform(-jitter, jitter)
            kind = rng.integers(0, 3)
            s = rng.uniform(scale_lo, scale_hi)
            ang = rng.uniform(angle_lo, angle_hi)
            colour = pink if rng.random() < delivery_style.STICKERS_PINK_CHANCE else light
            cxp, cyp = int(px * longest), int(py * longest)
            y0, y1 = max(cyp - radius, 0), min(cyp + radius, height)
            x0, x1 = max(cxp - radius, 0), min(cxp + radius, width)
            if y0 >= y1 or x0 >= x1:
                continue
            lx, ly = x[y0:y1, x0:x1] - px, y[y0:y1, x0:x1] - py
            u = lx * np.cos(ang) + ly * np.sin(ang)
            v = -lx * np.sin(ang) + ly * np.cos(ang)
            if kind == 0:
                r = delivery_style.STICKERS_MOON_RADIUS * s
                bite_dx = delivery_style.STICKERS_MOON_BITE_OFFSET[0] * r
                bite_dy = delivery_style.STICKERS_MOON_BITE_OFFSET[1] * r
                t = (_edge(r - np.hypot(u, v), soft)
                    * (1 - _edge(r * delivery_style.STICKERS_MOON_BITE_SHARE
                                - np.hypot(u - bite_dx, v - bite_dy), soft)))
            elif kind == 1:
                r = delivery_style.STICKERS_SPARKLE_RADIUS * s
                t = _edge((np.sqrt(r) - (np.sqrt(np.abs(u)) + np.sqrt(np.abs(v))))
                         * delivery_style.STICKERS_SPARKLE_SCALE,
                         delivery_style.STICKERS_SPARKLE_EDGE_SOFT)
            else:
                t = np.zeros_like(u)
                for side in (-1, 1):
                    a2 = side * delivery_style.STICKERS_EAR_TILT
                    ox = u - side * delivery_style.STICKERS_EAR_OFFSET * s
                    oy = v
                    uu = ox * np.cos(a2) + oy * np.sin(a2)
                    vv = -ox * np.sin(a2) + oy * np.cos(a2)
                    t = np.maximum(t, _edge(
                        (1 - np.sqrt((uu / (delivery_style.STICKERS_EAR_RX * s)) ** 2
                                    + (vv / (delivery_style.STICKERS_EAR_RY * s)) ** 2))
                        * delivery_style.STICKERS_EAR_RX * s, soft))
            img[y0:y1, x0:x1] = _paint(img[y0:y1, x0:x1], colour, t)
    return np.clip(img, 0, 255)


def ornament(height: int, width: int) -> np.ndarray:
    """Yukari's hair ornament, ringed disc with a small satellite, as a lattice motif."""
    x, y = _grid(height, width)
    base, light, _white = _palette()
    deep = _deep()
    pink = _pink()
    img = _flat(height, width, base)
    cell = delivery_style.ORNAMENT_CELL
    row = np.floor(y / cell)
    gx = ((x + cell / 2 * (row % 2)) % cell) - cell / 2
    gy = (y % cell) - cell / 2
    big = np.hypot(gx, gy)
    off_x, off_y = delivery_style.ORNAMENT_SATELLITE_OFFSET
    sat = np.hypot(gx - off_x, gy - off_y)
    soft = delivery_style.ORNAMENT_EDGE_SOFT
    img = _paint(img, deep, delivery_style.ORNAMENT_RIM_ALPHA
                * _edge(delivery_style.ORNAMENT_RIM_RADIUS - big, soft))
    img = _paint(img, light, _edge(delivery_style.ORNAMENT_RING_RADIUS - big, soft))
    img = _paint(img, pink, delivery_style.ORNAMENT_HUB_ALPHA
                * _edge(delivery_style.ORNAMENT_CENTER_RADIUS - big, soft))
    img = _paint(img, deep, delivery_style.ORNAMENT_RIM_ALPHA
                * _edge(delivery_style.ORNAMENT_CENTER_RADIUS - sat, soft))
    img = _paint(img, pink, delivery_style.ORNAMENT_HUB_ALPHA
                * _edge(delivery_style.ORNAMENT_SATELLITE_HUB_RADIUS - sat, soft))
    return np.clip(img, 0, 255)


def dots(height: int, width: int) -> np.ndarray:
    """Large white polka dots on lavender, offset rows."""
    x, y = _grid(height, width)
    base, light, _white = _palette()
    cell = delivery_style.DOTS_CELL
    row = np.floor(y / cell)
    gx = ((x + cell / 2 * (row % 2)) % cell) - cell / 2
    gy = (y % cell) - cell / 2
    radius = cell * delivery_style.DOTS_RADIUS_SHARE
    t = _edge(radius - np.hypot(gx, gy), delivery_style.DOTS_EDGE_SOFT)
    img = _paint(_flat(height, width, base), light, t)
    return np.clip(img, 0, 255)


def gingham(height: int, width: int) -> np.ndarray:
    """Lavender gingham: overlapping translucent bands, darker at crossings."""
    x, y = _grid(height, width)
    base, light, _white = _palette()
    cell = delivery_style.GINGHAM_CELL
    half_band = cell * delivery_style.GINGHAM_BAND_SHARE
    bx = _edge(half_band - np.abs((x % cell) - cell / 2), delivery_style.GINGHAM_EDGE_SOFT)
    by = _edge(half_band - np.abs((y % cell) - cell / 2), delivery_style.GINGHAM_EDGE_SOFT)
    img = _flat(height, width, light)
    img = _paint(img, base, delivery_style.GINGHAM_BAND_ALPHA * bx)
    img = _paint(img, base, delivery_style.GINGHAM_BAND_ALPHA * by)
    return np.clip(img, 0, 255)


def moons(height: int, width: int) -> np.ndarray:
    """Small white crescents scattered on a staggered lattice, weak burst behind."""
    x, y = _grid(height, width)
    base, light, _white = _palette()
    cell = delivery_style.MOONS_CELL
    row = np.floor(y / cell)
    gx = ((x + cell / 2 * (row % 2)) % cell) - cell / 2
    gy = (y % cell) - cell / 2
    hole_dx, hole_dy = delivery_style.MOONS_HOLE_OFFSET
    moon = (_edge(delivery_style.MOONS_RADIUS - np.hypot(gx, gy))
           * (1 - _edge(delivery_style.MOONS_HOLE_RADIUS
                       - np.hypot(gx - hole_dx, gy - hole_dy))))
    img = _paint(_flat(height, width, base), light, moon)
    img = _burst(img, x, y, delivery_style.MOONS_BURST)
    return np.clip(img, 0, 255)


def halftone(height: int, width: int) -> np.ndarray:
    """White halftone dots growing toward the lower-left, big faint crescent behind."""
    x, y = _grid(height, width)
    base, _light, white = _palette()
    cell = delivery_style.HALFTONE_CELL
    gx, gy = (x % cell) - cell / 2, (y % cell) - cell / 2
    t = np.clip((y - x + 0.5) / delivery_style.HALFTONE_GRADIENT_SPAN, 0, 1)
    radius = cell * (delivery_style.HALFTONE_RADIUS_MIN
                     + delivery_style.HALFTONE_RADIUS_GROWTH * t)
    img = _paint(_flat(height, width, base), white,
                delivery_style.HALFTONE_DOT_ALPHA
                * _edge(radius - np.hypot(gx, gy), delivery_style.HALFTONE_EDGE_SOFT))
    ax = x.max() * delivery_style.HALFTONE_CRESCENT_CENTER[0]
    ay = y.max() * delivery_style.HALFTONE_CRESCENT_CENTER[1]
    hole_dx, hole_dy = delivery_style.HALFTONE_CRESCENT_HOLE_OFFSET
    crescent = (_edge(delivery_style.HALFTONE_CRESCENT_RADIUS - np.hypot(x - ax, y - ay))
               * (1 - _edge(delivery_style.HALFTONE_CRESCENT_HOLE_RADIUS
                           - np.hypot(x - ax + hole_dx, y - ay - hole_dy))))
    img = _paint(img, white, delivery_style.HALFTONE_CRESCENT_ALPHA * crescent)
    return np.clip(img, 0, 255)


def sunburst(height: int, width: int) -> np.ndarray:
    """Alternating wide wedges from behind the figure, no bands."""
    x, y = _grid(height, width)
    base, light, _white = _palette()
    cx = x.max() * delivery_style.SUNBURST_CENTER[0]
    cy = y.max() * delivery_style.SUNBURST_CENTER[1]
    ang = np.arctan2(y - cy, x - cx)
    wedge = _edge(((ang * delivery_style.SUNBURST_WEDGES / (2 * np.pi)) % 1.0) - 0.5,
                 delivery_style.SUNBURST_EDGE_SOFT)
    img = _paint(_flat(height, width, light), base, wedge)
    return np.clip(img, 0, 255)


def chevron(height: int, width: int) -> np.ndarray:
    """Horizontal zigzag bands."""
    x, y = _grid(height, width)
    base, light, _white = _palette()
    zig = (delivery_style.CHEVRON_AMPLITUDE
          * np.abs(((x / delivery_style.CHEVRON_PERIOD) % 1.0) * 2 - 1))
    band = ((y + zig) / delivery_style.CHEVRON_PITCH) % 1.0
    img = _paint(_flat(height, width, light), base,
                _edge(band - 0.5, delivery_style.CHEVRON_EDGE_SOFT))
    return np.clip(img, 0, 255)


def checker(height: int, width: int) -> np.ndarray:
    """Diagonal checkerboard, lavender and light, weak burst behind."""
    x, y = _grid(height, width)
    base, light, _white = _palette()
    diagonal = delivery_style.CHECKER_CELL * 2 ** 0.5
    u, v = (x + y) / diagonal, (x - y) / diagonal
    c = (np.floor(u) + np.floor(v)) % 2
    img = _paint(_flat(height, width, light), base, c)
    img = _burst(img, x, y, delivery_style.CHECKER_BURST)
    return np.clip(img, 0, 255)


PATTERNS = {
    "stripes": stripes,
    "waveform": waveform,
    "ears": ears,
    "phases": phases,
    "hatching": hatching,
    "torn": torn,
    "stickers": stickers,
    "ornament": ornament,
    "dots": dots,
    "gingham": gingham,
    "moons": moons,
    "halftone": halftone,
    "sunburst": sunburst,
    "chevron": chevron,
    "checker": checker,
}


def render(backdrop: str | None, height: int, width: int) -> np.ndarray:
    """The delivery backdrop as a full `(height, width, 3)` float array.

    `None` is the flat delivery default; a `PATTERNS` key is that procedural
    pattern; anything else must be a `#RRGGBB` colour, broadcast flat.
    """
    from .delivery import parse_color

    if backdrop in PATTERNS:
        return PATTERNS[backdrop](height, width)
    color = parse_color(backdrop if backdrop is not None else delivery_style.BACKDROP)
    return np.broadcast_to(
        np.array(color, dtype=float), (height, width, 3)).copy()


def is_backdrop(value: str) -> bool:
    """True for a `PATTERNS` key or a parseable `#RRGGBB` colour."""
    if value in PATTERNS:
        return True
    from .delivery import parse_color

    try:
        parse_color(value)
    except SystemExit:
        return False
    return True


def thumbnail(name: str) -> bytes:
    """PNG bytes of the named pattern alone, at the catalog's thumbnail size."""
    width = delivery_style.BACKDROP_THUMBNAIL_WIDTH
    height = delivery_style.BACKDROP_THUMBNAIL_HEIGHT
    pixels = np.clip(PATTERNS[name](height, width), 0, 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(output, "PNG")
    return output.getvalue()
