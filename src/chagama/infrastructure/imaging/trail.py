"""A trail: one stroke across a staged picture, split into the part that
runs behind the figure and the part that crosses in front of it.

Layers here are straight colour 0..255 with alpha 0..1, like stage.py."""
from __future__ import annotations

import io
import math
from collections.abc import Sequence
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

# Long side the trail's particle sizes are tuned at.
REFERENCE_LONG_SIDE = 1440.0
SAMPLE_STEP_PX = 8.0
SUPERSAMPLE = 4


class Trail(NamedTuple):
    points: tuple[tuple[float, float], ...]  # four, shares of the canvas
    front: tuple[bool, ...]                  # per point
    path: str                                # "curve" | "line"
    rule: str                                # "both" | "split" | "start"
    width: float                             # percent of the longest side
    color: str                               # "#rrggbb"
    head: str                                # "light" | "sparkle"
    head_at_end: bool
    seed: int


class Path(NamedTuple):
    xy: np.ndarray        # N x 2 samples
    segment: np.ndarray   # N, index of the control segment each sample is on
    local_t: np.ndarray   # N, position along that segment 0..1
    arc: np.ndarray       # N, cumulative length
    tangent: np.ndarray   # N x 2 unit
    normal: np.ndarray    # N x 2 unit, left of the tangent

    @property
    def length(self) -> float:
        return float(self.arc[-1])


def hex_rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def darker(colour: str, factor: float) -> str:
    return "#" + "".join(f"{round(v * factor):02x}" for v in hex_rgb(colour))


def _catmull_rom(p0, p1, p2, p3, t: np.ndarray) -> np.ndarray:
    """Centripetal Catmull-Rom between p1 and p2."""
    def knot(a, b):
        return max(1e-3, math.sqrt(math.hypot(b[0] - a[0], b[1] - a[1])))
    t1 = knot(p0, p1)
    t2 = t1 + knot(p1, p2)
    t3 = t2 + knot(p2, p3)
    tt = (t1 + (t2 - t1) * t)[:, None]

    def lerp(a, b, ta, tb):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return ((tb - tt) * a + (tt - ta) * b) / (tb - ta)
    a1, a2, a3 = lerp(p0, p1, 0, t1), lerp(p1, p2, t1, t2), lerp(p2, p3, t2, t3)
    return lerp(lerp(a1, a2, 0, t2), lerp(a2, a3, t1, t3), t1, t2)


def sample_path(points: Sequence[Sequence[float]], curve: bool) -> Path:
    p = [np.asarray(q, float) for q in points]
    ext = [2 * p[0] - p[1], *p, 2 * p[-1] - p[-2]]
    xy, seg, lt = [], [], []
    for i in range(len(p) - 1):
        n = max(4, math.ceil(float(np.hypot(*(p[i + 1] - p[i]))) / SAMPLE_STEP_PX))
        t = np.arange(n + (1 if i == len(p) - 2 else 0)) / n
        pts = (_catmull_rom(ext[i], ext[i + 1], ext[i + 2], ext[i + 3], t) if curve
               else p[i] + (p[i + 1] - p[i]) * t[:, None])
        xy.append(pts)
        seg.append(np.full(len(t), i))
        lt.append(t)
    xy = np.concatenate(xy)
    step = np.hypot(*np.diff(xy, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    ahead = np.concatenate([xy[1:], xy[-1:]])
    behind = np.concatenate([xy[:1], xy[:-1]])
    d = ahead - behind
    tangent = d / np.maximum(np.hypot(*d.T), 1e-9)[:, None]
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    return Path(xy, np.concatenate(seg), np.concatenate(lt), arc, tangent, normal)


def pieces(front: Sequence[bool], rule: str) -> list[list[tuple[float, float, bool]]]:
    """Per control segment, its [t0, t1] stretches and whether each is in front."""
    out = []
    for a, b in zip(front, front[1:]):
        if rule == "both":
            out.append([(0.0, 1.0, a and b)])
        elif rule == "start":
            out.append([(0.0, 1.0, a)])
        else:
            out.append([(0.0, 1.0, a)] if a == b else [(0.0, 0.5, a), (0.5, 1.0, b)])
    return out


def front_polygons(path: Path, front: Sequence[bool], rule: str,
                   reach: float) -> list[np.ndarray]:
    """The regions of the stretches that cross in front, as polygons."""
    polys = []
    last = len(path.xy) - 1
    for i, stretches in enumerate(pieces(front, rule)):
        for t0, t1, is_front in stretches:
            if not is_front:
                continue
            ks = np.nonzero((path.segment == i) & (path.local_t >= t0 - 1e-6)
                            & (path.local_t <= t1 + 1e-6))[0]
            if len(ks) == 0:
                continue
            if ks[-1] < last and path.local_t[ks[-1]] < t1:
                ks = np.append(ks, ks[-1] + 1)
            left = path.xy[ks] + path.normal[ks] * reach
            right = path.xy[ks] - path.normal[ks] * reach
            if ks[0] == 0:
                back_ext = path.tangent[0] * reach * 2
                left, right = np.vstack([left[:1] - back_ext, left]), np.vstack([right[:1] - back_ext, right])
            if ks[-1] == last:
                fwd = path.tangent[last] * reach * 2
                left, right = np.vstack([left, left[-1:] + fwd]), np.vstack([right, right[-1:] + fwd])
            polys.append(np.vstack([left, right[::-1]]))
    return polys


def front_mask(polys: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Coverage 0..1 of the front polygons."""
    mask = np.zeros(shape, np.uint8)
    for poly in polys:
        cv2.fillPoly(mask, [np.round(poly * 16).astype(np.int32)], 255,
                     lineType=cv2.LINE_AA, shift=4)
    return mask.astype(float) / 255.0


class Canvas:
    """Premultiplied RGBA accumulation with the `over` operator."""

    def __init__(self, height: int, width: int) -> None:
        self.h, self.w = height, width
        self.rgb = np.zeros((height, width, 3))
        self.a = np.zeros((height, width))

    def _box(self, x0, y0, x1, y1):
        bx0, by0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
        bx1, by1 = min(self.w, int(math.ceil(x1)) + 1), min(self.h, int(math.ceil(y1)) + 1)
        if bx0 >= bx1 or by0 >= by1:
            return None
        return bx0, by0, bx1, by1

    def over(self, box, colour: np.ndarray, alpha: np.ndarray) -> None:
        """colour: h x w x 3 or 3 (straight), alpha: h x w, inside `box`."""
        bx0, by0, bx1, by1 = box
        a = np.clip(alpha, 0.0, 1.0)
        rgb = self.rgb[by0:by1, bx0:bx1]
        dst_a = self.a[by0:by1, bx0:bx1]
        src = np.broadcast_to(colour, rgb.shape) * a[..., None]
        rgb *= (1 - a)[..., None]
        rgb += src
        dst_a *= 1 - a
        dst_a += a

    def grid(self, box):
        bx0, by0, bx1, by1 = box
        ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(float)
        return xs + 0.5, ys + 0.5

    def radial(self, cx, cy, radius, stops) -> None:
        """stops: [(offset, rgb, alpha)], linear between them."""
        box = self._box(cx - radius, cy - radius, cx + radius, cy + radius)
        if box is None or radius <= 0:
            return
        xs, ys = self.grid(box)
        t = np.hypot(xs - cx, ys - cy) / radius
        offs = [s[0] for s in stops]
        alpha = np.interp(t, offs, [s[2] for s in stops], right=0.0)
        colour = np.stack([np.interp(t, offs, [s[1][c] for s in stops]) for c in range(3)], -1)
        self.over(box, colour, alpha)

    def disc(self, cx, cy, radius, rgb, alpha) -> None:
        box = self._box(cx - radius - 1, cy - radius - 1, cx + radius + 1, cy + radius + 1)
        if box is None:
            return
        xs, ys = self.grid(box)
        cover = np.clip(radius + 0.5 - np.hypot(xs - cx, ys - cy), 0.0, 1.0)
        self.over(box, rgb, cover * alpha)

    def _poly_cover(self, polys: Sequence[np.ndarray], box) -> np.ndarray:
        bx0, by0, bx1, by1 = box
        s = SUPERSAMPLE
        mask = np.zeros(((by1 - by0) * s, (bx1 - bx0) * s), np.uint8)
        pts = [np.round((np.asarray(p, float) - (bx0, by0)) * s * 16).astype(np.int32) for p in polys]
        cv2.fillPoly(mask, pts, 255, lineType=cv2.LINE_8, shift=4)
        return cv2.resize(mask, (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_AREA) / 255.0

    def polygon(self, poly: np.ndarray, rgb, alpha) -> None:
        poly = np.asarray(poly, float)
        box = self._box(*poly.min(0), *poly.max(0))
        if box is None:
            return
        self.over(box, rgb, self._poly_cover([poly], box) * alpha)

    def ray(self, cx, cy, angle, length, width, start_rgb, start_alpha, end_rgb) -> None:
        """A thin triangle from the centre, its colour fading along its length."""
        d = np.array([math.cos(angle), math.sin(angle)])
        n = np.array([-d[1], d[0]])
        c = np.array([cx, cy])
        poly = np.array([c + n * width / 2, c + d * length, c - n * width / 2])
        box = self._box(*poly.min(0), *poly.max(0))
        if box is None:
            return
        xs, ys = self.grid(box)
        t = np.clip(((xs - cx) * d[0] + (ys - cy) * d[1]) / length, 0.0, 1.0)
        colour = np.asarray(start_rgb)[None, None, :] * (1 - t)[..., None] + np.asarray(end_rgb)[None, None, :] * t[..., None]
        self.over(box, colour, self._poly_cover([poly], box) * start_alpha * (1 - t))

    def polyline(self, pts: np.ndarray, thickness: float, rgb, alpha) -> None:
        if len(pts) < 2:
            return
        box = self._box(*(pts.min(0) - thickness - 2), *(pts.max(0) + thickness + 2))
        if box is None:
            return
        bx0, by0, bx1, by1 = box
        s = SUPERSAMPLE
        mask = np.zeros(((by1 - by0) * s, (bx1 - bx0) * s), np.uint8)
        local = np.round((pts - (bx0, by0)) * s * 16).astype(np.int32)
        cv2.polylines(mask, [local], False, 255, thickness=max(1, round(thickness * s)),
                      lineType=cv2.LINE_AA, shift=4)
        cover = cv2.resize(mask, (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_AREA) / 255.0
        self.over(box, rgb, cover * alpha)

    def streak(self, cx, cy, angle, radius, rgb, alpha) -> None:
        """An ellipse of light, 3.2 x 0.7 radii long, trailing behind `angle`."""
        reach = radius * 3.6
        box = self._box(cx - reach, cy - reach, cx + reach, cy + reach)
        if box is None:
            return
        xs, ys = self.grid(box)
        c, s = math.cos(angle), math.sin(angle)
        u = (xs - cx) * c + (ys - cy) * s + radius * 0.35
        v = -(xs - cx) * s + (ys - cy) * c
        t = np.hypot(u / 3.2, v / 0.7) / radius
        self.over(box, rgb, np.clip(1 - t, 0.0, 1.0) * alpha)

    def straight(self) -> np.ndarray:
        colour = self.rgb / np.maximum(self.a, 1e-6)[..., None]
        return np.concatenate([np.clip(colour, 0, 255), self.a[..., None]], axis=-1)


class Rng:
    """mulberry32: the browser preview draws with the same sequence."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        m = 0xFFFFFFFF
        self.state = (self.state + 0x6D2B79F5) & m
        a = self.state
        t = ((a ^ (a >> 15)) * (1 | a)) & m
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & m)) & m) ^ t
        return ((t ^ (t >> 14)) & m) / 4294967296.0

    def gauss(self) -> float:
        u1, u2 = self.random(), self.random()
        return math.sqrt(-2.0 * math.log(u1 + 1e-12)) * math.cos(2.0 * math.pi * u2)


def _mix(rgb, f: float) -> list[float]:
    """Toward white for f > 0, toward black for f < 0."""
    target = 255.0 if f > 0 else 0.0
    return [v + (target - v) * abs(f) for v in rgb]


def _star(cx, cy, outer, inner, points, rot) -> list[list[float]]:
    return [[cx + math.cos(rot + k * math.pi / points) * (inner if k % 2 else outer),
             cy + math.sin(rot + k * math.pi / points) * (inner if k % 2 else outer)]
            for k in range(points * 2)]


class _Along:
    """Positions along the path by arc length, pushed sideways by an offset."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self, s: float, off: float) -> tuple[float, float]:
        p = self.path
        x = float(np.interp(s, p.arc, p.xy[:, 0]))
        y = float(np.interp(s, p.arc, p.xy[:, 1]))
        nx = float(np.interp(s, p.arc, p.normal[:, 0]))
        ny = float(np.interp(s, p.arc, p.normal[:, 1]))
        nl = max(math.hypot(nx, ny), 1e-9)
        return x + nx / nl * off, y + ny / nl * off


def _pick(rng: Rng, count: int, length: float, q_of, density) -> list[tuple[float, float]]:
    """Up to `count` arc positions, each candidate kept with `density(q)`."""
    out = []
    for _ in range(count * 3):
        s = rng.random() * length
        q = q_of(s)
        if rng.random() < density(q):
            out.append((s, q))
            if len(out) == count:
                break
    return out


def _body(path: Path, w: float, base, rng: Rng, head_at_end: bool, k: float) -> list:
    """The trail behind the head as primitives: wash, ribbons, glitter, bokeh, sparkles."""
    length = path.length
    at = _Along(path)
    prims: list = []

    def q_of(s):
        return (length - s) / length if head_at_end else s / length

    def sd(q):
        return w * (0.18 + 1.1 * q)

    light = _mix(base, 0.45)
    steps = math.ceil(length / (w * 0.2))
    for n in range(steps):
        s = n * w * 0.2
        q = q_of(s)
        if q < 0.03:
            continue
        x, y = at(s, rng.gauss() * sd(q) * 0.35)
        a = 0.2 * (1 - q) ** 0.5 * min(1.0, q / 0.12)
        prims.append(["radial", x, y, sd(q) * 1.3, [[0, light, a], [1, light, 0.0]]])

    for i in range(2):
        phase, period = rng.random() * 2 * math.pi, w * (1.9 + i * 0.8)
        run: list = []
        colour = _mix(base, 0.15 + i * 0.1)
        for n in range(int(length // 4) + 1):
            s = n * 4.0
            q = q_of(s)
            if 0.01 <= q <= 0.9:
                run.append(list(at(s, sd(q) * (0.55 + 0.35 * i) * math.sin(s / period + phase) * (0.4 + q))))
            elif run:
                prims.append(["polyline", run, (1 + i * 0.4) * k, colour, 0.8])
                run = []
        if run:
            prims.append(["polyline", run, (1 + i * 0.4) * k, colour, 0.8])

    for s, q in _pick(rng, round(length * 8 / k), length, q_of,
                      lambda q: 0.0 if q < 0.02 else min(1.0, q / 0.15) * (1 - q) ** 0.9):
        x, y = at(s, rng.gauss() * sd(q) * 0.45)
        radius = (0.6 + rng.random() ** 2 * 2) * k
        tone = rng.random() * 0.7 - 0.25
        prims.append(["disc", x, y, radius, _mix(base, tone), 0.55 + 0.45 * rng.random()])

    for s, q in _pick(rng, round(length / 8 / k), length, q_of,
                      lambda q: 0.0 if q < 0.02 else (1 - q) ** 0.5):
        x, y = at(s, rng.gauss() * sd(q) * 1.5)
        r = (1.5 + rng.random() ** 2 * 7) * k
        if rng.random() < 0.4:
            soft = _mix(base, 0.5)
            prims.append(["radial", x, y, r * 2.2, [[0, soft, 0.35], [1, soft, 0.0]]])
        else:
            prims.append(["radial", x, y, r, [[0, _mix(base, 0.45), 0.95], [1, _mix(base, -0.05), 0.8]]])

    for s, q in _pick(rng, round(length / 55 / k), length, q_of,
                      lambda q: 0.0 if q < 0.03 else 0.3 + 0.7 * (1 - q)):
        x, y = at(s, rng.gauss() * sd(q) * 1.9)
        if rng.random() < 0.8:
            r = (3 + rng.random() * 7) * k
            r *= 2 if rng.random() < 0.15 else 1
            glow = _mix(base, 0.6)
            prims.append(["radial", x, y, r * 0.9, [[0, glow, 0.45], [1, glow, 0.0]]])
            prims.append(["poly", _star(x, y, r, r * 0.14, 4, -math.pi / 2), _mix(base, 0.35), 0.95])
        else:
            r = (5 + rng.random() * 6) * k
            prims.append(["poly", _star(x, y, r, r * 0.45, 5, -math.pi / 2 + rng.random()),
                          _mix(base, 0.1), 0.9])
    return prims


def _flare(x: float, y: float, direction, w: float, base, light_ground: bool) -> list:
    """Light burst: halo, a streak trailing behind, cross rays and a white core."""
    r = w
    white = [255.0, 255.0, 255.0]
    core = _mix(base, 0.25) if light_ground else white
    edge = _mix(base, -0.1 if light_ground else 0.2)
    fade = _mix(base, 0.3)
    halo = ([[0, list(base), 0.95], [0.25, _mix(base, -0.05), 0.5], [1, list(base), 0.0]] if light_ground
            else [[0, _mix(base, 0.35), 0.75], [0.3, _mix(base, 0.1), 0.35], [1, list(base), 0.0]])
    prims: list = [["radial", x, y, r * 2.8, halo],
                   ["streak", x, y, math.atan2(direction[1], direction[0]), r, _mix(base, 0.55), 0.6]]
    for a, length in ((-math.pi / 2, 3.4), (math.pi / 2, 3.0), (0.0, 2.6), (math.pi, 2.6)):
        prims.append(["ray", x, y, a, r * length, r * 0.09 * 2.6, edge, 0.8, fade])
        prims.append(["ray", x, y, a, r * length, r * 0.09, core, 1.0, fade])
    for j in (1, 3, 5, 7):
        a = j * math.pi / 4 + 0.08
        prims.append(["ray", x, y, a, r * 1.3, r * 0.05 * 2.6, edge, 0.8, fade])
        prims.append(["ray", x, y, a, r * 1.3, r * 0.05, core, 1.0, fade])
    prims.append(["radial", x, y, r * 0.55, [[0, white, 1.0], [0.25, white, 1.0], [0.5, core, 1.0],
                                           [1, _mix(base, 0.2), 0.0]]])
    return prims


def _sparkle_head(x: float, y: float, w: float, base, rng: Rng) -> list:
    r = w * 1.3
    glow = _mix(base, 0.3)
    prims: list = [["radial", x, y, r * 1.6, [[0, glow, 0.55], [1, glow, 0.0]]],
                   ["poly", _star(x, y, r * 0.5, r * 0.12, 4, -math.pi / 4), _mix(base, 0.2), 0.85],
                   ["poly", _star(x, y, r, r * 0.11, 4, -math.pi / 2), _mix(base, 0.05), 1.0],
                   ["poly", _star(x, y, r * 0.55, r * 0.1, 4, -math.pi / 2), _mix(base, 0.6), 1.0]]
    for _ in range(6):
        a = rng.random() * 2 * math.pi
        d = r * (0.7 + rng.random() * 0.9)
        rr = r * (0.1 + rng.random() * 0.16)
        prims.append(["poly", _star(x + math.cos(a) * d, y + math.sin(a) * d, rr, rr * 0.18, 4, -math.pi / 2),
                      _mix(base, 0.35), 0.95])
    return prims


def plan(trail: Trail, height: int, width: int, *, light_ground: bool = False) -> dict:
    """What to draw, in drawing order, before any pixel: the body and head
    primitives, the polygons of the front stretches and whether the head is
    in front. The browser preview builds the same plan."""
    longest = max(height, width)
    k = longest / REFERENCE_LONG_SIDE
    w = trail.width / 100.0 * longest
    path = sample_path([(x * width, y * height) for x, y in trail.points], trail.path == "curve")
    base = [float(v) for v in hex_rgb(trail.color)]
    rng = Rng(trail.seed)
    body = _body(path, w, base, rng, trail.head_at_end, k)
    end = len(path.xy) - 1 if trail.head_at_end else 0
    hx, hy = (float(v) for v in path.xy[end])
    direction = path.tangent[end] if trail.head_at_end else -path.tangent[0]
    head = (_sparkle_head(hx, hy, w, base, rng) if trail.head == "sparkle"
            else _flare(hx, hy, direction, w, base, light_ground))
    stretch = pieces(trail.front, trail.rule)[-1 if trail.head_at_end else 0]
    return {"body": body, "head": head,
            "front": [poly.tolist() for poly in front_polygons(path, trail.front, trail.rule, w * 3.2)],
            "head_in_front": bool(stretch[-1 if trail.head_at_end else 0][2])}


def _draw(canvas: Canvas, prims: list) -> None:
    for prim in prims:
        kind, rest = prim[0], prim[1:]
        if kind == "radial":
            x, y, r, stops = rest
            canvas.radial(x, y, r, [(o, np.asarray(c), a) for o, c, a in stops])
        elif kind == "disc":
            x, y, r, c, a = rest
            canvas.disc(x, y, r, np.asarray(c), a)
        elif kind == "poly":
            pts, c, a = rest
            canvas.polygon(np.asarray(pts), np.asarray(c), a)
        elif kind == "polyline":
            pts, thickness, c, a = rest
            canvas.polyline(np.asarray(pts), thickness, np.asarray(c), a)
        elif kind == "ray":
            x, y, angle, length, wid, c0, a0, c1 = rest
            canvas.ray(x, y, angle, length, wid, np.asarray(c0), a0, np.asarray(c1))
        elif kind == "streak":
            x, y, angle, r, c, a = rest
            canvas.streak(x, y, angle, r, np.asarray(c), a)


def render(trail: Trail, height: int, width: int, *, light_ground: bool = False,
           back_box: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The back and front layers (H x W x 4, straight colour 0..255, alpha 0..1).

    `back_box` (x0, y0, x1, y1 as shares) keeps the back layer inside a frame."""
    drawn = plan(trail, height, width, light_ground=light_ground)
    body = Canvas(height, width)
    _draw(body, drawn["body"])
    layer = body.straight()
    cover = front_mask([np.asarray(p) for p in drawn["front"]], (height, width))
    front = layer.copy()
    front[..., 3] *= cover
    back = layer
    back[..., 3] *= 1 - cover
    head = Canvas(height, width)
    _draw(head, drawn["head"])
    _merge(front if drawn["head_in_front"] else back, head.straight())
    if back_box is not None:
        x0, y0, x1, y1 = back_box
        box = np.zeros((height, width))
        box[max(0, round(y0 * height)):round(y1 * height), max(0, round(x0 * width)):round(x1 * width)] = 1.0
        back[..., 3] *= box
    return back, front


def _merge(under: np.ndarray, top: np.ndarray) -> None:
    """`top` over `under`, in place, both straight."""
    ta, ua = top[..., 3], under[..., 3]
    a = ta + ua * (1 - ta)
    colour = (top[..., :3] * ta[..., None] + under[..., :3] * (ua * (1 - ta))[..., None]) / np.maximum(a, 1e-6)[..., None]
    under[..., :3] = colour
    under[..., 3] = a


def light_ground(backdrop: np.ndarray) -> bool:
    lum = backdrop[..., 0] * 0.3 + backdrop[..., 1] * 0.59 + backdrop[..., 2] * 0.11
    return float(lum.mean()) > 150.0


def _read(data: bytes, mode: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert(mode)).astype(float)


def _png(array: np.ndarray, mode: str) -> bytes:
    out = io.BytesIO()
    Image.fromarray(np.clip(np.round(array), 0, 255).astype(np.uint8), mode).save(out, "PNG")
    return out.getvalue()


def _rgba_png(layer: np.ndarray) -> bytes:
    return _png(np.concatenate([layer[..., :3], layer[..., 3:] * 255.0], axis=-1), "RGBA")


def _over_rgb(under: np.ndarray, layer: np.ndarray) -> np.ndarray:
    a = layer[..., 3:4]
    return under * (1 - a) + layer[..., :3] * a


def trail_pngs(figure_png: bytes, outline_png: bytes, backdrop_png: bytes,
               trail: Trail, back_box: Sequence[float] | None
               ) -> tuple[bytes, bytes, bytes]:
    """(picture, the backdrop with the back part painted in, the front layer)."""
    figure = _read(figure_png, "RGBA")
    outline = _read(outline_png, "RGBA")
    backdrop = _read(backdrop_png, "RGB")
    height, width = backdrop.shape[:2]
    back, front = render(trail, height, width, light_ground=light_ground(backdrop),
                         back_box=back_box)
    behind = _over_rgb(backdrop, back)
    picture = _over_rgb(behind, np.concatenate([outline[..., :3], outline[..., 3:] / 255.0], -1))
    picture = _over_rgb(picture, np.concatenate([figure[..., :3], figure[..., 3:] / 255.0], -1))
    picture = _over_rgb(picture, front)
    return _png(picture, "RGB"), _png(behind, "RGB"), _rgba_png(front)


def over_png(image_png: bytes, layer_png: bytes) -> bytes:
    """The straight RGBA `layer` over the picture, keeping its mode."""
    image = Image.open(io.BytesIO(image_png))
    mode = "RGBA" if image.mode in ("RGBA", "LA") else "RGB"
    picture = np.array(image.convert(mode)).astype(float)
    layer = _read(layer_png, "RGBA")
    layer[..., 3] /= 255.0
    if mode == "RGB":
        return _png(_over_rgb(picture, layer), "RGB")
    pa = picture[..., 3:4] / 255.0
    la = layer[..., 3:4]
    alpha = la + pa * (1 - la)
    colour = (layer[..., :3] * la + picture[..., :3] * pa * (1 - la)) / np.maximum(alpha, 1e-6)
    return _png(np.concatenate([colour, alpha * 255.0], -1), "RGBA")
