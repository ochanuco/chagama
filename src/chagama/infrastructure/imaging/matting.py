"""Per-pixel alpha for the delivery cut: a trimap from the retraced matte,
solved by a matting model, see-through clothing unmixed from the green key,
and the figure's own colour recovered under that alpha.

The model and the foreground estimator are passed in, so this module stays
numpy-only; the worker's node pack supplies both.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image
from scipy import ndimage

from ...domain.yukari import delivery_style
from . import delivery

# (rgb uint8 HxWx3, trimap uint8 HxW of 0/128/255) -> alpha float HxW in 0..1
Predict = Callable[[np.ndarray, np.ndarray], np.ndarray]
# (rgb float HxWx3 in 0..1, alpha float HxW) -> foreground float HxWx3 in 0..1
Foreground = Callable[[np.ndarray, np.ndarray], np.ndarray]

KNOWN_FIGURE = 255
UNKNOWN = 128
KNOWN_BACKDROP = 0


def trimap(figure: np.ndarray, width: int) -> np.ndarray:
    known = np.full(figure.shape, UNKNOWN, dtype=np.uint8)
    known[ndimage.binary_erosion(figure, iterations=width)] = KNOWN_FIGURE
    known[~ndimage.binary_dilation(figure, iterations=width)] = KNOWN_BACKDROP
    return known


def _starts(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, step))
    return starts + [length - tile]


def _feather(size: int, overlap: int) -> np.ndarray:
    index = np.arange(size, dtype=float)
    return np.clip(np.minimum(index + 1, size - index) / max(overlap, 1), 0.0, 1.0)


def tiled(predict: Predict, rgb: np.ndarray, known: np.ndarray, tile: int,
          overlap: int) -> np.ndarray:
    """`predict` over the tiles that hold unknown pixels, feathered together.
    Pixels no tile visits keep the trimap's own 0 or 1."""
    height, width = known.shape
    unknown = known == UNKNOWN
    total = np.zeros(known.shape)
    weight = np.zeros(known.shape)
    step = tile - overlap
    for top in _starts(height, tile, step):
        for left in _starts(width, tile, step):
            window = (slice(top, top + tile), slice(left, left + tile))
            if not unknown[window].any():
                continue
            alpha = predict(rgb[window], known[window])
            feather = np.outer(_feather(alpha.shape[0], overlap),
                               _feather(alpha.shape[1], overlap))
            total[window] += alpha * feather
            weight[window] += feather
    alpha = (known == KNOWN_FIGURE).astype(float)
    solved = unknown & (weight > 0)
    alpha[solved] = total[solved] / weight[solved]
    return alpha


def settle(alpha: np.ndarray, known: np.ndarray) -> np.ndarray:
    """`alpha` clipped to the unit range with the trimap's known regions
    forced onto it."""
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[known == KNOWN_FIGURE] = 1.0
    alpha[known == KNOWN_BACKDROP] = 0.0
    return alpha


def green_key(px: np.ndarray) -> np.ndarray | None:
    """The raw backdrop's colour when it is a green key, else None."""
    key = delivery._corner_seed(px)
    if key[1] - max(key[0], key[2]) < delivery_style.ENCLOSED_KEY_MIN_GREEN_EXCESS:
        return None
    return key


def key_share(px: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Each pixel's green excess as a share of the key's own: 1 on the key,
    0 or less on a neutral or on any colour green does not lead."""
    excess = px[..., 1] - np.maximum(px[..., 0], px[..., 2])
    return excess / max(float(key[1] - max(key[0], key[2])), 1.0)


def translucent(px: np.ndarray, alpha: np.ndarray, known: np.ndarray) -> np.ndarray:
    """The figure pixels the green key shows through: key-tinted components
    that reach the trimap's known figure. Empty off a green key."""
    key = green_key(px)
    if key is None:
        return np.zeros(alpha.shape, dtype=bool)
    tinted = (alpha > 0) & (key_share(px, key) >= delivery_style.TRANSLUCENT_MIN_KEY_SHARE)
    labels, count = ndimage.label(tinted, ndimage.generate_binary_structure(2, 2))
    if not count:
        return tinted
    index = range(1, count + 1)
    sizes = ndimage.sum(tinted, labels, index)
    reach = ndimage.sum(known == KNOWN_FIGURE, labels, index)
    keep = 1 + np.nonzero((sizes >= delivery_style.TRANSLUCENT_MIN_AREA) & (reach > 0))[0]
    return np.isin(labels, keep)


def see_through(px: np.ndarray, alpha: np.ndarray, known: np.ndarray) -> np.ndarray:
    """`alpha` capped at 1 - key share on the `translucent` pixels: a
    material green does not lead, laid over the key at that coverage."""
    region = translucent(px, alpha, known)
    if not region.any():
        return alpha
    cap = np.clip(1.0 - key_share(px, green_key(px)), 0.0, 1.0)
    return np.where(region, np.minimum(alpha, cap), alpha)


def pocket_key(px: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """`alpha` capped in the pockets the figure encloses that the binary
    cut kept: there the key diluted through a dark layer (its share under
    `ENCLOSED_KEY_DILUTED_RATIO` of its projection, which a true shade
    never is) reads at 1 - share / `POCKET_KEY_FULL_SHARE`, not opaque.
    Only tinted components of at least `POCKET_KEY_MIN_AREA` pixels sitting
    at least `POCKET_KEY_MIN_DEPTH_PX` inside the silhouette count, so the
    solved edge keeps its alpha. Untouched off a green key."""
    key = green_key(px)
    if key is None:
        return alpha
    share = key_share(px, key)
    projection = (px * key).sum(axis=2) / max(float((key * key).sum()), 1.0)
    diluted = (share > 0) & (
        share <= delivery_style.ENCLOSED_KEY_DILUTED_RATIO * projection)
    if not diluted.any():
        return alpha
    filled = ndimage.binary_fill_holes(alpha > 0)
    depth = ndimage.distance_transform_edt(filled)
    tinted = diluted & filled & (
        depth >= delivery_style.POCKET_KEY_MIN_DEPTH_PX)
    labels, count = ndimage.label(
        tinted, ndimage.generate_binary_structure(2, 2))
    if not count:
        return alpha
    sizes = ndimage.sum(tinted, labels, range(1, count + 1))
    region = np.isin(labels, 1 + np.nonzero(
        sizes >= delivery_style.POCKET_KEY_MIN_AREA)[0])
    if not region.any():
        return alpha
    cap = np.clip(1.0 - share / delivery_style.POCKET_KEY_FULL_SHARE, 0.0, 1.0)
    return np.where(region, np.minimum(alpha, cap), alpha)


def clearer(px: np.ndarray, alpha: np.ndarray, gain: float) -> np.ndarray:
    """`alpha` with the key showing through the see-through inside scaled by
    `gain`: 1 - gain * (1 - alpha) there. The soft edge is left as is."""
    if gain == 1.0:
        return alpha
    inside = ndimage.binary_erosion(alpha > 0, iterations=delivery_style.MATTING_TRIMAP_PX)
    known = np.where(inside, KNOWN_FIGURE, UNKNOWN).astype(np.uint8)
    region = translucent(px, alpha, known) & (alpha < 1.0)
    return np.where(region, np.clip(1.0 - gain * (1.0 - alpha), 0.0, 1.0), alpha)


def unmixed(px: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Where the green key shows through the figure's inside, the colour
    the key was blended with at `alpha`; elsewhere NaN. The soft edge, within
    the trimap band of the cut, is left to the foreground estimate."""
    solved = np.full(px.shape, np.nan)
    key = green_key(px)
    if key is None:
        return solved
    inside = ndimage.binary_erosion(alpha > 0, iterations=delivery_style.MATTING_TRIMAP_PX)
    region = (inside & (alpha < 1.0)
              & (alpha >= delivery_style.TRANSLUCENT_MIN_SOLVE_ALPHA)
              & (key_share(px, key) >= delivery_style.TRANSLUCENT_MIN_KEY_SHARE))
    safe = np.where(region, alpha, 1.0)[..., None]
    colour = np.clip(key + (px - key) / safe, 0.0, 255.0)
    return np.where(region[..., None], colour, solved)


_EIGHT = ndimage.generate_binary_structure(2, 2)


def white_backdrop(px: np.ndarray) -> bool:
    """Whether the raw backdrop is white (`WHITE_BACKDROP_MIN`)."""
    return bool(delivery._corner_seed(px).min() >= delivery_style.WHITE_BACKDROP_MIN)


_NEIGHBOURS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
_PAPER = 255.0


def _shifted(array: np.ndarray, dy: int, dx: int, fill: float) -> np.ndarray:
    """`array` seen from each pixel's (dy, dx) neighbour; `fill` off the edge."""
    out = np.full_like(array, fill)
    height, width = array.shape[:2]
    out[max(-dy, 0):height - max(dy, 0), max(-dx, 0):width - max(dx, 0)] = (
        array[max(dy, 0):height - max(-dy, 0), max(dx, 0):width - max(-dx, 0)])
    return out


def _uphill(deficit: np.ndarray, open_: np.ndarray) -> np.ndarray:
    """Per pixel, the largest `deficit` among its 8 neighbours in `open_`."""
    masked = np.where(open_, deficit, -1.0)
    return np.max([_shifted(masked, dy, dx, -1.0) for dy, dx in _NEIGHBOURS], axis=0)


def _border_labels(labels: np.ndarray) -> np.ndarray:
    """The nonzero labels that touch the picture's edge."""
    edge = np.unique(np.concatenate(
        (labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    return edge[edge > 0]


def drawn_alpha(px: np.ndarray, soft: np.ndarray) -> np.ndarray:
    """The alpha of a picture drawn on white: opaque but for the paper that
    reaches the backdrop and the antialias ramp from it up to the first ink
    ridge, whose alpha is its share of that ink (`WHITE_*`). `soft` is the
    matte model's output in 0..1."""
    deficit = (_PAPER - px).max(axis=2)
    whiteish = deficit < delivery_style.WHITE_PAPER_DEFICIT
    figure = soft >= 0.5
    labels, _ = ndimage.label(whiteish & (soft < delivery_style.WHITE_PAPER_MAX_MATTE))
    open_white = np.isin(labels, _border_labels(labels))
    labels, _ = ndimage.label(whiteish)
    seen = np.unique(labels[~figure & whiteish])
    pockets = np.setdiff1d(seen[seen > 0], _border_labels(labels))
    if pockets.size:
        mean = ndimage.mean(deficit, labels, pockets)
        pockets = pockets[np.asarray(mean) <= delivery_style.WHITE_POCKET_MAX_DEFICIT]
    pocket = np.isin(labels, pockets)
    open_white |= pocket
    near_ink = ndimage.binary_dilation(
        ~whiteish, _EIGHT, iterations=delivery_style.WHITE_NEAR_INK_PX)
    free = delivery_style.WHITE_PAPER_FREE_MATTE
    paper = open_white & (near_ink | (soft < free))
    faint = ((deficit < delivery_style.WHITE_FAINT_DEFICIT)
             & (np.ptp(px, axis=2) < delivery_style.WHITE_FAINT_CHROMA)
             & ((soft < free) | ndimage.binary_dilation(pocket, iterations=2)))
    labels, _ = ndimage.label(paper | faint)
    ids = np.unique(labels[paper])
    paper = np.isin(labels, ids[ids > 0])
    alpha = np.ones(soft.shape)
    alpha[paper] = 0.0
    white_on_white = open_white & ~paper
    alpha[white_on_white] = np.clip(
        (soft[white_on_white] - free) / (1.0 - 2.0 * free), 0.0, 1.0)
    reached = paper.copy()
    ramp = np.zeros(soft.shape, dtype=bool)
    for _ in range(delivery_style.WHITE_RAMP_STEPS):
        front = ndimage.binary_dilation(reached) & ~reached & ~whiteish
        lowest = np.min([_shifted(np.where(reached, deficit, np.inf), dy, dx, np.inf)
                         for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))], axis=0)
        step = (front & (deficit > lowest)
                & (_uphill(deficit, ~reached) - deficit > delivery_style.WHITE_RAMP_RISE))
        ramp |= step
        reached |= step
    ink = ridge_colour(px, ramp)
    toward = _PAPER - ink
    share = ((_PAPER - px) * toward).sum(axis=2) / np.maximum(
        (toward * toward).sum(axis=2), 1e-6)
    alpha[ramp] = np.clip(share[ramp], 0.0, 1.0)
    far = (ndimage.distance_transform_edt(~ndimage.binary_fill_holes(figure))
           > delivery_style.WHITE_FAR_PX)
    return np.where(far, np.minimum(alpha, soft), alpha)


def ridge_colour(px: np.ndarray, ramp: np.ndarray) -> np.ndarray:
    """`px`, with each `ramp` pixel taking the colour of the ink ridge it
    climbs to: up the steepest neighbour for at most `WHITE_RAMP_STEPS` + 1
    steps."""
    deficit = (_PAPER - px).max(axis=2)
    height, width = deficit.shape
    ys, xs = np.nonzero(ramp)
    for _ in range(delivery_style.WHITE_RAMP_STEPS + 1):
        best_y, best_x, best = ys.copy(), xs.copy(), deficit[ys, xs]
        for dy, dx in _NEIGHBOURS:
            ny = np.clip(ys + dy, 0, height - 1)
            nx = np.clip(xs + dx, 0, width - 1)
            higher = deficit[ny, nx] > best
            best_y = np.where(higher, ny, best_y)
            best_x = np.where(higher, nx, best_x)
            best = np.where(higher, deficit[ny, nx], best)
        ys, xs = best_y, best_x
    colour = px.copy()
    colour[ramp] = px[ys, xs]
    return colour


def key_split(px: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each pixel's share of the key along the key's own chroma (1 on the
    key, 0 on a grey or a colour across it), and what is left once that
    share of the key is taken out (premultiplied)."""
    direction = key - key.mean()
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    along = ((px - px.mean(axis=2, keepdims=True)) * direction).sum(axis=2)
    share = np.clip(along / max(float((key - key.mean()) @ direction), 1e-6), 0.0, 1.0)
    return share, np.clip(px - share[..., None] * key, 0.0, 255.0)


def free_lines(px: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Thin dark marks drawn on the open key: the key darkened, with the
    clean key `CHROMA_LINE_SIDE_PX` across the mark on both sides. A rim
    with the figure on one side is not one."""
    shade = (px * key).sum(axis=2) / max(float((key * key).sum()), 1.0)
    clamped = np.clip(shade, 0.0, 1.0)
    residual = np.abs(px - clamped[..., None] * key).max(axis=2)
    keyish = ((residual <= delivery_style.MATTE_EDGE_TOLERANCE)
              & (shade >= delivery_style.CHROMA_DARK_LINE))
    v = ndimage.gaussian_filter(1 - clamped, 1.0)
    vyy = np.gradient(np.gradient(v, axis=0), axis=0)
    vxx = np.gradient(np.gradient(v, axis=1), axis=1)
    vxy = np.gradient(np.gradient(v, axis=0), axis=1)
    theta = 0.5 * np.arctan2(2 * vxy, vxx - vyy)
    across = (vxx + vyy) / 2 - np.sqrt(((vxx - vyy) / 2) ** 2 + vxy ** 2)
    nx, ny = -np.sin(theta), np.cos(theta)
    clean = ((residual <= delivery_style.MATTE_EDGE_TOLERANCE)
             & (shade >= delivery_style.CHROMA_LINE_CLEAN_SHADE)).astype(float)
    yy, xx = np.mgrid[0:shade.shape[0], 0:shade.shape[1]].astype(float)
    both = np.ones(shade.shape, dtype=bool)
    for sign in (1, -1):
        for step in delivery_style.CHROMA_LINE_SIDE_PX:
            both &= ndimage.map_coordinates(
                clean, [yy + sign * step * ny, xx + sign * step * nx],
                order=1, mode="nearest") >= 0.99
    line = (keyish & (shade <= delivery_style.CHROMA_LINE_MAX_SHADE)
            & (across < 0) & both)
    line = ndimage.binary_dilation(line, _EIGHT) & keyish & (shade <= 0.95)
    labels, count = ndimage.label(line, _EIGHT)
    if not count:
        return line
    keep = [index + 1 for index, box in enumerate(ndimage.find_objects(labels))
            if max(box[0].stop - box[0].start, box[1].stop - box[1].start)
            >= delivery_style.CHROMA_LINE_MIN_LENGTH]
    return np.isin(labels, keep)


def chroma_edge(px: np.ndarray, figure: np.ndarray, key: np.ndarray) -> np.ndarray:
    """`figure` with its soft edge keyed against the green `key` (see
    `delivery_style.CHROMA_BAND_PX`): no matting model and no pixel forced
    to 0, so the key between a line and its rim fades instead of holing."""
    share, rest = key_split(px, key)
    total = 1.0 - share
    shade = (px * key).sum(axis=2) / max(float((key * key).sum()), 1.0)
    reference = figure & (share <= 0.02) & (rest.max(axis=2) >= delivery_style.CHROMA_REF_MIN)
    if reference.any():
        index = ndimage.distance_transform_edt(
            ~reference, return_distances=False, return_indices=True)
        nearest = px[index[0], index[1]]
        coloured = np.clip((rest * nearest).sum(axis=2)
                           / np.maximum((nearest * nearest).sum(axis=2), 1.0), 0.0, total)
        dark = total - coloured
        core = (coloured >= 0.5) | ((shade <= delivery_style.CHROMA_DARK_LINE) & (share <= 0.5))
        near, far = delivery_style.CHROMA_INK_PX
        ink = np.clip((far - ndimage.distance_transform_edt(~core)) / (far - near), 0.0, 1.0)
        ink = np.maximum(ink, ndimage.binary_dilation(
            free_lines(px, key), _EIGHT, iterations=2))
        cover = coloured + ink * dark
    else:
        cover = total
    noise = delivery_style.CHROMA_NOISE
    cover = np.clip((cover - noise) / (1.0 - noise), 0.0, 1.0)
    width = delivery_style.CHROMA_BAND_PX
    band = (ndimage.binary_dilation(figure, iterations=width)
            & ~ndimage.binary_erosion(figure, iterations=width))
    return np.where(band, cover, figure.astype(float))


def figure_colour(px: np.ndarray, alpha: np.ndarray,
                  foreground: Foreground) -> np.ndarray:
    """The figure's colour under `alpha`: estimated where alpha > 0 (unmixed
    from the key where it shows through) and despilled of the raw key on the
    soft edge, the raw pixel where alpha is 0. On a green key a soft pixel's
    colour is what taking its key share out leaves; on white it is the raw
    pixel, but on the antialias ramp (`drawn_alpha`) the ink it climbs to."""
    if white_backdrop(px):
        deficit = (_PAPER - px).max(axis=2)
        ramp = ((alpha > 0) & (alpha < 1)
                & (deficit >= delivery_style.WHITE_PAPER_DEFICIT))
        return ridge_colour(px, ramp)
    colour = np.clip(foreground(px / 255.0, alpha), 0.0, 1.0) * 255.0
    solved = unmixed(px, alpha)
    colour = np.where(np.isnan(solved), colour, solved)
    edge = (alpha > 0) & ~ndimage.binary_erosion(alpha >= 1.0)
    colour = delivery.despill(colour, ndimage.binary_dilation(edge, iterations=2),
                              delivery._corner_seed(px))
    key = green_key(px)
    if key is not None:
        _, rest = key_split(px, key)
        keyed = np.clip(rest / np.maximum(alpha, 0.05)[..., None], 0.0, 255.0)
        colour = np.where(((alpha > 0) & (alpha < 1))[..., None], keyed, colour)
    return np.where((alpha > 0)[..., None], colour, px)


def _png(array: np.ndarray, mode: str | None = None) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode).save(
        output, "PNG")
    return output.getvalue()


def own_alpha(data: bytes, size: tuple[int, int] | None = None) -> bytes | None:
    """The alpha channel of the PNG `data` as an L PNG at `size`, or None
    when the picture carries no alpha of its own: only an alpha-carrying
    picture's cut can take its alpha from the picture itself."""
    try:
        image = Image.open(io.BytesIO(data))
        if "A" not in image.getbands():
            return None
        alpha = image.getchannel("A")
    except Exception:
        return None
    if alpha.getextrema()[0] == 255:
        return None
    if size is not None and alpha.size != size:
        alpha = alpha.resize(size, Image.BICUBIC)
    output = io.BytesIO()
    alpha.save(output, "PNG")
    return output.getvalue()


def known_regions(px: np.ndarray, soft: np.ndarray) -> np.ndarray:
    """The trimap of the retraced matte, with the key the figure encloses
    known backdrop: a gap narrower than the unknown band is otherwise left
    to the model, which reads it as figure."""
    figure = delivery.cut_figure(px, soft)
    known = trimap(figure, delivery_style.MATTING_TRIMAP_PX)
    known[delivery.key_holes(px, figure, delivery_style.MATTE_EDGE_TOLERANCE)] = (
        KNOWN_BACKDROP)
    return known


def alpha_png(data: bytes, matte: bytes, predict: Predict) -> bytes:
    """The alpha (L PNG) of the figure in `data`, from the matte model's soft
    output `matte`: the drawing on a white backdrop (`drawn_alpha`), keyed on its
    edge (`chroma_edge`) on a green key, matted by `predict` otherwise."""
    px = np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(float)
    soft = np.array(Image.open(io.BytesIO(matte)).convert("L"))
    if white_backdrop(px):
        return _png(drawn_alpha(px, soft / 255.0) * 255, "L")
    key = green_key(px)
    if key is not None:
        figure = delivery.cut_figure(px, soft)
        known = trimap(figure, delivery_style.MATTING_TRIMAP_PX)
        return _png(see_through(px, chroma_edge(px, figure, key), known) * 255, "L")
    known = known_regions(px, soft)
    alpha = tiled(predict, px.astype(np.uint8), known,
                  delivery_style.MATTING_TILE_PX,
                  delivery_style.MATTING_TILE_OVERLAP_PX)
    return _png(pocket_key(px, see_through(px, settle(alpha, known), known))
                * 255, "L")


def clearer_png(data: bytes, alpha: bytes, gain: float) -> bytes:
    """`clearer` over the alpha (L PNG) of the figure in `data`."""
    px = np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(float)
    solved = np.array(Image.open(io.BytesIO(alpha)).convert("L")).astype(float) / 255.0
    return _png(clearer(px, solved, gain) * 255, "L")


def cutout_png(foreground: bytes, alpha: bytes) -> bytes:
    """The figure's colour (RGB PNG) with the alpha (L PNG) as a straight
    RGBA PNG."""
    rgba = Image.open(io.BytesIO(foreground)).convert("RGB")
    rgba.putalpha(Image.open(io.BytesIO(alpha)).convert("L"))
    output = io.BytesIO()
    rgba.save(output, "PNG")
    return output.getvalue()


def foreground_png(data: bytes, alpha: bytes, foreground: Foreground) -> bytes:
    """The figure's colour (RGB PNG) in `data` under the alpha (L PNG)."""
    px = np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(float)
    solved = np.array(Image.open(io.BytesIO(alpha)).convert("L")).astype(float) / 255.0
    return _png(figure_colour(px, solved, foreground))
