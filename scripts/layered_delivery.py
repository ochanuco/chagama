#!/usr/bin/env python3
"""Re-layer a delivery whose cut kept furniture: the furniture behind the
character as backdrop, the object in front of her as part of her.

The figure layer is split by hue on the delivery before saturation, where
the furniture's navy and the character's purples sit apart; thin dark lines
go to the side they border. The bands follow the outer contour of the
character and the front object, over the furniture too; holes the character
encloses get none, and gaps narrower than the bands are filled white.

    uv run scripts/layered_delivery.py <delivered> <before-saturation> <out.png>
"""
import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from chagama.infrastructure.chimera.client import ChimeraClient
from chagama.infrastructure.imaging import delivery

OUTLINES = [{"color": "#ffffff", "width": 0.8}, {"color": "#885b80", "width": 1.04}]
OBJECT_HUE = (135, 172)          # PIL 0..255
FRONT_MAX_S, FRONT_MIN_V = 168, 70
EDGE_PX = 6

CHARA, BEHIND, FRONT = 1, 2, 3


def fetch_figure(client: ChimeraClient, short_id: str) -> np.ndarray:
    headers = {**client.credentials(), "User-Agent": "chagama-layered"}
    url = f"{client.base_url}/g/{short_id}/assets/layer-figure"
    data = urllib.request.urlopen(urllib.request.Request(url, headers=headers)).read()
    return np.array(Image.open(io.BytesIO(data)).convert("RGBA"))


def _drop_small(labels: np.ndarray, label: int, share: float) -> None:
    mask = labels == label
    parts, count = ndimage.label(mask)
    sizes = ndimage.sum(mask, parts, range(1, count + 1))
    small = np.isin(parts, 1 + np.flatnonzero(sizes < mask.size * share))
    if small.any():
        others = np.where(small, 0, labels)
        iy, ix = ndimage.distance_transform_edt(
            others == 0, return_distances=False, return_indices=True)
        labels[small] = others[iy[small], ix[small]]


def classify(pre: np.ndarray) -> np.ndarray:
    inside = pre[..., 3] > 12
    hsv = np.array(Image.fromarray(pre[..., :3]).convert("HSV")).astype(float)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    objects = (hue >= OBJECT_HUE[0]) & (hue <= OBJECT_HUE[1]) & (sat > 70) & (val > 22) & inside
    dark = inside & (val < 80) & ~objects
    line = (inside & (val < 40) & ~objects) | (dark & ~ndimage.binary_opening(dark, iterations=3))
    chara = inside & ~objects & ~line
    objects = ndimage.binary_opening(objects, iterations=1)

    front = objects & (sat < FRONT_MAX_S) & (val > FRONT_MIN_V)
    parts, count = ndimage.label(ndimage.binary_closing(front, iterations=3))
    sizes = ndimage.sum(front, parts, range(1, count + 1))
    front = (parts == 1 + int(np.argmax(sizes))) & objects
    front = ndimage.binary_fill_holes(front) & inside & ~chara

    labels = np.zeros(inside.shape, np.int8)
    labels[chara], labels[objects & ~front], labels[front] = CHARA, BEHIND, FRONT
    nearest = np.stack([ndimage.distance_transform_edt(labels != k) - (0.5 if k == CHARA else 0)
                        for k in (CHARA, BEHIND, FRONT)])
    labels = np.where(inside & (labels == 0), 1 + np.argmin(nearest, axis=0), labels)
    labels[~inside] = 0
    for label in (BEHIND, FRONT, CHARA):
        _drop_small(labels, label, 0.0004)
    for k in range(EDGE_PX):
        labels[k, :] = np.where(labels[k, :] > 0, labels[EDGE_PX, :], 0)
        labels[-1 - k, :] = np.where(labels[-1 - k, :] > 0, labels[-1 - EDGE_PX, :], 0)
        labels[:, k] = np.where(labels[:, k] > 0, labels[:, EDGE_PX], 0)
        labels[:, -1 - k] = np.where(labels[:, -1 - k] > 0, labels[:, -1 - EDGE_PX], 0)
    return labels


def _soft(mask: np.ndarray) -> np.ndarray:
    soft = ndimage.gaussian_filter(mask.astype(float), 0.7)
    soft[ndimage.binary_erosion(mask)] = 1.0
    return soft


def _over(dst_rgb, dst_a, src_rgb, src_a):
    alpha = src_a + dst_a * (1 - src_a)
    rgb = (src_rgb * src_a[..., None] + dst_rgb * (dst_a * (1 - src_a))[..., None]) \
        / np.maximum(alpha, 1e-6)[..., None]
    return rgb, alpha


def compose(figure: np.ndarray, labels: np.ndarray, light: str = "n") -> np.ndarray:
    px, fa = figure[..., :3].astype(float), figure[..., 3] / 255.0
    chara = (labels == CHARA) | (labels == FRONT)
    behind = labels == BEHIND

    r = sum(delivery._band_widths(*chara.shape, OUTLINES))
    pad = int(r) + 4
    padded = np.pad(chara, pad)
    closed = ndimage.distance_transform_edt(ndimage.distance_transform_edt(~padded) <= r) > r
    closed |= padded
    outer = ndimage.binary_fill_holes(closed)[pad:-pad, pad:-pad]
    closed = closed[pad:-pad, pad:-pad]
    gaps = closed & ~chara & ~behind & (fa < 0.5)

    colours = delivery._band_colours(OUTLINES)
    under = behind | (chara & ndimage.binary_dilation(behind, iterations=3))
    rgb, alpha = px.copy(), np.where(under, fa, 0.0)
    rgb, alpha = _over(rgb, alpha, np.broadcast_to(colours[0], px.shape), gaps.astype(float))
    for band, colour in zip(reversed(delivery.band_alphas(outer, OUTLINES, light)), reversed(colours)):
        rgb, alpha = _over(rgb, alpha, np.broadcast_to(colour, px.shape), np.where(outer, 0.0, band))
    rgb, alpha = _over(rgb, alpha, px, fa * _soft(chara))
    return np.dstack([rgb.clip(0, 255), (alpha * 255).clip(0, 255)]).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("delivered", help="short_id of the delivery to re-layer")
    parser.add_argument("before", help="short_id of the same delivery before saturation")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    client = ChimeraClient(Path.cwd())
    labels = classify(fetch_figure(client, args.before))
    Image.fromarray(compose(fetch_figure(client, args.delivered), labels), "RGBA").save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
