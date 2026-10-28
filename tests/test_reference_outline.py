from __future__ import annotations

import base64
import io
import unittest

import numpy as np
from PIL import Image

from chagama.infrastructure.reference import outline


class Segmenter:
    def run(self, outputs, feeds):
        mask = np.zeros((1, 1, 1024, 1024), np.float32)
        mask[..., 200:800, 300:700] = 1.0
        return [mask]


def near_left(image):
    width, height = image.size
    return np.tile(np.linspace(1.0, 0.0, width, dtype=np.float32), (height, 1))


def decode(data):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data))))


class OutlineTest(unittest.TestCase):
    def picture(self):
        rgb = np.full((400, 200, 3), 255, np.uint8)
        rgb[100:300, 50:150] = 40
        rgb[360:380, 10:30] = 0
        return Image.fromarray(rgb)

    def test_the_silhouette_is_the_figure_in_black(self):
        found = outline.outlines(self.picture(), Segmenter(), near_left)
        silhouette = decode(found["silhouette"])
        self.assertEqual(silhouette.shape, (400, 200))
        self.assertEqual(silhouette[200, 100], 0)
        self.assertEqual(silhouette[10, 10], 255)

    def test_the_lineart_draws_edges_inside_the_figure_only(self):
        lines = decode(outline.outlines(self.picture(), Segmenter(), near_left)["lineart"])
        self.assertLess(lines[200, 49:52].min(), 128)
        self.assertEqual(lines[200, 100], 255)
        self.assertEqual(lines[355:385, 5:35].min(), 255)

    def test_a_large_picture_is_shrunk(self):
        big = Image.new("RGB", (1600, 800), "white")
        self.assertEqual(decode(outline.outlines(big, Segmenter(), near_left)["silhouette"]).shape,
                         (384, 768))

    def test_the_depth_is_white_near_inside_the_figure_and_black_outside(self):
        depth = decode(outline.outlines(self.picture(), Segmenter(), near_left)["depth"])
        self.assertEqual(depth.shape, (400, 200))
        self.assertEqual(depth[10, 100], 0)
        self.assertGreater(depth[200, 30], depth[200, 160])
        self.assertGreaterEqual(depth[200, 160], round(outline.DEPTH_RANGE[0] * 255) - 1)
