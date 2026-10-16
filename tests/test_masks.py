"""Tests for rendering a soft redraw mask PNG from rects."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from chagama.domain.regions import Rect
from chagama.infrastructure.imaging.masks import render_soft_mask_png


class RenderSoftMaskPngTest(unittest.TestCase):
    def test_shape_and_mode(self):
        data = render_soft_mask_png(20, 10, [], 0.25, 0)
        image = Image.open(io.BytesIO(data))
        self.assertEqual(image.size, (20, 10))
        self.assertEqual(image.mode, "RGB")

    def test_no_rects_is_all_white(self):
        data = render_soft_mask_png(20, 10, [], 0.25, 0)
        pixels = np.array(Image.open(io.BytesIO(data)))
        self.assertTrue((pixels == 255).all())

    def test_a_rect_paints_the_inside_level_with_no_feather(self):
        data = render_soft_mask_png(40, 40, [Rect(10, 10, 30, 30)], 0.25, 0)
        pixels = np.array(Image.open(io.BytesIO(data)))
        self.assertTrue((pixels[20, 20] == round(0.25 * 255)).all())
        self.assertTrue((pixels[0, 0] == 255).all())

    def test_inside_level_tracks_the_strength_argument(self):
        low = np.array(Image.open(io.BytesIO(
            render_soft_mask_png(40, 40, [Rect(10, 10, 30, 30)], 0.1, 0))))
        high = np.array(Image.open(io.BytesIO(
            render_soft_mask_png(40, 40, [Rect(10, 10, 30, 30)], 0.9, 0))))
        self.assertLess(low[20, 20, 0], high[20, 20, 0])

    def test_feather_ramps_monotonically_out_from_the_rect(self):
        data = render_soft_mask_png(80, 80, [Rect(20, 30, 60, 50)], 0.25, 8)
        pixels = np.array(Image.open(io.BytesIO(data)))
        # Sample outward from the rect's centre along its own row: strictly
        # inside stays at the (feathered) inside level, then the value rises
        # monotonically back toward 255 as the sample crosses the rect edge.
        row = pixels[40, 40:80, 0].astype(int)
        self.assertTrue(all(a <= b for a, b in zip(row, row[1:])))
        self.assertGreaterEqual(row[-1], 254)

    def test_feather_zero_leaves_a_hard_edge(self):
        data = render_soft_mask_png(40, 40, [Rect(10, 10, 30, 30)], 0.25, 0)
        pixels = np.array(Image.open(io.BytesIO(data)))
        self.assertEqual(pixels[20, 9, 0], 255)
        self.assertEqual(pixels[20, 10, 0], round(0.25 * 255))
