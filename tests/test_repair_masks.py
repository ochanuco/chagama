"""Tests for rendering a repair mask PNG from circles/rects."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from chagama.domain.repair.regions import Circle, Rect
from chagama.infrastructure.imaging.masks import (
    mask_bbox_fraction,
    render_mask_png,
    render_soft_mask_png,
)


class RenderMaskPngTest(unittest.TestCase):
    def test_shape_and_mode(self):
        data = render_mask_png(20, 10, [], [])
        image = Image.open(io.BytesIO(data))
        self.assertEqual(image.size, (20, 10))
        self.assertEqual(image.mode, "RGB")

    def test_empty_shapes_yields_an_all_black_mask(self):
        data = render_mask_png(20, 10, [], [])
        pixels = np.array(Image.open(io.BytesIO(data)))
        self.assertTrue((pixels == 0).all())

    def test_a_rect_paints_white_inside_and_black_outside(self):
        data = render_mask_png(20, 20, [], [Rect(5, 5, 10, 10)])
        pixels = np.array(Image.open(io.BytesIO(data)))
        self.assertTrue((pixels[7, 7] == 255).all())
        self.assertTrue((pixels[0, 0] == 0).all())

    def test_a_circle_paints_white_at_its_centre(self):
        data = render_mask_png(40, 40, [Circle(cx=20, cy=20, r=5)], [])
        pixels = np.array(Image.open(io.BytesIO(data)))
        self.assertTrue((pixels[20, 20] == 255).all())
        self.assertTrue((pixels[0, 0] == 0).all())

    def test_a_circle_outside_the_radius_stays_black(self):
        data = render_mask_png(40, 40, [Circle(cx=20, cy=20, r=5)], [])
        pixels = np.array(Image.open(io.BytesIO(data)))
        self.assertTrue((pixels[39, 39] == 0).all())

    def test_shapes_off_canvas_are_clipped_without_raising(self):
        data = render_mask_png(10, 10, [Circle(cx=-5, cy=-5, r=3)],
                               [Rect(8, 8, 20, 20)])
        image = Image.open(io.BytesIO(data))
        self.assertEqual(image.size, (10, 10))


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


class MaskBboxFractionTest(unittest.TestCase):
    def test_no_shapes_yields_zero_bbox(self):
        self.assertEqual(mask_bbox_fraction(100, 100, [], []), (0.0, 0.0, 0.0, 0.0))

    def test_a_circle_bbox_is_its_bounding_square_as_fractions(self):
        bbox = mask_bbox_fraction(100, 200, [Circle(cx=50, cy=100, r=10)], [])
        self.assertAlmostEqual(bbox[0], 0.40)
        self.assertAlmostEqual(bbox[1], 0.45)
        self.assertAlmostEqual(bbox[2], 0.60)
        self.assertAlmostEqual(bbox[3], 0.55)

    def test_bbox_is_clipped_to_the_canvas(self):
        bbox = mask_bbox_fraction(100, 100, [Circle(cx=0, cy=0, r=50)], [])
        self.assertEqual(bbox[0], 0.0)
        self.assertEqual(bbox[1], 0.0)

    def test_union_of_a_circle_and_a_rect(self):
        bbox = mask_bbox_fraction(
            100, 100, [Circle(cx=10, cy=10, r=5)], [Rect(50, 50, 90, 90)])
        self.assertAlmostEqual(bbox[0], 0.05)
        self.assertAlmostEqual(bbox[1], 0.05)
        self.assertAlmostEqual(bbox[2], 0.90)
        self.assertAlmostEqual(bbox[3], 0.90)


if __name__ == "__main__":
    unittest.main()
