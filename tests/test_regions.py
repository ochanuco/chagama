"""Tests for caller-given redraw region geometry."""

from __future__ import annotations

import unittest

from chagama.domain.regions import Rect, rects_from_fractions


class RectsFromFractionsTest(unittest.TestCase):
    def test_fractions_scale_by_width_and_height(self):
        rects = rects_from_fractions([[0.1, 0.2, 0.3, 0.4]], 800, 1000)
        self.assertEqual(rects, [Rect(80.0, 200.0, 240.0, 400.0)])

    def test_empty_regions_yields_no_rects(self):
        self.assertEqual(rects_from_fractions([], 800, 1000), [])

    def test_multiple_regions_preserve_order(self):
        rects = rects_from_fractions(
            [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]], 100, 100)
        self.assertEqual(rects, [Rect(0, 0, 50, 50), Rect(50, 50, 100, 100)])
