"""The depth node's `resolution` (target short side) for a picture's size."""

from __future__ import annotations

import unittest

from chagama.infrastructure.comfyui.refinement_graph import (
    DEPTH_MAX_PIXELS,
    DEPTH_MIN_RESOLUTION,
    depth_resolution,
)


class DepthResolutionTest(unittest.TestCase):
    def test_a_picture_under_the_cap_keeps_its_short_side(self):
        self.assertEqual(depth_resolution((1448, 1808)), 1448)
        self.assertEqual(depth_resolution((1808, 1448)), 1448)

    def test_a_picture_at_the_cap_is_native(self):
        self.assertEqual(depth_resolution((2048, 2560)), 2048)

    def test_a_small_picture_is_never_upscaled(self):
        self.assertEqual(depth_resolution((300, 500)), 300)

    def test_a_large_landscape_is_shrunk_to_the_cap(self):
        width, height = 6000, 4000
        short = depth_resolution((width, height))
        scaled_long = round(width * short / height)
        self.assertLess(short, 4000)
        self.assertLessEqual(short * scaled_long, DEPTH_MAX_PIXELS)
        self.assertGreater(short * scaled_long, DEPTH_MAX_PIXELS * 0.99)

    def test_a_large_portrait_is_shrunk_the_same_way(self):
        self.assertEqual(depth_resolution((4000, 6000)), depth_resolution((6000, 4000)))

    def test_a_sliver_keeps_the_nodes_minimum(self):
        self.assertEqual(depth_resolution((10, 4000)), DEPTH_MIN_RESOLUTION)


if __name__ == "__main__":
    unittest.main()
