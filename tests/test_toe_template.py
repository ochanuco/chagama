"""Tests for the repair reroll's synthetic ControlNet reference hint."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image
from scipy import ndimage

from chagama.infrastructure.imaging.toe_template import reference_hint


class ReferenceHintTest(unittest.TestCase):
    def test_shape_and_mode(self):
        data = reference_hint(512)
        image = Image.open(io.BytesIO(data))
        self.assertEqual(image.size, (512, 512))
        self.assertEqual(image.mode, "RGB")

    def test_deterministic(self):
        self.assertEqual(reference_hint(256), reference_hint(256))

    def test_scales_with_size(self):
        small = Image.open(io.BytesIO(reference_hint(256)))
        large = Image.open(io.BytesIO(reference_hint(512)))
        self.assertEqual(small.size, (256, 256))
        self.assertEqual(large.size, (512, 512))

    def test_draws_five_separate_toe_outlines(self):
        pixels = np.array(Image.open(io.BytesIO(reference_hint(512))).convert("L"))
        dark = pixels < 128
        self.assertTrue(dark.any())
        _labels, count = ndimage.label(dark, structure=np.ones((3, 3)))
        self.assertEqual(count, 5)


if __name__ == "__main__":
    unittest.main()
