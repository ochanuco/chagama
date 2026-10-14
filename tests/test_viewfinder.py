"""Viewfinder overlay: size, alpha handling and the focus frame."""

from __future__ import annotations

import io
import unittest

from PIL import Image

from chagama.infrastructure.imaging.viewfinder import FOCUS, viewfinder_png


def png(mode: str, size: tuple[int, int], colour) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, colour).save(buffer, "PNG")
    return buffer.getvalue()


class ViewfinderTest(unittest.TestCase):
    def test_rgb_in_rgb_out_at_the_same_size(self):
        out = Image.open(io.BytesIO(
            viewfinder_png(png("RGB", (640, 800), (30, 30, 30)), (0.5, 0.4), 2.8)))
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.size, (640, 800))

    def test_alpha_is_kept(self):
        out = Image.open(io.BytesIO(viewfinder_png(
            png("RGBA", (640, 800), (30, 30, 30, 0)), (0.5, 0.4), 2.8)))
        self.assertEqual(out.mode, "RGBA")
        self.assertEqual(out.getpixel((10, 10))[3], 0)

    def test_focus_bracket_corner_is_the_focus_colour(self):
        width, height = 1280, 1280
        out = Image.open(io.BytesIO(viewfinder_png(
            png("RGB", (width, height), (30, 30, 30)), (0.5, 0.25), 2.8)))
        corner = (width // 2 - 40, round(height * 0.25) - 40)
        self.assertEqual(out.getpixel(corner), FOCUS[:3])


if __name__ == "__main__":
    unittest.main()
