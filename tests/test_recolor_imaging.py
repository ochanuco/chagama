from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from chagama.domain.yukari.delivery_style import RECOLOR_TARGETS
from chagama.infrastructure.imaging.recolor import (
    lines, recolor, recolor_png,
)

HOODIE = (10, 200, 90)   # dark and saturated -- either the hoodie or legwear


def png(pixels: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(output, "PNG")
    return output.getvalue()


def block(hsv_center: tuple[int, int, int], shape: tuple[int, int]) -> np.ndarray:
    hsv = np.zeros(shape + (3,), dtype=np.uint8)
    hsv[..., 0], hsv[..., 1], hsv[..., 2] = hsv_center
    return np.array(Image.fromarray(hsv, "HSV").convert("RGB"))


def swatch(hsv_center: tuple[int, int, int]) -> np.ndarray:
    """64x64 flat #808080 backdrop, a 32x32 centre block at the given HSV."""
    pixels = np.full((64, 64, 3), 0x80, dtype=np.uint8)
    pixels[16:48, 16:48] = block(hsv_center, (32, 32))
    return pixels


def figure(fill_rows: tuple[int, int]) -> np.ndarray:
    """101x80 backdrop with a full-height spine, so the figure's bounding
    box spans the whole canvas and a fill's row range sets its own `cy`.

    `fill_rows` is padded by 4px of its own colour on every side (more than
    the line window's 3px reach) so the fill/backdrop boundary -- which
    reads as relief, not as the render's own linework -- never erodes the
    32x20 core a test actually samples.
    """
    pixels = np.full((101, 80, 3), 0x80, dtype=np.uint8)
    pixels[:, 10:14] = block((100, 60, 150), (101, 4))
    top, bottom = fill_rows
    r0, r1 = max(top - 4, 0), min(bottom + 4, 101)
    pixels[r0:r1, 26:54] = block(HOODIE, (r1 - r0, 28))
    return pixels


class RecolorTest(unittest.TestCase):
    def test_dark_saturated_fill_high_in_figure_becomes_hoodie(self):
        pixels = figure((10, 30))
        out, report = recolor(pixels)
        hsv = np.array(Image.fromarray(out).convert("HSV")).astype(float)
        core = hsv[10:30, 30:50]
        target = RECOLOR_TARGETS["hoodie"]
        self.assertAlmostEqual(np.median(core[..., 0]), target[0], delta=4)
        self.assertAlmostEqual(np.median(core[..., 1]), target[1], delta=4)
        self.assertAlmostEqual(np.median(core[..., 2]), target[2], delta=4)
        self.assertTrue(any("hoodie" in line for line in report))

    def test_same_fill_low_in_figure_lands_on_the_legwear_gradient(self):
        pixels = figure((60, 101))
        out, report = recolor(pixels)
        hsv = np.array(Image.fromarray(out).convert("HSV")).astype(float)
        # cy = 82 / 100 = 0.82, the RECOLOR_LEG_STOPS knee that goes
        # near-black before the purple comes back up toward the foot.
        row = hsv[82, 35:45]
        self.assertAlmostEqual(np.median(row[..., 0]), 0, delta=4)
        self.assertAlmostEqual(np.median(row[..., 1]), 6, delta=4)
        self.assertAlmostEqual(np.median(row[..., 2]), 32, delta=4)
        self.assertTrue(any("tights" in line for line in report))

    def test_a_bright_fill_low_in_figure_is_still_legwear(self):
        # A washed-out render draws the legs brighter than the dress -- one
        # measured pair put them at value 239 and 255 -- so value cannot find
        # them and the fill's size and height have to.
        pixels = np.full((101, 80, 3), 0x80, dtype=np.uint8)
        pixels[:, 10:14] = block((100, 60, 150), (101, 4))
        pixels[56:101, 26:54] = block((190, 84, 239), (45, 28))
        out, report = recolor(pixels)
        hsv = np.array(Image.fromarray(out).convert("HSV")).astype(float)
        row = hsv[82, 35:45]
        self.assertAlmostEqual(np.median(row[..., 1]), 6, delta=6)
        self.assertLess(np.median(row[..., 2]), 80)
        self.assertTrue(any("tights" in line for line in report))

    def test_a_magenta_stroke_is_not_linework(self):
        # Magenta is dark by luminance while its red channel is at full, and
        # some renders draw the creases between fingers that way. Read as
        # line it would ship verbatim over corrected skin.
        pixels = np.full((64, 64, 3), 0x80, dtype=np.uint8)
        pixels[16:48, 16:48] = block((20, 30, 250), (32, 32))
        pixels[24:40, 30:33] = block((230, 240, 200), (16, 3))
        self.assertFalse(lines(pixels)[24:40, 30:33].any())
        out, _ = recolor(pixels)
        stroke = np.array(Image.fromarray(out).convert("HSV")).astype(float)
        self.assertLess(np.median(stroke[24:40, 31, 1]), 80)

    def test_a_region_keeps_its_own_hue_and_saturation_variation(self):
        # Pinning a fill to one hue and one saturation flattens the drawing:
        # the shading of the hair and the blush on the cheek live there, not
        # in value, and asserting the target alone greys the whole figure.
        pixels = np.full((101, 80, 3), 0x80, dtype=np.uint8)
        pixels[:, 10:14] = block((100, 60, 150), (101, 4))
        fill = block((190, 60, 230), (60, 28))
        fill[30:, :] = block((190, 120, 230), (30, 28))
        pixels[20:80, 26:54] = fill
        out, _ = recolor(pixels)
        hsv = np.array(Image.fromarray(out).convert("HSV")).astype(float)
        pale = np.median(hsv[25:45, 30:50, 1])
        deep = np.median(hsv[55:75, 30:50, 1])
        self.assertGreater(deep - pale, 20)

    def test_accent_saturation_keeps_its_own_hue_and_saturation(self):
        pixels = swatch((210, 200, 200))
        out, _ = recolor(pixels)
        hsv = np.array(Image.fromarray(out).convert("HSV")).astype(float)
        center = hsv[16:48, 16:48]
        self.assertAlmostEqual(np.median(center[..., 0]), 210, delta=2)
        self.assertAlmostEqual(np.median(center[..., 1]), 200, delta=2)

    def test_line_pixels_come_through_byte_identical(self):
        pixels = np.full((64, 64, 3), 0x80, dtype=np.uint8)
        pixels[10:54, 10:30] = block((30, 180, 220), (44, 20))
        pixels[10:54, 34:54] = block((190, 150, 230), (44, 20))
        pixels[10:54, 30:34] = np.array([5, 5, 5], dtype=np.uint8)
        line = lines(pixels)
        self.assertGreater(line.sum(), 0)
        out, _ = recolor(pixels)
        self.assertTrue(np.array_equal(out[line], pixels[line]))

    def test_warm_bright_fill_becomes_skin(self):
        pixels = swatch((10, 80, 200))
        out, report = recolor(pixels)
        hsv = np.array(Image.fromarray(out).convert("HSV")).astype(float)
        center = hsv[16:48, 16:48]
        target = RECOLOR_TARGETS["skin"]
        self.assertAlmostEqual(np.median(center[..., 0]), target[0], delta=4)
        self.assertAlmostEqual(np.median(center[..., 1]), target[1], delta=4)
        self.assertAlmostEqual(np.median(center[..., 2]), target[2], delta=4)
        self.assertTrue(any("skin" in line for line in report))

    def test_recolor_png_roundtrips_through_pil(self):
        data = png(swatch((10, 80, 200)))
        out, report = recolor_png(data)
        self.assertIsInstance(out, bytes)
        image = Image.open(io.BytesIO(out))
        self.assertEqual(image.size, (64, 64))
        self.assertTrue(report)


if __name__ == "__main__":
    unittest.main()
