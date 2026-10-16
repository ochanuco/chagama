from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from chagama.domain.yukari.delivery_style import (
    ACCENT_KEEP, REPIN_DARK, REPIN_LIGHT,
)
from chagama.infrastructure.imaging.palette import (
    measure, repin, summarize,
)

KNEE, RATIO = REPIN_LIGHT


def compressed(saturation: float) -> float:
    """The light-band curve for a field pixel (below the accent ramp)."""
    return KNEE + (saturation - KNEE) * RATIO


def png(pixels: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(output, "PNG")
    return output.getvalue()


def swatch(hsv_center: tuple[int, int, int]) -> np.ndarray:
    """64x64 flat #808080 backdrop, a 32x32 centre block at the given HSV."""
    pixels = np.full((64, 64, 3), 0x80, dtype=np.uint8)
    hsv = np.zeros((32, 32, 3), dtype=np.uint8)
    hsv[..., 0], hsv[..., 1], hsv[..., 2] = hsv_center
    block = np.array(Image.fromarray(hsv, "HSV").convert("RGB"))
    pixels[16:48, 16:48] = block
    return pixels


class RepinTest(unittest.TestCase):
    def test_field_saturation_purple_is_compressed_toward_the_knee(self):
        # S 120 is field grade -- vivid cloth, below the accent ramp -- so
        # only the excess over the knee survives, at the band's ratio.
        pixels = swatch((191, 120, 200))
        rgb, report = repin(pixels)
        hsv = np.array(Image.fromarray(rgb).convert("HSV")).astype(float)
        center = hsv[16:48, 16:48]
        self.assertAlmostEqual(center[..., 1].mean(), compressed(120), delta=8)
        self.assertAlmostEqual(center[..., 2].mean(), 200, delta=2)
        backdrop = rgb[:8, :8]
        self.assertTrue(np.all(backdrop == 0x80))
        self.assertTrue(any("compressed" in line for line in report))

    def test_accent_saturation_keeps_most_of_its_excess(self):
        # Iris-grade saturation sits above the accent ramp and keeps
        # ACCENT_KEEP of its excess instead of the field ratio.
        pixels = swatch((191, 230, 200))
        rgb, _ = repin(pixels)
        hsv = np.array(Image.fromarray(rgb).convert("HSV")).astype(float)
        s = hsv[16:48, 16:48, 1].mean()
        self.assertGreater(s, 140)
        self.assertAlmostEqual(s, KNEE + (230 - KNEE) * ACCENT_KEEP, delta=10)

    def test_a_dark_field_is_not_mistaken_for_an_accent(self):
        # The black hoodie and tights reach iris-grade saturation on some
        # renders. Saturation alone would read the whole garment as an
        # accent and preserve it, delivering a navy hoodie; an accent is
        # bright as well as saturated, so the dark band pins pale.
        pixels = swatch((191, 240, 50))
        rgb, _ = repin(pixels)
        hsv = np.array(Image.fromarray(rgb).convert("HSV")).astype(float)
        s = hsv[16:48, 16:48, 1].mean()
        knee, ratio = REPIN_DARK
        self.assertAlmostEqual(s, knee + (240 - knee) * ratio, delta=8)
        self.assertLess(s, 60)

    def test_pale_purple_is_left_nearly_unchanged(self):
        pixels = swatch((191, 30, 200))
        rgb, _ = repin(pixels)
        hsv_before = np.array(Image.fromarray(pixels).convert("HSV")).astype(float)
        hsv_after = np.array(Image.fromarray(rgb).convert("HSV")).astype(float)
        before_s = hsv_before[16:48, 16:48, 1].mean()
        after_s = hsv_after[16:48, 16:48, 1].mean()
        # Scaling only ever goes down, and a render already paler than the
        # palette target should not move much.
        self.assertLessEqual(after_s, before_s + 1)
        self.assertGreater(after_s, before_s - 3)


class SummarizeTest(unittest.TestCase):
    def test_high_saturation_render_fails(self):
        data = png(swatch((191, 200, 200)))
        summary = summarize(data)
        self.assertTrue(summary["fails"])

    def test_low_saturation_render_passes_figure_bands(self):
        data = png(swatch((191, 30, 200)))
        summary = summarize(data)
        figure_fails = [f for f in summary["fails"] if "figure midtone" in f]
        self.assertEqual(figure_fails, [])


class RGBAMeasureTest(unittest.TestCase):
    def test_garbage_under_alpha_zero_measures_as_the_flat_backdrop(self):
        # A plain low-saturation figure on a transparent field whose RGB
        # channels, ignored by every renderer, hold garbage that would spike
        # corner_spread if it leaked into the measurement uncomposited.
        pixels = swatch((191, 30, 200))
        alpha = np.full((64, 64), 255, dtype=np.uint8)
        alpha[:16, :16] = 0
        alpha[:16, -16:] = 0
        alpha[-16:, :16] = 0
        alpha[-16:, -16:] = 0
        garbage = pixels.copy()
        garbage[:16, :16] = (255, 0, 0)
        garbage[:16, -16:] = (0, 255, 0)
        garbage[-16:, :16] = (0, 0, 255)
        garbage[-16:, -16:] = (255, 255, 0)
        rgba = np.dstack([garbage, alpha])
        output = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(output, "PNG")
        summary = summarize(output.getvalue())
        self.assertLess(summary["corner_spread"], 1.0)
        figure_fails = [f for f in summary["fails"] if "figure midtone" in f]
        self.assertEqual(figure_fails, [])

    def test_measure_composites_onto_the_flat_grey_backdrop(self):
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[..., :3] = (10, 200, 10)
        rgba[..., 3] = 0
        output = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(output, "PNG")
        summary = measure(output.getvalue())
        self.assertEqual(summary["bg"], (200, 200, 200))


if __name__ == "__main__":
    unittest.main()
