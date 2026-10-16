"""Tests for the saturation imaging: repin's curve on a layer through its own
alpha, and the layers put back together."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from chagama.infrastructure.imaging.palette import repin
from chagama.infrastructure.imaging.saturation import (
    recompose,
    saturate_layer,
    saturation_pngs,
)

OUTLINE_PURPLE = (0x88, 0x5B, 0x80)


def mean_saturation(rgb: np.ndarray) -> float:
    return float(np.array(Image.fromarray(rgb.astype(np.uint8)).convert("HSV"))[..., 1].mean())


def hsv_rgb(hue: int, saturation: int, value: int) -> tuple[int, int, int]:
    pixel = np.array([[[hue, saturation, value]]], dtype=np.uint8)
    return tuple(int(c) for c in np.array(Image.fromarray(pixel, "HSV").convert("RGB"))[0, 0])


def layer(colour, *, covered=(slice(16, 48), slice(16, 48)), alpha=255) -> np.ndarray:
    """64x64 layer: `colour` where `covered`, with that alpha, else transparent
    pixels that still carry a vivid colour."""
    pixels = np.zeros((64, 64, 4), dtype=np.float32)
    pixels[..., :3] = hsv_rgb(191, 200, 200)
    pixels[covered] = (*colour, alpha)
    return pixels


def png(array: np.ndarray, mode: str) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode).save(output, "PNG")
    return output.getvalue()


class RepinFigureTest(unittest.TestCase):
    def test_a_given_range_replaces_the_estimated_one(self):
        pixels = np.full((64, 64, 3), hsv_rgb(191, 120, 200), dtype=np.uint8)
        only_left = np.zeros((64, 64), dtype=bool)
        only_left[:, :32] = True
        rgb, _ = repin(pixels, figure=only_left)
        self.assertLess(mean_saturation(rgb[:, :32]), mean_saturation(pixels[:, :32]) - 20)
        self.assertAlmostEqual(mean_saturation(rgb[:, 32:]), mean_saturation(pixels[:, 32:]),
                               delta=2)

    def test_strength_scales_the_correction(self):
        pixels = np.full((64, 64, 3), hsv_rgb(191, 120, 200), dtype=np.uint8)
        everywhere = np.ones((64, 64), dtype=bool)
        full = mean_saturation(repin(pixels, figure=everywhere)[0])
        half = mean_saturation(repin(pixels, figure=everywhere, strength=0.5)[0])
        self.assertLess(full, half)
        self.assertLess(half, mean_saturation(pixels))


class SaturateLayerTest(unittest.TestCase):
    def test_the_curve_is_applied_only_where_alpha_is_above_zero(self):
        source = layer(hsv_rgb(191, 120, 200))
        out = saturate_layer(source, 1.0)
        covered = source[..., 3] > 0
        self.assertLess(mean_saturation(out[covered][:, :3].reshape(1, -1, 3)),
                        mean_saturation(source[covered][:, :3].reshape(1, -1, 3)) - 20)
        np.testing.assert_array_equal(out[~covered], source[~covered])
        np.testing.assert_array_equal(out[..., 3], source[..., 3])

    def test_a_translucent_edge_counts_as_covered(self):
        source = layer(hsv_rgb(191, 120, 200), alpha=1)
        out = saturate_layer(source, 1.0)
        self.assertFalse(np.array_equal(out[32, 32], source[32, 32]))

    def test_strength_zero_is_the_identity(self):
        source = layer(hsv_rgb(191, 120, 200))
        np.testing.assert_array_equal(saturate_layer(source, 0.0), source)

    def test_the_outline_purple_gets_less_saturated(self):
        source = layer(OUTLINE_PURPLE)
        out = saturate_layer(source, 1.0)
        before = mean_saturation(np.array([[source[32, 32, :3]]]))
        after = mean_saturation(np.array([[out[32, 32, :3]]]))
        self.assertLess(after, before - 10)

    def test_a_layer_with_no_coverage_is_returned_as_it_is(self):
        source = layer((0, 0, 0), alpha=0)
        np.testing.assert_array_equal(saturate_layer(source, 1.0), source)


class RecomposeTest(unittest.TestCase):
    def test_a_backdrop_gets_the_outline_then_the_figure(self):
        backdrop = np.full((64, 64, 3), 10.0, dtype=np.float32)
        figure = layer((200, 0, 0), covered=(slice(0, 32), slice(0, 32)))
        outline = layer((0, 200, 0), covered=(slice(0, 64), slice(0, 64)))
        picture = recompose(figure, outline, backdrop)
        self.assertEqual(picture.shape, (64, 64, 3))
        np.testing.assert_allclose(picture[5, 5], (200, 0, 0))
        np.testing.assert_allclose(picture[50, 50], (0, 200, 0))

    def test_without_a_backdrop_the_picture_is_rgba(self):
        figure = layer((200, 0, 0), covered=(slice(0, 32), slice(0, 32)))
        outline = layer((0, 200, 0), covered=(slice(0, 40), slice(0, 40)))
        picture = recompose(figure, outline, None)
        self.assertEqual(picture.shape, (64, 64, 4))
        np.testing.assert_allclose(picture[5, 5], (200, 0, 0, 255), atol=1)
        np.testing.assert_allclose(picture[36, 36], (0, 200, 0, 255), atol=1)
        self.assertEqual(picture[60, 60, 3], 0)


class SaturationPngsTest(unittest.TestCase):
    def test_the_picture_is_the_compressed_layers_over_the_backdrop(self):
        figure = layer(hsv_rgb(191, 120, 200), covered=(slice(16, 48), slice(16, 48)))
        outline = layer(OUTLINE_PURPLE, covered=(slice(8, 56), slice(8, 56)))
        backdrop = np.full((64, 64, 3), 128, dtype=np.uint8)
        picture, figure_png, outline_png = saturation_pngs(
            png(figure, "RGBA"), png(outline, "RGBA"), png(backdrop, "RGB"), strength=1.0)
        shown = np.array(Image.open(io.BytesIO(picture)))
        self.assertEqual(shown.shape, (64, 64, 3))
        np.testing.assert_array_equal(shown[0, 0], backdrop[0, 0])
        saturated_figure = np.array(Image.open(io.BytesIO(figure_png)))
        np.testing.assert_array_equal(shown[32, 32], saturated_figure[32, 32, :3])
        self.assertLess(mean_saturation(shown[32:33, 32:33]),
                        mean_saturation(figure[32:33, 32:33, :3]))
        self.assertEqual(np.array(Image.open(io.BytesIO(outline_png))).shape, (64, 64, 4))

    def test_a_transparent_delivery_comes_back_rgba(self):
        figure = layer(hsv_rgb(191, 120, 200))
        outline = layer(OUTLINE_PURPLE, covered=(slice(8, 56), slice(8, 56)))
        picture, *_ = saturation_pngs(png(figure, "RGBA"), png(outline, "RGBA"), None,
                                      strength=1.0)
        shown = Image.open(io.BytesIO(picture))
        self.assertEqual(shown.mode, "RGBA")
        self.assertEqual(np.array(shown)[0, 0, 3], 0)


if __name__ == "__main__":
    unittest.main()
