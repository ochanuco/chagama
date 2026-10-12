"""Tests for the lens distortion of a delivered picture."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from comfyui_recipes.infrastructure.imaging import lens


def png(array: np.ndarray, mode: str) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array, mode).save(output, "PNG")
    return output.getvalue()


def decoded(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def dot(size=(96, 64), at=(32, 48), radius=4) -> np.ndarray:
    height, width = size
    pixels = np.full((height, width, 3), 240, np.uint8)
    ys, xs = np.mgrid[:height, :width]
    pixels[np.hypot(xs - at[0], ys - at[1]) <= radius] = (90, 40, 160)
    return pixels


def dark_area(array: np.ndarray) -> int:
    return int((array[..., 0] < 165).sum())


class SourceRadiusTest(unittest.TestCase):
    def test_the_reach_stays_put_and_the_two_lenses_invert_each_other(self):
        radius = np.linspace(0, 1, 11)
        for strength in (0.3, 0.7, 1.0):
            fisheye = lens.source_radius(radius, strength)
            self.assertAlmostEqual(float(fisheye[-1]), 1.0)
            self.assertTrue((fisheye[1:-1] < radius[1:-1]).all())
            self.assertTrue((np.diff(fisheye) > 0).all())
            np.testing.assert_allclose(lens.source_radius(fisheye, -strength), radius,
                                       atol=1e-9)


class SphereTest(unittest.TestCase):
    def test_the_disc_rim_stays_put_and_the_bulge_and_pinch_invert_at_full_strength(self):
        radius = np.linspace(0, 1, 11)
        bulge = lens.sphere_radius(radius, 1.0)
        self.assertAlmostEqual(float(bulge[-1]), 1.0)
        self.assertTrue((bulge[1:-1] < radius[1:-1]).all())
        np.testing.assert_allclose(lens.sphere_radius(bulge, -1.0), radius, atol=1e-9)
        outside = np.array([1.2, 2.0])
        np.testing.assert_allclose(lens.sphere_radius(outside, 0.7), outside)

    def test_the_sphere_swells_its_disc_and_leaves_the_rest(self):
        source = dot()
        source[:6, :6] = (10, 200, 10)
        data, _ = lens.lens_png(png(source, "RGB"), (32 / 63, 48 / 95), 0.9,
                                "sphere", 0.3)
        result = np.asarray(decoded(data))
        self.assertGreater(dark_area(result), 1.3 * dark_area(source))
        np.testing.assert_array_equal(result[:6, :6], source[:6, :6])
        np.testing.assert_array_equal(result[-10:], source[-10:])

    def test_a_sphere_needs_a_radius(self):
        with self.assertRaisesRegex(ValueError, "radius"):
            lens.distort(np.zeros((8, 8, 3), np.float32), (0.5, 0.5), 0.5, "sphere")


class LensTest(unittest.TestCase):
    def test_the_fisheye_swells_what_sits_at_its_centre(self):
        source = dot()
        data, mode = lens.lens_png(png(source, "RGB"), (32 / 63, 48 / 95), 0.8)
        self.assertEqual(mode, "RGB")
        result = np.asarray(decoded(data))
        self.assertEqual(result.shape, source.shape)
        self.assertGreater(dark_area(result), 1.5 * dark_area(source))

    def test_the_inverse_shrinks_what_sits_at_its_centre(self):
        source = dot(radius=10)
        data, _ = lens.lens_png(png(source, "RGB"), (32 / 63, 48 / 95), -0.8)
        self.assertLess(dark_area(np.asarray(decoded(data))), 0.8 * dark_area(source))

    def test_the_fisheye_keeps_the_farthest_corner(self):
        source = np.zeros((96, 64, 3), np.uint8)
        source[-6:, -6:] = 255
        data, _ = lens.lens_png(png(source, "RGB"), (0.0, 0.0), 0.9)
        self.assertEqual(tuple(np.asarray(decoded(data))[-1, -1]), (255, 255, 255))

    def test_the_inverse_samples_only_inside_the_picture(self):
        for center in ((0.5, 0.5), (0.2, 0.9), (1.0, 0.0)):
            width, height = 64, 96
            cx, cy = center[0] * (width - 1), center[1] * (height - 1)
            reach = max(np.hypot(x - cx, y - cy)
                        for x in (0, width - 1) for y in (0, height - 1))
            zoom = lens._zoom(width, height, cx, cy, reach, -1.0)
            ys, xs = np.mgrid[:height, :width].astype(np.float64)
            dx, dy = xs - cx, ys - cy
            gain = lens._gain(np.hypot(dx, dy) / reach, -1.0) / zoom
            with self.subTest(center=center):
                self.assertGreaterEqual(zoom, 1.0)
                self.assertTrue(((cx + dx * gain) >= -1e-6).all())
                self.assertTrue(((cx + dx * gain) <= width - 1 + 1e-6).all())
                self.assertTrue(((cy + dy * gain) >= -1e-6).all())
                self.assertTrue(((cy + dy * gain) <= height - 1 + 1e-6).all())

    def test_a_cut_out_picture_keeps_its_alpha_and_clean_colour(self):
        rgba = np.zeros((96, 64, 4), np.uint8)
        rgba[30:66, 16:48] = (90, 40, 160, 255)
        data, mode = lens.lens_png(png(rgba, "RGBA"), (0.5, 0.5), 0.7)
        self.assertEqual(mode, "RGBA")
        result = np.asarray(decoded(data).convert("RGBA")).astype(int)
        self.assertEqual(result[0, 0, 3], 0)
        self.assertEqual(result[48, 32, 3], 255)
        edge = (result[..., 3] > 20) & (result[..., 3] < 235)
        self.assertTrue(edge.any())
        np.testing.assert_allclose(result[edge][:, :3], np.tile((90, 40, 160), (edge.sum(), 1)),
                                   atol=3)


if __name__ == "__main__":
    unittest.main()
