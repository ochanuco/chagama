"""Matting stage: trimap, known-region forcing and the PNG round trip."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from chagama.domain.yukari import delivery_style
from chagama.infrastructure.imaging import delivery, matting


def rgb_png(pixels: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(output, "PNG")
    return output.getvalue()


def figure_mask(size=64, box=(16, 48, 16, 48)) -> np.ndarray:
    figure = np.zeros((size, size), dtype=bool)
    top, bottom, left, right = box
    figure[top:bottom, left:right] = True
    return figure


class TrimapTest(unittest.TestCase):
    def test_trimap_marks_the_eroded_core_figure_and_the_far_backdrop(self):
        known = matting.trimap(figure_mask(), 4)
        self.assertEqual(set(np.unique(known)), {0, 128, 255})
        self.assertTrue((known[20:44, 20:44] == 255).all())
        self.assertEqual(known[0, 0], 0)

    def test_trimap_unknown_band_straddles_the_figure_edge(self):
        known = matting.trimap(figure_mask(), 3)
        self.assertEqual(known[12, 32], 0)
        self.assertEqual(known[13, 32], 128)
        self.assertEqual(known[18, 32], 128)
        self.assertEqual(known[19, 32], 255)
        self.assertEqual(known[44, 32], 255)
        self.assertEqual(known[45, 32], 128)
        self.assertEqual(known[50, 32], 128)
        self.assertEqual(known[51, 32], 0)


class KnownRegionsTest(unittest.TestCase):
    def test_a_key_pocket_narrower_than_the_band_is_known_backdrop(self):
        pixels = np.full((64, 64, 3), (138, 196, 138), dtype=np.uint8)
        pixels[16:48, 16:48] = (170, 140, 210)
        pixels[26:36, 30:33] = (122, 168, 120)       # the key in a loop of hair
        soft = np.zeros((64, 64), dtype=np.uint8)
        soft[16:48, 16:48] = 255
        px = pixels.astype(float)
        known = matting.known_regions(px, soft)
        figure = delivery.cut_figure(px, soft)
        self.assertFalse(figure[26:36, 30:33].any())
        self.assertTrue((known[26:36, 30:33] == matting.KNOWN_BACKDROP).all())
        self.assertEqual(known[24, 31], matting.UNKNOWN)
        np.testing.assert_array_equal(
            known[~figure_mask(box=(26, 36, 30, 33))],
            matting.trimap(figure, delivery_style.MATTING_TRIMAP_PX)[
                ~figure_mask(box=(26, 36, 30, 33))])


class SettleTest(unittest.TestCase):
    def test_settle_forces_the_known_regions_onto_the_alpha(self):
        known = np.full((8, 8), matting.UNKNOWN, dtype=np.uint8)
        known[:, :2] = matting.KNOWN_FIGURE
        known[:, 6:] = matting.KNOWN_BACKDROP
        out = matting.settle(np.full((8, 8), 0.5), known)
        self.assertTrue((out[:, :2] == 1.0).all())
        self.assertTrue((out[:, 6:] == 0.0).all())
        self.assertTrue((out[:, 2:6] == 0.5).all())

    def test_settle_clips_the_models_alpha_to_the_unit_range(self):
        known = np.full((4, 4), matting.UNKNOWN, dtype=np.uint8)
        alpha = np.array([[-0.5, 1.5, 0.25, 0.75]] * 4)
        np.testing.assert_array_equal(
            matting.settle(alpha, known)[0], [0.0, 1.0, 0.25, 0.75])


class FigureColourTest(unittest.TestCase):
    def test_keeps_the_raw_pixels_where_alpha_is_zero(self):
        px = np.zeros((8, 8, 3))
        px[...] = (10.0, 20.0, 30.0)
        px[:, 6:] = (200.0, 100.0, 50.0)
        alpha = np.full((8, 8), 0.5)
        alpha[:, 6:] = 0.0
        colour = matting.figure_colour(px, alpha, lambda rgb, a: np.zeros_like(rgb))
        np.testing.assert_array_equal(colour[:, 6:], px[:, 6:])
        self.assertEqual(float(colour[4, 0].max()), 0.0)


class AlphaAndForegroundPngTest(unittest.TestCase):
    def setUp(self):
        self.pixels = np.full((64, 64, 3), (210, 230, 235), dtype=np.uint8)
        self.pixels[16:48, 16:48] = (40, 40, 40)
        soft = np.zeros((64, 64), dtype=np.uint8)
        soft[16:48, 16:48] = 255
        self.soft = soft
        self.seen = {}

    def predict(self, rgb, known):
        self.seen["rgb"], self.seen["known"] = rgb, known
        return np.full(known.shape, 0.5)

    def test_alpha_png_is_the_settled_alpha_as_an_l_png(self):
        alpha_img = Image.open(io.BytesIO(matting.alpha_png(
            rgb_png(self.pixels), rgb_png(self.soft), self.predict)))
        alpha = np.array(alpha_img)
        self.assertEqual(alpha_img.mode, "L")
        self.assertEqual(self.seen["rgb"].dtype, np.uint8)
        self.assertEqual(self.seen["known"].shape, (64, 64))
        self.assertEqual(alpha[32, 32], 255)
        self.assertEqual(alpha[0, 0], 0)
        unknown = self.seen["known"] == matting.UNKNOWN
        self.assertTrue(unknown.any())
        self.assertTrue((alpha[unknown] == 128).all())

    def test_foreground_png_matches_the_colour_the_alpha_asks_for(self):
        alpha_png = matting.alpha_png(
            rgb_png(self.pixels), rgb_png(self.soft), self.predict)
        image = np.array(Image.open(io.BytesIO(matting.foreground_png(
            rgb_png(self.pixels), alpha_png, lambda rgb, alpha: rgb))))
        self.assertEqual(image.shape, (64, 64, 3))
        np.testing.assert_array_equal(image[0, 0], self.pixels[0, 0])
        np.testing.assert_allclose(image[32, 32], (40, 40, 40), atol=1)

    def test_composed_stages_equal_the_single_pass_the_node_used_to_run(self):
        px = self.pixels.astype(float)
        known = matting.known_regions(px, self.soft)
        alpha = matting.settle(matting.tiled(
            self.predict, self.pixels, known, delivery_style.MATTING_TILE_PX,
            delivery_style.MATTING_TILE_OVERLAP_PX), known)
        quantised = np.rint(alpha * 255) / 255.0

        def foreground(rgb, a):
            return rgb * a[..., None] + 0.1 * (1 - a[..., None])

        expected = matting.figure_colour(px, quantised, foreground)
        composed = np.array(Image.open(io.BytesIO(matting.foreground_png(
            rgb_png(self.pixels),
            matting.alpha_png(rgb_png(self.pixels), rgb_png(self.soft),
                              self.predict),
            foreground))))
        np.testing.assert_array_equal(
            composed, np.clip(np.rint(expected), 0, 255).astype(np.uint8))


KEY = np.array((120.0, 185.0, 119.0))
WHITE = np.array((255.0, 255.0, 255.0))


def film_scene(film=0.7, key=KEY, area=(20, 60, 20, 60)):
    """A purple figure on `key` with a white film at coverage `film` over
    the key inside it."""
    px = np.zeros((96, 96, 3))
    px[...] = key
    px[8:88, 8:88] = (170.0, 140.0, 210.0)
    top, bottom, left, right = area
    px[top:bottom, left:right] = film * WHITE + (1 - film) * key
    known = matting.trimap(figure_mask(96, (8, 88, 8, 88)), 4)
    return px, known


class SeeThroughTest(unittest.TestCase):
    def test_a_film_over_the_key_is_capped_at_its_coverage(self):
        px, known = film_scene()
        alpha = matting.see_through(px, (known > 0).astype(float), known)
        np.testing.assert_allclose(alpha[40, 40], 0.7, atol=1e-6)
        self.assertEqual(alpha[12, 12], 1.0)
        self.assertEqual(alpha[0, 0], 0.0)

    def test_off_a_green_key_the_alpha_is_untouched(self):
        grey = np.array((200.0, 200.0, 200.0))
        px, known = film_scene(key=grey)
        alpha = (known > 0).astype(float)
        np.testing.assert_array_equal(matting.see_through(px, alpha, known), alpha)

    def test_a_tinted_speck_under_the_minimum_area_stays_opaque(self):
        px, known = film_scene(area=(40, 44, 40, 44))
        alpha = (known > 0).astype(float)
        np.testing.assert_array_equal(matting.see_through(px, alpha, known), alpha)

    def test_a_pale_cyan_the_key_does_not_lead_is_not_see_through(self):
        px, known = film_scene()
        px[20:60, 20:60] = (200.0, 235.0, 245.0)
        alpha = (known > 0).astype(float)
        np.testing.assert_array_equal(matting.see_through(px, alpha, known), alpha)

    def test_clearer_scales_the_key_showing_through_the_inside(self):
        px, known = film_scene()
        alpha = matting.see_through(px, (known > 0).astype(float), known)
        clearer = matting.clearer(px, alpha, 2.0)
        np.testing.assert_allclose(clearer[40, 40], 0.4, atol=1e-6)
        self.assertEqual(clearer[12, 12], 1.0)
        np.testing.assert_array_equal(matting.clearer(px, alpha, 1.0), alpha)

    def test_the_film_colour_is_unmixed_from_the_key(self):
        px, known = film_scene()
        alpha = matting.see_through(px, (known > 0).astype(float), known)
        colour = matting.figure_colour(px, alpha, lambda rgb, a: np.zeros_like(rgb))
        np.testing.assert_allclose(colour[40, 40], WHITE, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
