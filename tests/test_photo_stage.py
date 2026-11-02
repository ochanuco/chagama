"""Tests for standing a delivered figure in a photo and blurring it by depth."""

from __future__ import annotations

import unittest

import numpy as np

from chagama.infrastructure.imaging import photo_stage

H, W = 120, 180


def scene():
    """A striped photo whose depth rises toward the bottom, with a sign at a
    middle depth on the right."""
    ys, xs = np.mgrid[:H, :W]
    photo = np.where(((xs // 2) % 2)[..., None] == 0, 40.0, 220.0) * np.ones(3)
    depth = (ys / H * 200 + 20).astype(np.float32)
    photo[30:60, 120:170] = np.where(((xs[30:60, 120:170] // 2) % 2)[..., None] == 0,
                                     0.0, 255.0)
    depth[30:60, 120:170] = depth[70, 145]
    return photo.astype(np.float32), depth


def layers():
    figure = np.zeros((80, 60, 4), np.float32)
    figure[10:70, 15:45] = (200, 60, 120, 255)
    figure[10:70, 29:31, :3] = 255
    outline = np.zeros_like(figure)
    outline[5:75, 10:50] = (255, 255, 255, 255)
    return figure, outline


def contrast(image, rows, cols):
    return float(np.std(image[rows, cols]))


class CardTest(unittest.TestCase):
    def test_the_card_is_cropped_to_what_is_visible_and_placed_by_its_base(self):
        figure, outline = layers()
        card = photo_stage.card(figure, outline)
        self.assertEqual(card.shape[:2], (70, 40))
        placed = photo_stage.place_card(card, (W, H), (0.5, 1.0), 0.5)
        rows, cols = np.nonzero(placed[..., 3] > 0.5)
        self.assertEqual(rows.max() - rows.min() + 1, 60)
        self.assertEqual(rows.max(), H - 1)
        self.assertAlmostEqual((cols.min() + cols.max()) / 2, W / 2, delta=1.5)

    def test_a_base_below_the_frame_crops_the_feet(self):
        figure, outline = layers()
        card = photo_stage.card(figure, outline)
        placed = photo_stage.place_card(card, (W, H), (0.5, 1.3), 1.0)
        rows, _ = np.nonzero(placed[..., 3] > 0.5)
        self.assertEqual(rows.max(), H - 1)
        self.assertEqual(rows.min(), round(1.3 * H) - H)

    def test_an_empty_delivery_is_refused(self):
        empty = np.zeros((10, 10, 4), np.float32)
        with self.assertRaisesRegex(ValueError, "figure"):
            photo_stage.card(empty, empty)


class StageTest(unittest.TestCase):
    def stage(self, **options):
        photo, depth = scene()
        figure, outline = layers()
        defaults = {"place": (0.25, 0.95), "height": 0.6, "stand_at": None,
                    "nearer": 0.0, "focus": photo_stage.SUBJECT, "f_number": 1.4}
        return photo_stage.stage(photo, depth, figure, outline, **{**defaults, **options})

    def test_focusing_on_the_card_blurs_the_far_scene_and_keeps_the_card(self):
        result = self.stage()
        photo, _ = scene()
        self.assertLess(contrast(result, slice(2, 12), slice(0, 180)),
                        0.6 * contrast(photo, slice(2, 12), slice(0, 180)))
        np.testing.assert_allclose(result[100, 38], (200, 60, 120), atol=2)

    def test_a_card_standing_at_the_sign_shares_its_focus(self):
        photo, _ = scene()
        sign = (slice(32, 58), slice(122, 168))
        beside = self.stage(stand_at=(0.8, 0.5), focus=(0.8, 0.4))
        np.testing.assert_allclose(beside[100, 38], (200, 60, 120), atol=2)
        self.assertGreater(contrast(beside, *sign), 0.9 * contrast(photo, *sign))
        apart = self.stage(focus=(0.8, 0.4))
        card_edge = (slice(50, 110), slice(40, 50))
        self.assertLess(contrast(apart, *card_edge), contrast(beside, *card_edge))

    def test_nearer_moves_the_card_out_of_the_focused_sign_plane(self):
        sharp = self.stage(stand_at=(0.8, 0.5), focus=(0.8, 0.4))
        pulled = self.stage(stand_at=(0.8, 0.5), nearer=0.5, focus=(0.8, 0.4))
        card_edge = (slice(50, 110), slice(40, 50))
        self.assertLess(contrast(pulled, *card_edge), contrast(sharp, *card_edge))

    def test_a_tilted_card_focused_at_its_top_blurs_toward_its_base(self):
        photo, depth = scene()
        figure = np.zeros((80, 60, 4), np.float32)
        figure[..., :3] = 240
        figure[..., 3] = 255
        figure[::4, :, :3] = 40
        outline = np.zeros_like(figure)
        options = {"place": (0.5, 1.0), "height": 0.95, "stand_at": None,
                   "nearer": 0.0, "focus": (0.5, 0.15), "f_number": 1.4}
        upright = photo_stage.stage(photo, depth, figure, outline, **options)
        tilted = photo_stage.stage(photo, depth, figure, outline, tilt=0.6, **options)
        top, base = (slice(12, 30), slice(80, 100)), (slice(95, 115), slice(80, 100))
        self.assertGreater(contrast(tilted, *top), 0.9 * contrast(upright, *top))
        self.assertLess(contrast(tilted, *base), 0.8 * contrast(upright, *base))


class BlurScaleTest(unittest.TestCase):
    def test_a_far_background_blurs_by_the_aperture_at_the_card_scale(self):
        # 85mm at f/2 is a 42.5 mm aperture; a 1500 px card is 1 px per mm.
        radius = photo_stage.blur_scale(1500, 85.0, 2.0, 1.0, 10_000)
        self.assertAlmostEqual(radius, 42.5 / 2)

    def test_a_closer_up_card_and_a_longer_lens_blur_more(self):
        base = photo_stage.blur_scale(1000, 85.0, 2.8, 1.0, 10_000)
        self.assertAlmostEqual(photo_stage.blur_scale(2000, 85.0, 2.8, 1.0, 10_000), 2 * base)
        self.assertGreater(photo_stage.blur_scale(1000, 135.0, 2.8, 1.0, 10_000), base)

    def test_the_scale_is_per_focus_depth_and_bounded(self):
        near = photo_stage.blur_scale(1000, 85.0, 2.8, 1.0, 10_000)
        self.assertAlmostEqual(photo_stage.blur_scale(1000, 85.0, 2.8, 0.5, 10_000), 2 * near)
        self.assertAlmostEqual(photo_stage.blur_scale(1000, 85.0, 2.8, 0.0, 10_000),
                               near / photo_stage.FOCUS_DEPTH_FLOOR)
        self.assertAlmostEqual(photo_stage.blur_scale(100_000, 85.0, 1.4, 1.0, 1000),
                               photo_stage.MAX_RADIUS_FRACTION * 1000)


class LayeredBlurTest(unittest.TestCase):
    def test_a_near_out_of_focus_object_spreads_over_the_sharp_scene_behind(self):
        linear = np.full((60, 80, 3), 0.1, np.float32)
        depth = np.zeros((60, 80), np.float32)
        linear[:, :40] = 1.0
        depth[:, :40] = 1.0
        blurred = photo_stage.layered_blur(linear, depth, focus_depth=0.0, scale=10.0)
        self.assertGreater(blurred[30, 43, 0], 0.2)
        np.testing.assert_allclose(blurred[30, 70], (0.1, 0.1, 0.1), atol=1e-4)

    def test_a_far_out_of_focus_scene_stays_behind_a_sharp_near_object(self):
        linear = np.full((60, 80, 3), 0.1, np.float32)
        depth = np.ones((60, 80), np.float32)
        linear[:, 40:] = 1.0
        depth[:, 40:] = 0.0
        blurred = photo_stage.layered_blur(linear, depth, focus_depth=1.0, scale=10.0)
        np.testing.assert_allclose(blurred[30, 36], (0.1, 0.1, 0.1), atol=1e-4)


class BokehTest(unittest.TestCase):
    def test_out_of_focus_highlights_open_into_bright_discs(self):
        photo = np.full((H, W, 3), 15.0, np.float32)
        photo[20:24, 100:104] = 255.0
        depth = np.tile(np.linspace(0, 255, H, dtype=np.float32)[:, None], (1, W))
        figure, outline = layers()
        # A long lens, so the small card in this tiny photo still blurs by pixels.
        options = {"place": (0.1, 1.0), "height": 0.3, "stand_at": None, "nearer": 0.0,
                   "focus": photo_stage.SUBJECT, "f_number": 1.4, "focal_mm": 600.0}
        plain = photo_stage.stage(photo, depth, figure, outline, **options)
        lifted = photo_stage.stage(photo, depth, figure, outline, bokeh=1.0, **options)
        around = (slice(14, 31), slice(94, 111))
        self.assertGreater((lifted[around] - 15).sum(), 1.5 * (plain[around] - 15).sum())

    def test_in_focus_highlights_stay_as_they_were(self):
        linear = np.full((4, 4, 3), 0.95, np.float32)
        lifted = photo_stage.lift_highlights(linear, np.zeros((4, 4), np.float32), 1.0)
        np.testing.assert_array_equal(lifted, linear)
        self.assertIs(photo_stage.lift_highlights(linear, np.ones((4, 4)), 0.0), linear)


if __name__ == "__main__":
    unittest.main()
