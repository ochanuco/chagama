"""Tests for setting a delivery's layers on a backdrop and lighting them by it."""

from __future__ import annotations

import unittest

import numpy as np

from chagama.infrastructure.imaging import photo_stage, relight, stage


def layers(height=80, width=60):
    figure = np.zeros((height, width, 4), np.float32)
    figure[10:70, 15:45] = (200, 60, 120, 255)
    outline = np.zeros_like(figure)
    outline[5:75, 10:50] = (255, 255, 255, 255)
    return figure, outline


class PatternTest(unittest.TestCase):
    def test_the_layers_keep_their_canvas_and_composite_over_the_colour(self):
        figure, outline = layers()
        staged = stage.on_pattern(figure, outline, "#102030")
        self.assertEqual(staged.picture.shape, (80, 60, 3))
        np.testing.assert_allclose(staged.picture[0, 0], (16, 32, 48))
        np.testing.assert_allclose(staged.picture[40, 30], (200, 60, 120))
        np.testing.assert_allclose(staged.picture[6, 12], (255, 255, 255))
        np.testing.assert_array_equal(staged.figure_alpha, figure[..., 3] / 255.0)

    def test_a_shadow_darkens_the_backdrop_away_from_the_light(self):
        figure, outline = layers()
        staged = stage.on_pattern(figure, outline, "#ffffff", shadow_from="nw", outlines=[])
        # Lit from the top left, the shadow falls to the bottom right.
        self.assertLess(staged.picture[75, 50].mean(), 250)
        np.testing.assert_allclose(staged.picture[2, 2], (255, 255, 255))
        np.testing.assert_allclose(staged.picture[40, 30], (200, 60, 120))


    def test_a_scene_tints_the_backdrop_and_casts_its_coloured_shadow(self):
        figure, outline = layers()
        plain = stage.on_pattern(figure, outline, "#ffffff", outlines=[])
        moon = stage.on_pattern(figure, outline, "#ffffff", shadow_from="nw", outlines=[],
                                scene="moon", light_from="nw")
        # The moon dims the backdrop and turns it blue.
        self.assertLess(moon.backdrop[2, 2].mean(), plain.backdrop[2, 2].mean())
        self.assertGreater(moon.backdrop[2, 2][2], moon.backdrop[2, 2][0])
        # Its shadow is painted into the backdrop, not the outline layer.
        self.assertLess(moon.picture[75, 50].mean(), moon.backdrop[2, 2].mean() * 0.95)
        np.testing.assert_allclose(moon.outline_alpha, outline[..., 3] / 255.0)
        np.testing.assert_allclose(moon.picture[40, 30], (200, 60, 120))


def framed(**frame):
    return {"box": [0.25, 0.25, 1.0, 1.0],
            "sides": {"top": "out", "right": "cut", "bottom": "cut", "left": "cut"},
            "line": {"color": "#102030", "width": 5.0}, "paper": "#ffffff",
            "canvas": None, "strokes": [], **frame}


class FrameTest(unittest.TestCase):
    def setUp(self):
        self.figure, self.outline = layers()
        self.outline[..., 3] = 0

    def test_a_cut_side_clips_at_the_line_and_an_out_side_lets_the_figure_over(self):
        # 80x60 delivery: the line is 4 px wide, the box's left edge at x 15.
        staged = stage.in_frame(self.figure, self.outline,
                                framed(canvas=[0.0, 0.0, 1.0, 1.0]), "#ffffff")
        self.assertEqual(staged.picture.shape, (80, 60, 3))
        alpha = staged.figure_alpha
        self.assertEqual(alpha[40, 16], 0.0)
        self.assertEqual(alpha[40, 17], 1.0)
        np.testing.assert_allclose(staged.picture[40, 14], (16, 32, 48))
        # The top side is out: the figure stays above the box and over its line.
        self.assertEqual(alpha[12, 30], 1.0)
        np.testing.assert_allclose(staged.picture[20, 30], (200, 60, 120))
        np.testing.assert_allclose(staged.backdrop[20, 30], (16, 32, 48))

    def test_the_rim_edges_both_sides_of_the_line_and_moves_the_cut_inward(self):
        # The line 4 px wide on the box's left edge at x 15, a 2 px rim either side.
        staged = stage.in_frame(
            self.figure, self.outline,
            framed(rim={"color": "#405060", "width": 2.5}, canvas=[0.0, 0.0, 1.0, 1.0]),
            "#ffffff")
        np.testing.assert_allclose(staged.backdrop[40, 12], (64, 80, 96))
        np.testing.assert_allclose(staged.backdrop[40, 14], (16, 32, 48))
        np.testing.assert_allclose(staged.backdrop[40, 18], (64, 80, 96))
        np.testing.assert_allclose(staged.backdrop[40, 10], (255, 255, 255))
        self.assertEqual(staged.figure_alpha[40, 18], 0.0)
        self.assertEqual(staged.figure_alpha[40, 19], 1.0)

    def test_a_behind_side_keeps_the_figure_both_sides_with_the_border_over_it(self):
        sides = {"top": "out", "right": "cut", "bottom": "cut", "left": "behind"}
        staged = stage.in_frame(
            self.figure, self.outline,
            framed(box=[0.3, 0.25, 1.0, 1.0], sides=sides, canvas=[0.0, 0.0, 1.0, 1.0]),
            "#ffffff")
        # Box left edge at x 18: the strip is x 16..20, the figure stays either side.
        alpha = staged.figure_alpha
        self.assertEqual(alpha[40, 15], 1.0)
        self.assertEqual(alpha[40, 17], 0.0)
        self.assertEqual(alpha[40, 20], 1.0)
        np.testing.assert_allclose(staged.picture[40, 17], (16, 32, 48))
        # Above the box the left side does not reach: the figure stays.
        self.assertEqual(alpha[12, 17], 1.0)

    def test_the_backdrop_fills_the_box_and_paper_the_rest(self):
        staged = stage.in_frame(self.figure, self.outline,
                                framed(line=None, canvas=[0.0, 0.0, 1.0, 1.0]), "#405060")
        np.testing.assert_allclose(staged.backdrop[2, 2], (255, 255, 255))
        np.testing.assert_allclose(staged.backdrop[30, 20], (64, 80, 96))

    def test_the_default_canvas_holds_box_line_and_visible_figure_with_a_margin(self):
        canvas = stage.frame_canvas_share(
            stage._png(self.figure, "RGBA"), stage._png(self.outline, "RGBA"), framed())
        # Left: the box's line at 13; top: the figure out to row 10; right and
        # bottom: the line 2 px past the delivery. 3.2 px around, rounded outward.
        np.testing.assert_allclose(canvas, [9 / 60, 6 / 80, 66 / 60, 86 / 80])
        kept_in = stage.frame_canvas_share(
            stage._png(self.figure, "RGBA"), stage._png(self.outline, "RGBA"),
            framed(expand=False))
        np.testing.assert_allclose(kept_in, [9 / 60, 6 / 80, 1.0, 1.0])
        staged = stage.in_frame(self.figure, self.outline, framed(), "#ffffff")
        self.assertEqual(staged.picture.shape, (round(canvas[3] * 80) - round(canvas[1] * 80),
                                                round(canvas[2] * 60) - round(canvas[0] * 60), 3))

    def test_strokes_erase_then_restore_in_order(self):
        erase = {"mode": "erase", "width": 10.0, "points": [[0.5, 0.5]]}
        restore = {"mode": "restore", "width": 10.0, "points": [[0.5, 0.5], [0.5, 0.95]]}
        whole = [0.0, 0.0, 1.0, 1.0]
        erased = stage.in_frame(self.figure, self.outline,
                                framed(canvas=whole, strokes=[erase]), "#ffffff")
        self.assertEqual(erased.figure_alpha[40, 30], 0.0)
        self.assertEqual(erased.figure_alpha[30, 30], 1.0)
        back = stage.in_frame(self.figure, self.outline,
                              framed(canvas=whole, strokes=[erase, restore]), "#ffffff")
        self.assertEqual(back.figure_alpha[40, 30], 1.0)
        # A restore below the bottom cut brings back the figure, never more of it.
        self.assertEqual(back.figure_alpha[69, 30], 1.0)
        self.assertEqual(back.figure_alpha[74, 30], 0.0)


class PhotoTest(unittest.TestCase):
    def test_figure_and_outline_scale_together_onto_the_photo(self):
        figure, outline = layers()
        photo = np.full((100, 200, 3), 90.0)
        staged = stage.on_photo(photo, figure, outline, place=(0.5, 0.9), height=0.7)
        self.assertEqual(staged.figure.shape, (100, 200, 3))
        rows, cols = np.nonzero(staged.outline_alpha > 0.5)
        self.assertAlmostEqual(rows.max() + 1, 90, delta=1)
        self.assertAlmostEqual(rows.max() + 1 - rows.min(), 70, delta=1)
        self.assertAlmostEqual((cols.min() + cols.max() + 1) / 2, 100, delta=1)
        # The figure keeps its place inside the outline: 5 of 70 rows below.
        figure_rows, _ = np.nonzero(staged.figure_alpha > 0.5)
        self.assertAlmostEqual(figure_rows.max() + 1, 85, delta=1)
        np.testing.assert_allclose(staged.picture[2, 2], (90, 90, 90))

    def test_the_blur_of_placed_layers_matches_the_photo_stage(self):
        figure, outline = layers()
        ys = np.mgrid[:100, :200][0]
        photo = np.where((np.arange(200) // 2 % 2)[None, :, None] == 0, 40.0, 220.0) \
            * np.ones((100, 1, 3))
        depth = (ys / 100 * 200 + 20).astype(np.float32)
        options = {"place": (0.5, 0.9), "height": 0.7, "stand_at": None, "nearer": 0.1,
                   "focus": photo_stage.SUBJECT, "f_number": 2.0}
        staged = stage.on_photo(photo, figure, outline, place=(0.5, 0.9), height=0.7)
        layered = stage.blur_on_photo(
            staged.backdrop, depth, np.dstack([staged.figure, staged.figure_alpha * 255]),
            np.dstack([staged.outline, staged.outline_alpha * 255]), **options)
        direct = photo_stage.stage(photo, depth, figure, outline, **options)
        np.testing.assert_allclose(layered, direct, atol=2)


class RelightTest(unittest.TestCase):
    def setUp(self):
        figure, outline = layers()
        self.figure, self.outline = figure[..., :3], outline[..., :3]
        self.figure_alpha, self.outline_alpha = figure[..., 3] / 255, outline[..., 3] / 255

    def lit(self, backdrop, direction=None, strength=1.0, **scene):
        figure, outline, _, picture = relight.relight(
            self.figure, self.figure_alpha, self.outline, self.outline_alpha, backdrop,
            direction=direction, strength=strength, **scene)
        return figure, outline, picture

    def test_a_dark_ambient_darkens_the_figure_and_the_outline(self):
        figure, outline, picture = self.lit(np.full((80, 60, 3), 20.0))
        self.assertLess(figure[40, 30].mean(), self.figure[40, 30].mean())
        self.assertLess(outline[6, 12].mean(), 255)
        np.testing.assert_allclose(picture[0, 0], (20, 20, 20))

    def test_strength_zero_leaves_the_layers_alone(self):
        figure, outline, _ = self.lit(np.full((80, 60, 3), 20.0), "nw", 0.0)
        np.testing.assert_allclose(figure, self.figure, atol=1)
        np.testing.assert_allclose(outline, self.outline, atol=1)

    def test_a_coloured_ambient_casts_its_colour(self):
        backdrop = np.zeros((80, 60, 3))
        backdrop[...] = (80, 120, 220)
        figure, _, _ = self.lit(backdrop)
        before = self.figure[40, 30] / self.figure[40, 30].sum()
        after = figure[40, 30] / figure[40, 30].sum()
        self.assertGreater(after[2], before[2])

    def test_a_scene_lights_the_figure_and_tints_only_a_pattern_backdrop(self):
        backdrop = np.full((80, 60, 3), 200.0)
        args = (self.figure, self.figure_alpha, self.outline, self.outline_alpha, backdrop)
        figure, _, tinted, _ = relight.relight(*args, direction=None, strength=0.0,
                                               scene="sunset", tint_backdrop=True)
        # Sunset warms the figure and the pattern behind it.
        self.assertGreater(figure[40, 30][0] / figure[40, 30][2],
                           self.figure[40, 30][0] / self.figure[40, 30][2])
        self.assertGreater(tinted[2, 2][0], tinted[2, 2][2])
        _, _, photo, _ = relight.relight(*args, direction="ne", strength=0.0, scene="moon")
        np.testing.assert_allclose(photo, backdrop)

    def test_without_depth_an_inner_edge_casts_no_local_shadow(self):
        # A notch cut into the figure's right side faces the light's far side; the
        # silhouette alone must not darken the flat area beside it.
        figure = np.zeros((80, 60, 3)) + (230, 200, 190)
        alpha = np.zeros((80, 60))
        alpha[10:70, 10:50] = 1.0
        alpha[38:42, 30:50] = 0.0
        backdrop = np.full((80, 60, 3), 200.0)
        lit, _, _, _ = relight.relight(figure, alpha, np.zeros_like(figure), np.zeros((80, 60)),
                                       backdrop, direction="nw", strength=0.0, scene="sunset")
        # Above and below the notch at the same distance from the light stay alike.
        self.assertLess(abs(lit[36, 30].mean() - lit[44, 26].mean()), 6)

    def test_a_direction_shades_the_far_side(self):
        backdrop = np.full((80, 60, 3), 200.0)
        figure, _, _ = self.lit(backdrop, "w")
        self.assertGreater(figure[40, 17].mean(), figure[40, 43].mean())


if __name__ == "__main__":
    unittest.main()
