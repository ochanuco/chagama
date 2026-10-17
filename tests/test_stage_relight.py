"""Tests for setting a delivery's layers on a backdrop and lighting them by it."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

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


def solid(height, width):
    figure = np.zeros((height, width, 4), np.float32)
    figure[...] = (200, 60, 120, 255)
    return figure, np.zeros_like(figure)


def big_frame(**frame):
    return framed(**{"line": {"color": "#102030", "width": 3.0},
                     "canvas": [0.0, 0.0, 1.0, 1.0], **frame})


class FrameTest(unittest.TestCase):
    def setUp(self):
        self.figure, self.outline = layers()
        self.outline[..., 3] = 0

    def test_a_cut_side_ends_the_figure_at_the_border_and_an_out_side_lets_it_over(self):
        # 400x300 delivery, line 12 px wide: the box's left edge at x 75, top at y 100.
        figure, outline = solid(400, 300)
        staged = stage.in_frame(figure, outline, big_frame(), "#ffffff")
        self.assertEqual(staged.picture.shape, (400, 300, 3))
        alpha = staged.figure_alpha
        self.assertEqual(alpha[150:350, :68].max(), 0.0)
        self.assertEqual(alpha[150:350, 95:280].min(), 1.0)
        np.testing.assert_allclose(np.median(staged.backdrop[150:350, 75], axis=0),
                                   (16, 32, 48), atol=6)
        # The top side is out: the figure stays above the box and over its line.
        self.assertEqual(alpha[50:100, 100:280].min(), 1.0)
        np.testing.assert_allclose(staged.picture[100, 150], (200, 60, 120))
        np.testing.assert_allclose(np.median(staged.backdrop[100, 100:280], axis=0),
                                   (16, 32, 48), atol=6)

    def test_the_rim_edges_both_sides_of_the_line_and_keeps_the_cut_off_it(self):
        figure, outline = solid(400, 300)
        staged = stage.in_frame(
            figure, outline, big_frame(rim={"color": "#405060", "width": 1.0}), "#ffffff")
        rows = slice(150, 350)
        for x, colour in ((60, (255, 255, 255)), (67, (64, 80, 96)), (75, (16, 32, 48)),
                          (83, (64, 80, 96))):
            np.testing.assert_allclose(np.median(staged.backdrop[rows, x], axis=0), colour,
                                       atol=6, err_msg=f"x {x}")
        self.assertEqual(staged.figure_alpha[rows, :64].max(), 0.0)
        self.assertEqual(staged.figure_alpha[rows, 100:280].min(), 1.0)

    def test_a_behind_side_keeps_the_figure_both_sides_with_the_border_over_it(self):
        figure, outline = solid(400, 300)
        sides = {"top": "out", "right": "cut", "bottom": "cut", "left": "behind"}
        staged = stage.in_frame(
            figure, outline, big_frame(box=[0.3, 0.25, 1.0, 1.0], sides=sides), "#ffffff")
        # Box left edge at x 90.
        alpha = staged.figure_alpha
        self.assertEqual(alpha[150:350, :70].min(), 1.0)
        self.assertEqual(alpha[150:350, 110:280].min(), 1.0)
        self.assertLess(alpha[150:350, 90].mean(), 0.1)
        np.testing.assert_allclose(np.median(staged.picture[150:350, 90], axis=0),
                                   (16, 32, 48), atol=6)
        # Above the box the left side does not reach: the figure stays.
        self.assertEqual(alpha[50:90, 80:100].min(), 1.0)

    def test_a_slash_line_lays_slashes_of_its_colour_on_the_paper(self):
        figure, outline = layers(400, 300)
        line = {"color": "#102030", "width": 5.0, "pattern": "slash"}
        staged = stage.in_frame(figure, outline,
                                framed(line=line, canvas=[0.0, 0.0, 1.0, 1.0]), "#ffffff")
        # 400x300: the left side's line runs about x 65..85.
        band = staged.backdrop[200:240, 68:82]
        self.assertLess(band.min(), 30)
        self.assertGreater(band.max(), 250)
        np.testing.assert_allclose(staged.backdrop[300, 150], (255, 255, 255))

    def test_the_same_frame_draws_the_same_pixels(self):
        figure, outline = solid(400, 300)
        frame = big_frame(rim={"color": "#405060", "width": 1.0})
        first = stage.in_frame(figure, outline, frame, "#ffffff")
        second = stage.in_frame(figure, outline, frame, "#ffffff")
        np.testing.assert_array_equal(first.picture, second.picture)

    def test_the_border_is_drawn_by_hand_not_a_straight_band(self):
        figure, outline = solid(400, 300)
        staged = stage.in_frame(figure, outline, big_frame(), "#ffffff")
        covered = (255 - staged.backdrop[150:350, 40:76, 0]) / (255 - 16) >= 0.5
        edge = covered.argmax(axis=1)
        self.assertTrue(covered.any(axis=1).all())
        self.assertGreater(len(np.unique(edge)), 2)

    def test_the_cut_follows_the_border_on_a_shifted_canvas(self):
        figure, outline = solid(400, 300)
        rim = (64, 80, 96)
        staged = stage.in_frame(
            figure, outline,
            big_frame(rim={"color": "#405060", "width": 1.0}, canvas=[0.1, 0.1, 1.0, 1.0]),
            "#ffffff")
        # The canvas starts at delivery (30, 40); look below the out top side.
        painted = (staged.backdrop[90:] == rim).all(axis=2)
        self.assertGreater(painted.sum(), 500)
        self.assertLessEqual(staged.figure_alpha[90:][painted].max(), 0.02)

    def test_the_backdrop_fills_the_box_and_paper_the_rest(self):
        staged = stage.in_frame(self.figure, self.outline,
                                framed(line=None, canvas=[0.0, 0.0, 1.0, 1.0]), "#405060")
        np.testing.assert_allclose(staged.backdrop[2, 2], (255, 255, 255))
        np.testing.assert_allclose(staged.backdrop[30, 20], (64, 80, 96))

    def test_the_default_canvas_holds_box_line_and_visible_figure_with_a_margin(self):
        canvas = stage.frame_canvas_share(
            stage._png(self.figure, "RGBA"), stage._png(self.outline, "RGBA"),
            framed(expand=True))
        # Left: the box's line at 13; top: the figure out to row 10; right and
        # bottom: the line 2 px past the delivery. 3.2 px around, rounded outward.
        np.testing.assert_allclose(canvas, [9 / 60, 6 / 80, 66 / 60, 86 / 80])
        layers = stage._png(self.figure, "RGBA"), stage._png(self.outline, "RGBA")
        kept_in = stage.frame_canvas_share(*layers, framed(expand=False))
        np.testing.assert_allclose(kept_in, [9 / 60, 6 / 80, 1.0, 1.0])
        right_only = stage.frame_canvas_share(*layers, framed(expand={
            "top": True, "right": False, "bottom": True, "left": True}))
        np.testing.assert_allclose(right_only, [9 / 60, 6 / 80, 1.0, 86 / 80])
        dragged = stage.frame_canvas_share(
            *layers, framed(canvas=[None, -0.25, 1.5, None], expand=True))
        np.testing.assert_allclose(dragged, [9 / 60, -0.25, 1.5, 86 / 80])
        staged = stage.in_frame(self.figure, self.outline, framed(expand=True), "#ffffff")
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

    def test_without_a_backdrop_only_the_shading_applies_and_the_picture_stays_clear(self):
        figure, outline, backdrop, picture = relight.relight(
            self.figure, self.figure_alpha, self.outline, self.outline_alpha, None,
            direction="w", strength=1.0)
        self.assertIsNone(backdrop)
        self.assertEqual(picture.shape, (80, 60, 4))
        self.assertGreater(figure[40, 17].mean(), figure[40, 43].mean())
        np.testing.assert_allclose(outline, self.outline)
        self.assertEqual(picture[0, 0, 3], 0)
        self.assertEqual(picture[40, 30, 3], 255)

    def test_without_a_backdrop_the_exposure_is_not_adjusted(self):
        figure, _, _, _ = relight.relight(
            self.figure, self.figure_alpha, self.outline, self.outline_alpha, None,
            direction=None, strength=1.0, scene="sunset")
        self.assertGreater(figure[40, 30][0] / figure[40, 30][2],
                           self.figure[40, 30][0] / self.figure[40, 30][2])
        np.testing.assert_allclose(figure[0, 0], self.figure[0, 0])

    def test_the_png_form_gives_an_rgba_picture_and_no_backdrop_layer(self):
        def png(array):
            output = io.BytesIO()
            Image.fromarray(array.astype(np.uint8), "RGBA").save(output, "PNG")
            return output.getvalue()

        figure, outline = layers()
        picture, figure_png, outline_png, backdrop_png = relight.relight_pngs(
            png(figure), png(outline), None, direction="nw", strength=0.8)
        self.assertIsNone(backdrop_png)
        self.assertEqual(Image.open(io.BytesIO(picture)).mode, "RGBA")
        self.assertEqual(Image.open(io.BytesIO(figure_png)).mode, "RGBA")


if __name__ == "__main__":
    unittest.main()
