"""The trail drawing: where its parts land, what it keeps out, and that a
bigger picture draws the same trail."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from chagama.infrastructure.imaging import trail as T

TSU = T.Trail(points=((0.02, 0.34), (0.72, 0.20), (1.0, 0.52), (0.22, 0.94)),
              front=(False, False, True, True), path="curve", rule="both", width=5.0,
              color="#6f5c78", head="light", head_at_end=True, seed=3)


def _png(array, mode):
    out = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode).save(out, "PNG")
    return out.getvalue()


class PiecesTest(unittest.TestCase):
    def test_both_puts_a_segment_in_front_only_when_both_ends_are(self):
        self.assertEqual([s[0][2] for s in T.pieces((False, True, True, False), "both")],
                         [False, True, False])

    def test_split_switches_at_the_midpoint_of_a_mixed_segment(self):
        self.assertEqual(T.pieces((False, True, True, True), "split")[0],
                         [(0.0, 0.5, False), (0.5, 1.0, True)])

    def test_start_follows_the_segments_first_point(self):
        self.assertEqual([s[0][2] for s in T.pieces((True, False, True, False), "start")],
                         [True, False, True])


class PathTest(unittest.TestCase):
    def test_the_curve_passes_through_every_point(self):
        points = [(10, 20), (300, 40), (380, 300), (60, 380)]
        path = T.sample_path(points, curve=True)
        for i, point in enumerate(points[:3]):
            first = int(np.nonzero(path.segment == i)[0][0])
            np.testing.assert_allclose(path.xy[first], point, atol=1e-6)
        np.testing.assert_allclose(path.xy[-1], points[-1], atol=1e-6)

    def test_a_line_runs_straight_between_points(self):
        path = T.sample_path([(0, 0), (100, 0), (100, 100), (0, 100)], curve=False)
        on_first = path.xy[path.segment == 0]
        np.testing.assert_allclose(on_first[:, 1], 0.0, atol=1e-9)


class RenderTest(unittest.TestCase):
    def test_the_back_and_front_layers_split_the_trail_by_segment(self):
        back, front = T.render(TSU, 300, 200)
        self.assertGreater(back[..., 3].sum(), 0)
        self.assertGreater(front[..., 3].sum(), 0)
        # The top band holds the first two (back) segments only.
        self.assertLess(front[:60, :, 3].sum(), back[:60, :, 3].sum() * 0.05)
        # The head sits at the end point, in front.
        hx, hy = round(0.22 * 200), round(0.94 * 300)
        self.assertGreater(front[hy, hx, 3], 0.9)

    def test_the_back_box_keeps_the_back_layer_inside_it(self):
        back, _front = T.render(TSU, 300, 200, back_box=(0.1, 0.1, 0.9, 0.9))
        self.assertEqual(back[:, :19, 3].max(), 0.0)
        self.assertEqual(back[:29, :, 3].max(), 0.0)

    def test_the_same_seed_draws_the_same_trail(self):
        a = T.render(TSU, 150, 100)[0]
        b = T.render(TSU, 150, 100)[0]
        np.testing.assert_array_equal(a, b)

    def test_coverage_does_not_grow_with_the_picture(self):
        small = T.render(TSU, 720, 480)
        large = T.render(TSU, 1440, 960)
        share = [sum(layer[..., 3].mean() for layer in pair) for pair in (small, large)]
        self.assertAlmostEqual(share[1] / share[0], 1.0, delta=0.25)


class PngTest(unittest.TestCase):
    def test_the_picture_is_backdrop_back_outline_figure_front(self):
        h, w = 120, 80
        figure = np.zeros((h, w, 4)); figure[40:80, 30:50] = (200, 10, 10, 255)
        outline = np.zeros((h, w, 4))
        backdrop = np.full((h, w, 3), 50.0)
        picture, behind, front = T.trail_pngs(_png(figure, "RGBA"), _png(outline, "RGBA"),
                                              _png(backdrop, "RGB"), TSU, None)
        picture = np.array(Image.open(io.BytesIO(picture)))
        front_a = np.array(Image.open(io.BytesIO(front)))[..., 3]
        self.assertEqual(Image.open(io.BytesIO(behind)).mode, "RGB")
        # Where nothing of the trail lands on the figure, the figure shows as it was.
        clear = (front_a[40:80, 30:50] == 0)
        self.assertTrue(clear.any())
        self.assertTrue((picture[40:80, 30:50][clear] == [200, 10, 10]).all())

    def test_over_keeps_an_opaque_picture_opaque(self):
        image = _png(np.full((4, 4, 3), 100.0), "RGB")
        layer = np.zeros((4, 4, 4)); layer[0, 0] = (255, 255, 255, 255)
        out = np.array(Image.open(io.BytesIO(T.over_png(image, _png(layer, "RGBA")))))
        self.assertEqual(out.shape, (4, 4, 3))
        self.assertEqual(tuple(out[0, 0]), (255, 255, 255))
        self.assertEqual(tuple(out[1, 1]), (100, 100, 100))


if __name__ == "__main__":
    unittest.main()
