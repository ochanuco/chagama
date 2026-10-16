"""Tests for the depth-of-field blur over a delivery's layers and its graph."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from chagama.infrastructure.comfyui.dof_graph import dof_graph
from chagama.infrastructure.comfyui.refinement_graph import DEPTH_NODE
from chagama.infrastructure.imaging.depth_blur import (
    blur_layers,
    placed_depth,
    blur_layers_png,
    sharp_reach_per_f,
)

ALL = {"figure": True, "outline": True, "backdrop": True}
NONE = {"figure": False, "outline": False, "backdrop": False}
SIZE = (160, 120)


def layers():
    """A striped figure, a ring of outline around it, a striped backdrop and
    a depth that is near at the bottom, far at the top."""
    height, width = SIZE
    yy, xx = np.mgrid[0:height, 0:width]
    ellipse = ((xx - 60) / 30.0) ** 2 + ((yy - 80) / 60.0) ** 2
    figure_alpha = (ellipse <= 1.0).astype(np.float32)
    outline_alpha = ((ellipse <= 1.3) & (ellipse > 1.0)).astype(np.float32)
    figure = np.zeros((height, width, 3), np.float32)
    figure[..., 0] = np.where(xx % 6 < 3, 230, 30)
    figure[..., 1] = 80
    figure[..., 2] = 120
    outline = np.zeros((height, width, 3), np.float32)
    outline[...] = (136, 91, 128)
    backdrop = np.zeros((height, width, 3), np.float32)
    backdrop[...] = np.where((yy % 8 < 4)[..., None], 240, 60)
    depth = (yy / height * 255.0).astype(np.float32)
    return figure, figure_alpha, outline, outline_alpha, backdrop, depth


def plain_composite(figure, figure_alpha, outline, outline_alpha, backdrop):
    result = backdrop
    for colour, alpha in ((outline, outline_alpha), (figure, figure_alpha)):
        result = colour * alpha[..., None] + result * (1 - alpha[..., None])
    return result


def run(scope, focus=(0.5, 0.15), f_number=1.4, transparent=False):
    figure, figure_alpha, outline, outline_alpha, backdrop, depth = layers()
    return blur_layers(figure, figure_alpha, outline, outline_alpha,
                       None if transparent else backdrop, depth, *focus,
                       f_number, scope)


class BlurLayersTest(unittest.TestCase):
    def setUp(self):
        self.figure, self.figure_alpha, self.outline, self.outline_alpha, \
            self.backdrop, _ = layers()
        self.plain = plain_composite(self.figure, self.figure_alpha, self.outline,
                                     self.outline_alpha, self.backdrop)

    def test_no_scope_reproduces_the_delivered_composite(self):
        colour, alpha = run(NONE)
        np.testing.assert_allclose(colour, self.plain, atol=1e-3)
        np.testing.assert_array_equal(alpha, 1.0)

    def test_figure_only_leaves_what_the_blur_cannot_reach_sharp(self):
        colour, _ = run({"figure": True, "outline": False, "backdrop": False})
        changed = np.abs(colour - self.plain).max(axis=2) > 0.5
        far_from_figure = np.ones(SIZE, bool)
        far_from_figure[:, 10:110] = False
        self.assertTrue(changed[self.figure_alpha > 0].any())
        self.assertFalse(changed[far_from_figure].any())

    def test_backdrop_only_keeps_the_figure_and_its_outline_sharp(self):
        colour, _ = run({"figure": False, "outline": False, "backdrop": True},
                        focus=(0.5, 0.9))
        covered = (self.figure_alpha + self.outline_alpha) >= 1
        np.testing.assert_allclose(colour[covered], self.plain[covered], atol=1e-3)
        self.assertGreater(np.abs(colour - self.plain)[~covered].max(), 10)

    def test_the_far_side_of_the_figure_blurs_more_than_the_focus(self):
        colour, _ = run({"figure": True, "outline": False, "backdrop": False},
                        focus=(0.5, 0.85))
        stripes = colour[..., 0]
        near, far = stripes[130:135, 45:75], stripes[30:35, 45:75]
        self.assertGreater(near.std(), far.std() * 1.5)

    def test_larger_f_number_blurs_less(self):
        wide, _ = run(ALL, f_number=1.4)
        narrow, _ = run(ALL, f_number=16)
        self.assertGreater(np.abs(wide - self.plain).sum(),
                           np.abs(narrow - self.plain).sum() * 2)

    def test_a_transparent_delivery_stays_transparent(self):
        colour, alpha = run(ALL, transparent=True)
        self.assertEqual(alpha[0, 0], 0.0)
        self.assertGreater(alpha[80, 60], 0.99)
        _, sharp = run(NONE, transparent=True)
        np.testing.assert_allclose(
            sharp, self.figure_alpha + self.outline_alpha * (1 - self.figure_alpha))

    def test_flat_depth_blurs_nothing(self):
        figure, figure_alpha, outline, outline_alpha, backdrop, _ = layers()
        colour, _ = blur_layers(figure, figure_alpha, outline, outline_alpha,
                                backdrop, np.full(SIZE, 100.0, np.float32),
                                0.5, 0.5, 1.4, ALL)
        np.testing.assert_allclose(colour, self.plain, atol=1e-3)

    def test_the_guide_radius_stays_scale_invariant(self):
        self.assertAlmostEqual(sharp_reach_per_f(), 0.0417, places=4)


def png(array: np.ndarray, mode: str) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode).save(output, "PNG")
    return output.getvalue()


class BlurLayersPngTest(unittest.TestCase):
    def pngs(self):
        figure, figure_alpha, outline, outline_alpha, backdrop, depth = layers()
        return (png(np.dstack([figure, figure_alpha * 255]), "RGBA"),
                png(np.dstack([outline, outline_alpha * 255]), "RGBA"),
                png(backdrop, "RGB"), png(depth, "L"))

    def test_rgb_over_a_backdrop_rgba_without_one(self):
        figure, outline, backdrop, depth = self.pngs()
        opaque = Image.open(io.BytesIO(blur_layers_png(
            figure, outline, backdrop, depth, 0.5, 0.5, 2.8, ALL)))
        cut = Image.open(io.BytesIO(blur_layers_png(
            figure, outline, None, depth, 0.5, 0.5, 2.8, ALL)))
        self.assertEqual((opaque.mode, opaque.size), ("RGB", (120, 160)))
        self.assertEqual((cut.mode, cut.size), ("RGBA", (120, 160)))

    def test_a_smaller_depth_is_resized_to_the_layers(self):
        figure, outline, backdrop, _ = self.pngs()
        depth = png(np.tile(np.linspace(0, 255, 40)[:, None], (1, 30)), "L")
        out = Image.open(io.BytesIO(blur_layers_png(
            figure, outline, backdrop, depth, 0.5, 0.5, 2.8, ALL)))
        self.assertEqual(out.size, (120, 160))


def nodes_of(graph: dict, class_type: str) -> list[tuple[str, dict]]:
    return [(key, node) for key, node in graph.items()
            if node["class_type"] == class_type]


def saved(graph: dict) -> list[str]:
    return [node["inputs"]["filename_prefix"] for _, node in nodes_of(graph, "SaveImage")]


def build(**overrides) -> dict:
    kwargs = dict(depth_image="depth.png", source_image=None, focus=(0.4, 0.6),
                  f_number=2.8, scope=ALL, viewfinder="off")
    kwargs.update(overrides)
    return dof_graph("figure.png", "outline.png", "backdrop.png", "dof-g", **kwargs)


class PlacedDepthTest(unittest.TestCase):
    def test_the_depth_fills_the_canvas_or_its_place_in_it(self):
        depth = np.full((10, 8), 200, np.float32)
        self.assertEqual(placed_depth(depth, 16, 20).shape, (20, 16))
        placed = placed_depth(depth, 16, 20, (0.25, -0.5, 0.75, 0.5))
        self.assertEqual(placed.shape, (20, 16))
        self.assertEqual(placed[5, 8], 200)
        self.assertEqual(placed[5, 2], 0)
        self.assertEqual(placed[15, 8], 0)


class DofGraphTest(unittest.TestCase):
    def test_the_layers_and_their_masks_feed_the_blur(self):
        graph = build()
        loads = {node["inputs"]["image"]: key for key, node in nodes_of(graph, "LoadImage")}
        [(_, blur)] = nodes_of(graph, "YukariDepthOfField")
        inputs = blur["inputs"]
        self.assertEqual(inputs["figure"], [loads["figure.png"], 0])
        self.assertEqual(inputs["figure_mask"], [loads["figure.png"], 1])
        self.assertEqual(inputs["outline_mask"], [loads["outline.png"], 1])
        self.assertEqual(inputs["backdrop"], [loads["backdrop.png"], 0])
        self.assertEqual(inputs["depth"], [loads["depth.png"], 0])
        self.assertEqual((inputs["focus_x"], inputs["focus_y"], inputs["f_number"]),
                         (0.4, 0.6, 2.8))
        self.assertNotIn(DEPTH_NODE, [node["class_type"] for node in graph.values()])
        self.assertEqual(saved(graph), ["dof-g-dof"])

    def test_the_scope_reaches_the_node(self):
        graph = build(scope={"figure": True, "outline": False, "backdrop": True})
        [(_, blur)] = nodes_of(graph, "YukariDepthOfField")
        self.assertEqual((blur["inputs"]["blur_figure"], blur["inputs"]["blur_outline"],
                          blur["inputs"]["blur_backdrop"]), (True, False, True))

    def test_without_a_stored_depth_it_is_built_from_the_source_and_saved(self):
        graph = dof_graph("figure.png", "outline.png", None, "dof-g",
                          depth_image=None, source_image="source.png",
                          focus=(0.5, 0.5), f_number=2.8, scope=ALL, viewfinder="off")
        [(depth_id, depth)] = nodes_of(graph, DEPTH_NODE)
        loads = {node["inputs"]["image"]: key for key, node in nodes_of(graph, "LoadImage")}
        self.assertEqual(depth["inputs"]["image"], [loads["source.png"], 0])
        self.assertEqual(saved(graph), ["dof-g-depth", "dof-g-dof"])
        [(_, blur)] = nodes_of(graph, "YukariDepthOfField")
        self.assertNotIn("backdrop", blur["inputs"])
        self.assertEqual(blur["inputs"]["depth"], [depth_id, 0])

    def test_viewfinder_on_overlays_the_single_picture(self):
        graph = build(viewfinder="on")
        [(view_id, _)] = nodes_of(graph, "YukariViewfinder")
        [(_, save)] = nodes_of(graph, "SaveImage")
        self.assertEqual(save["inputs"]["images"], [view_id, 0])
        self.assertEqual(saved(graph), ["dof-g-dof"])

    def test_viewfinder_both_saves_two_pictures(self):
        graph = build(viewfinder="both")
        [(blur_id, _)] = nodes_of(graph, "YukariDepthOfField")
        [(view_id, _)] = nodes_of(graph, "YukariViewfinder")
        images = {node["inputs"]["filename_prefix"]: node["inputs"]["images"]
                  for _, node in nodes_of(graph, "SaveImage")}
        self.assertEqual(images, {"dof-g-dof": [blur_id, 0],
                                  "dof-g-viewfinder": [view_id, 0]})


if __name__ == "__main__":
    unittest.main()
