"""Scene lighting: the underpaint and the tinted delivery."""

from __future__ import annotations

import io
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from chagama.infrastructure.comfyui.deliver_graph import deliver_graph
from chagama.infrastructure.imaging import light
from chagama.infrastructure.imaging.delivery import band_alphas, clean_background, sticker

def find(graph: dict, class_type: str) -> tuple[str, dict]:
    matches = [(key, node) for key, node in graph.items()
               if node.get("class_type") == class_type]
    assert len(matches) == 1, class_type
    return matches[0]


def png(pixels: np.ndarray, mode: str = "RGB") -> bytes:
    output = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8), mode).save(output, "PNG")
    return output.getvalue()


class DeliverGraphTest(unittest.TestCase):
    def deliver(self, **kwargs) -> dict:
        graph = deliver_graph(
            "src.png", "birefnet", "fin", keep_scene=False,
            transparent=False, backdrop=None, stroke_light=None,
            outlines=[],
            **{"light_scene": None, "light_from": None, **kwargs})
        return find(graph, "YukariDeliver")[1]["inputs"]

    def test_the_delivery_gets_the_scene(self):
        inputs = self.deliver(light_scene="moon", light_from="sw")
        self.assertEqual(inputs["light_scene"], "moon")
        self.assertEqual(inputs["light_from"], "sw")

    def test_without_a_scene_the_delivery_is_unchanged(self):
        self.assertNotIn("light_scene", self.deliver())


def figure_image(colour=(128, 128, 128), size=128):
    rgb = np.full((size, size, 3), 40, dtype=np.uint8)
    matte = np.zeros((size, size), dtype=np.uint8)
    matte[16:112, 16:112] = 255
    rgb[16:112, 16:112] = colour
    return rgb, matte, np.zeros((size, size), dtype=np.uint8)


def underpaint(scene: str, direction: str = "nw", colour=(128, 128, 128)):
    rgb, matte, depth = figure_image(colour)
    data = light.underpaint_png(png(rgb), png(depth, "L"), png(matte, "L"),
                                direction, scene)
    return rgb, np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(int)


class UnderpaintTest(unittest.TestCase):
    def test_pixels_outside_the_matte_are_unchanged(self):
        rgb, lit = underpaint("sunset")
        np.testing.assert_array_equal(lit[:16], rgb[:16])
        np.testing.assert_array_equal(lit[:, 112:], rgb[:, 112:])

    def test_the_side_facing_the_light_is_brighter_than_the_far_side(self):
        _, lit = underpaint("sunset", "nw")
        near = lit[28:48, 28:48].mean()
        far = lit[80:100, 80:100].mean()
        self.assertGreater(near, far)
        _, flipped = underpaint("sunset", "se")
        self.assertLess(flipped[28:48, 28:48].mean(), flipped[80:100, 80:100].mean())

    def test_a_depth_map_of_another_size_is_resized_to_the_picture(self):
        rgb, matte, _ = figure_image()
        small = np.zeros((32, 32), dtype=np.uint8)
        data = light.underpaint_png(png(rgb), png(small, "L"), png(matte, "L"),
                                    "nw", "moon")
        self.assertEqual(Image.open(io.BytesIO(data)).size, (128, 128))

    def test_skin_keeps_more_of_its_colour_under_moonlight(self):
        skin = (235, 190, 170)
        rgb, _, _ = figure_image(skin)
        figure = np.zeros(rgb.shape[:2], dtype=bool)
        figure[16:112, 16:112] = True
        shifts = {}
        for name, mask in (("plain", np.zeros_like(figure)), ("skin", figure)):
            with mock.patch.object(light.palette, "skin_mask", return_value=mask):
                _, lit = underpaint("moon", colour=skin)
            shifts[name] = np.abs(lit[16:112, 16:112] - np.array(skin)).mean()
        self.assertLess(shifts["skin"], shifts["plain"])

    def test_sunset_does_not_protect_skin(self):
        skin = (235, 190, 170)
        figure = np.ones((128, 128), dtype=bool)
        with mock.patch.object(light.palette, "skin_mask", return_value=figure):
            _, protected = underpaint("sunset", colour=skin)
        with mock.patch.object(light.palette, "skin_mask",
                               return_value=np.zeros_like(figure)):
            _, plain = underpaint("sunset", colour=skin)
        np.testing.assert_array_equal(protected, plain)


class SceneDeliveryTest(unittest.TestCase):
    def scene_pixels(self):
        pixels = np.full((256, 256, 3), 255, dtype=np.uint8)
        soft = np.zeros((256, 256), dtype=np.uint8)
        soft[90:170, 90:170] = 255
        pixels[90:170, 90:170] = (200, 180, 190)
        return pixels, soft

    def test_the_backdrop_is_tinted_and_brighter_toward_the_light(self):
        pixels, soft = self.scene_pixels()
        plain, _ = clean_background(png(pixels), png(soft, "L"), light="nw",
                                    backdrop="#808080")
        lit, _ = clean_background(png(pixels), png(soft, "L"), light="nw",
                                  backdrop="#808080", scene="sunset")
        plain = np.array(Image.open(io.BytesIO(plain))).astype(float)
        lit = np.array(Image.open(io.BytesIO(lit))).astype(float)
        corner = lit[4:20, 4:20].mean(axis=(0, 1))
        self.assertGreater(corner[0], corner[2])
        self.assertGreater(corner[0], plain[4:20, 4:20, 0].mean())
        far = lit[-20:-4, -20:-4].mean(axis=(0, 1))
        self.assertGreater(corner.sum(), far.sum())

    def test_without_a_scene_the_delivery_is_what_it_was(self):
        pixels, soft = self.scene_pixels()
        first = clean_background(png(pixels), png(soft, "L"), light="nw",
                                 backdrop="dots")
        second = clean_background(png(pixels), png(soft, "L"), light="nw",
                                  backdrop="dots", scene=None)
        self.assertEqual(first, second)

    def test_the_drop_shadow_takes_the_scenes_cast_colour(self):
        size = 160
        px = np.full((size, size, 3), 200.0)
        figure = np.zeros((size, size), dtype=bool)
        figure[60:100, 60:100] = True
        coverage = figure.astype(float)
        backdrop = np.full((size, size, 3), 200.0)
        without = sticker(px, figure, coverage, backdrop, "nw", shadow=False)
        neutral = sticker(px, figure, coverage, backdrop, "nw", shadow=True)
        moon = sticker(px, figure, coverage, backdrop, "nw", shadow=True,
                       scene="moon")
        bands = sum(band_alphas(figure, light="nw"))
        shadow = (neutral != without).any(axis=2) & (bands == 0)
        self.assertTrue(shadow.any())
        self.assertTrue(np.allclose(neutral[shadow][:, 0], neutral[shadow][:, 2]))
        self.assertTrue((moon[shadow][:, 2] > moon[shadow][:, 0]).all())
