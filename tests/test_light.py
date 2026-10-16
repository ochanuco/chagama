"""Scene lighting: the light pass graph, the underpaint and the tinted delivery."""

from __future__ import annotations

import copy
import io
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from chagama.domain.yukari import delivery_style
from chagama.infrastructure.comfyui.base_graph import base_roles
from chagama.infrastructure.comfyui.deliver_graph import deliver_graph
from chagama.infrastructure.comfyui.light_graph import light_graph, light_words
from chagama.infrastructure.comfyui.refinement_graph import DEPTH_NODE
from chagama.infrastructure.imaging import light
from chagama.infrastructure.imaging.delivery import band_alphas, clean_background, sticker

SOURCE = {
    "1": {"class_type": "UNETLoader", "inputs": {}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
    "5": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 1024, "height": 1640, "batch_size": 1}},
    "3": {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "seed": 5, "steps": 10, "cfg": 2,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        "latent_image": ["5", 0], "positive": ["6", 0], "negative": ["7", 0]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "p"}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "n"}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage",
          "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
    "10": {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}},
}


def find(graph: dict, class_type: str) -> tuple[str, dict]:
    matches = [(key, node) for key, node in graph.items()
               if node.get("class_type") == class_type]
    assert len(matches) == 1, class_type
    return matches[0]


def png(pixels: np.ndarray, mode: str = "RGB") -> bytes:
    output = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8), mode).save(output, "PNG")
    return output.getvalue()


class LightGraphTest(unittest.TestCase):
    def setUp(self):
        source = copy.deepcopy(SOURCE)
        self.graph = light_graph(source, base_roles(source), "up.png", "birefnet",
                                 "moon", "se", "light-g")
        self.source = source

    def test_the_sources_outputs_are_removed_and_the_source_is_untouched(self):
        saves = [node for node in self.graph.values()
                 if node["class_type"] in ("SaveImage", "PreviewImage")]
        self.assertEqual(len(saves), 1)
        self.assertEqual(saves[0]["inputs"]["filename_prefix"], "light-g")
        self.assertEqual(self.source, SOURCE)

    def test_the_lit_picture_is_resampled_with_the_source_seed(self):
        _, light_node = find(self.graph, "YukariLight")
        encode_id, encode = find(self.graph, "VAEEncode")
        self.assertEqual(encode["inputs"]["pixels"], [find(self.graph, "YukariLight")[0], 0])
        resample = next(node for node in self.graph.values()
                        if node["class_type"] == "KSampler"
                        and node["inputs"]["latent_image"] == [encode_id, 0])
        self.assertEqual(resample["inputs"]["denoise"], delivery_style.LIGHT_DENOISE)
        self.assertEqual(resample["inputs"]["seed"], 5)
        self.assertEqual(resample["inputs"]["negative"], ["7", 0])
        self.assertEqual(resample["inputs"]["model"], ["1", 0])
        self.assertEqual(light_node["inputs"]["direction"], "se")
        self.assertEqual(light_node["inputs"]["scene"], "moon")

    def test_the_positive_adds_the_scene_and_direction_words(self):
        resample = next(node for node in self.graph.values()
                        if node["class_type"] == "KSampler"
                        and node["inputs"]["denoise"] == delivery_style.LIGHT_DENOISE)
        text = self.graph[resample["inputs"]["positive"][0]]["inputs"]["text"]
        self.assertTrue(text.startswith("p, "))
        self.assertIn("(moonlight:1.15)", text)
        self.assertIn("(light from lower right:1.1)", text)
        self.assertEqual(text, "p, " + light_words("moon", "se"))

    def test_the_underpaint_reads_depth_and_matte_of_the_uploaded_picture(self):
        load_id, load = find(self.graph, "LoadImage")
        depth_id, depth = find(self.graph, DEPTH_NODE)
        _, light_node = find(self.graph, "YukariLight")
        self.assertEqual(load["inputs"]["image"], "up.png")
        self.assertEqual(depth["inputs"]["image"], [load_id, 0])
        self.assertEqual(light_node["inputs"]["image"], [load_id, 0])
        self.assertEqual(light_node["inputs"]["depth"], [depth_id, 0])
        remove_id, _ = find(self.graph, "RemoveBackground")
        self.assertEqual(light_node["inputs"]["matte"], [remove_id, 0])


class DeliverGraphTest(unittest.TestCase):
    def deliver(self, **kwargs) -> dict:
        graph = deliver_graph(
            "src.png", "birefnet", "fin", repin=True, keep_scene=False,
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
