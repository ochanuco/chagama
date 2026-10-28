from __future__ import annotations

import base64
import io
import json
import unittest

import numpy as np
from PIL import Image

from chagama.domain.generation import anima_blocks
from chagama.domain.generation.patches import apply_patches, parse_patches
from chagama.domain.yukari.recipe import render_spec
from chagama.infrastructure.comfyui import anima_graph
from chagama.application.generate import graph_prompts
from chagama.infrastructure.comfyui.base_graph import base_roles, hires_graph, source_prompts
from chagama.infrastructure.imaging import pose_skeleton

POINTS = [[0.5, 0.5, 0.9]] * 133


def _black_png():
    buffer = io.BytesIO()
    Image.new("L", (8, 16), 0).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


PICTURE = _black_png()


def pose_patch(**value):
    return {"target": "render.pose", "op": "set", "reason": "r",
            "value": {"points": POINTS, "aspect": 0.5, "strength": 1.0, **value}}


def classes(graph):
    return {node_id: node["class_type"] for node_id, node in graph.items()}


def find(graph, class_type):
    (node_id,) = [node_id for node_id, name in classes(graph).items()
                  if name == class_type]
    return node_id


class PosePatchTest(unittest.TestCase):
    def test_parses_into_the_spec(self):
        spec = apply_patches(render_spec("coffee", 7, "p"), parse_patches([pose_patch()]))
        self.assertEqual(len(spec.pose.points), 133)
        self.assertEqual((spec.pose.aspect, spec.pose.strength), (0.5, 1.0))

    def test_only_set(self):
        with self.assertRaises(ValueError):
            parse_patches([{"target": "render.pose", "op": "remove", "reason": "r"}])

    def test_rejects_a_malformed_pose(self):
        for bad in ({"points": POINTS[:17]}, {"aspect": 0}, {"strength": 2.5},
                    {"points": [[0.5, 0.5]] * 133}, {"strength": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_patches([pose_patch(**bad)])


def outline_patch(**value):
    return {"target": "render.pose", "op": "set", "reason": "r",
            "value": {"control": "silhouette", "picture": PICTURE, "aspect": 0.5,
                      "strength": 1.0, **value}}


class OutlinePatchTest(unittest.TestCase):
    def test_parses_into_the_spec(self):
        spec = apply_patches(render_spec("coffee", 7, "p"),
                             parse_patches([outline_patch()]))
        self.assertEqual((spec.pose.control, spec.pose.picture, spec.pose.points),
                         ("silhouette", PICTURE, ()))

    def test_rejects_a_malformed_outline(self):
        for bad in ({"control": "normal"}, {"picture": ""}, {"points": POINTS},
                    {"end": 0}, {"end": 1.5}, {"end": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_patches([outline_patch(**bad)])


class OutlineGraphTest(unittest.TestCase):
    def graph(self, hires=0):
        spec = render_spec("coffee", 7, "p", hires=hires)
        return anima_graph.build_graph(apply_patches(spec, parse_patches([outline_patch()])))

    def test_the_sampler_draws_through_the_lllite(self):
        graph = self.graph()
        apply_id = find(graph, "AnimaLLLiteApply")
        self.assertEqual(graph["3"]["inputs"]["model"], [apply_id, 0])
        inputs = graph[apply_id]["inputs"]
        self.assertEqual(inputs["model"], ["1", 0])
        self.assertEqual(graph[inputs["model_patch"][0]]["inputs"]["name"],
                         anima_graph.OUTLINE_ADAPTER)
        picture = graph[inputs["image"][0]]
        self.assertEqual(picture["class_type"], "YukariPoseOutline")
        self.assertEqual(picture["inputs"]["picture"], PICTURE)
        self.assertNotIn("AnimaControlApply", classes(graph).values())
        self.assertEqual(inputs["end_percent"], 1.0)

    def test_end_stops_the_lllite_partway(self):
        spec = apply_patches(render_spec("coffee", 7, "p"),
                             parse_patches([outline_patch(end=0.5)]))
        graph = anima_graph.build_graph(spec)
        self.assertEqual(graph[find(graph, "AnimaLLLiteApply")]["inputs"]["end_percent"], 0.5)

    def test_a_redraw_hires_peels_the_lllite_off(self):
        graph = self.graph()
        result = hires_graph(graph, base_roles(graph), 2048 * 2048, 0.45, "p")
        upscale_id = find(result, "LatentUpscale")
        (resample,) = [node for node in result.values()
                       if node["inputs"].get("latent_image") == [upscale_id, 0]]
        self.assertEqual(resample["inputs"]["model"], ["1", 0])


class DepthGraphTest(unittest.TestCase):
    def graph(self, hires=0):
        spec = render_spec("coffee", 7, "p", hires=hires)
        patch = outline_patch(control="depth", end=0.8)
        return anima_graph.build_graph(apply_patches(spec, parse_patches([patch])))

    def test_the_first_pass_samples_the_controlled_prompts(self):
        graph = self.graph()
        apply_id = find(graph, "ACN_AdvancedControlNetApply_v2")
        apply = graph[apply_id]["inputs"]
        self.assertEqual((apply["positive"], apply["negative"]), (["6", 0], ["7", 0]))
        self.assertEqual((apply["vae_optional"], apply["end_percent"]), (["4", 0], 0.8))
        remap = graph[apply["control_net"][0]]
        self.assertEqual(remap["class_type"], "YukariAnimaControlRemap")
        self.assertEqual(remap["inputs"]["model"], ["1", 0])
        self.assertEqual(graph[remap["inputs"]["control_net"][0]]["inputs"]["cnet"],
                         anima_graph.DEPTH_ADAPTER)
        self.assertEqual(graph[apply["image"][0]]["inputs"]["fill"], 0)
        for sampler in [node for node in graph.values()
                        if node["class_type"].startswith("KSampler")]:
            self.assertEqual((sampler["inputs"]["positive"], sampler["inputs"]["negative"]),
                             ([apply_id, 0], [apply_id, 1]))
            self.assertEqual(sampler["inputs"]["model"], ["1", 0])

    def test_the_prompts_are_read_through_the_controlnet(self):
        graph = self.graph()
        texts = (graph["6"]["inputs"]["text"], graph["7"]["inputs"]["text"])
        self.assertEqual(graph_prompts(graph), texts)
        self.assertEqual(source_prompts(graph), texts)

    def test_the_hires_pass_samples_the_plain_prompts(self):
        graph = self.graph(hires=2048)
        upscale_id = find(graph, "LatentUpscale")
        (resample,) = [node for node in graph.values()
                       if node["inputs"].get("latent_image") == [upscale_id, 0]]
        self.assertEqual((resample["inputs"]["positive"], resample["inputs"]["negative"]),
                         (["6", 0], ["7", 0]))

    def test_a_redraw_hires_peels_the_controlnet_off(self):
        graph = self.graph()
        result = hires_graph(graph, base_roles(graph), 2048 * 2048, 0.45, "p")
        upscale_id = find(result, "LatentUpscale")
        (resample,) = [node for node in result.values()
                       if node["inputs"].get("latent_image") == [upscale_id, 0]]
        self.assertEqual((resample["inputs"]["positive"], resample["inputs"]["negative"]),
                         (["6", 0], ["7", 0]))


class AnimaBlocksTest(unittest.TestCase):
    def test_the_vace_layers_land_on_the_old_blocks_of_the_2_9b(self):
        self.assertEqual(anima_blocks.remap([0, 7, 14, 21], 40), [0, 10, 20, 31])
        self.assertEqual(anima_blocks.remap([0, 7, 14, 21], 28), [0, 7, 14, 21])
        self.assertEqual(len(set(anima_blocks.MAP_28_40)), 28)

    def test_an_unknown_size_is_refused(self):
        with self.assertRaises(ValueError):
            anima_blocks.remap([0], 52)


class OutlineCanvasTest(unittest.TestCase):
    def test_fits_the_frame_with_white_around(self):
        canvas = pose_skeleton.outline(base64.b64decode(PICTURE), 0.5, 64, 32)
        self.assertEqual(canvas.shape, (32, 64, 3))
        self.assertEqual(canvas[16, 32].tolist(), [0, 0, 0])
        self.assertEqual(canvas[16, 4].tolist(), [255, 255, 255])

    def test_rejects_an_unreadable_picture(self):
        with self.assertRaises(ValueError):
            pose_skeleton.outline(b"nope", 0.5, 64, 32)

    def test_fills_around_with_the_given_grey(self):
        canvas = pose_skeleton.outline(base64.b64decode(PICTURE), 0.5, 64, 32, fill=0)
        self.assertEqual(canvas[16, 4].tolist(), [0, 0, 0])


class PoseGraphTest(unittest.TestCase):
    def graph(self, hires=0):
        spec = render_spec("coffee", 7, "p", hires=hires)
        return anima_graph.build_graph(apply_patches(spec, parse_patches([pose_patch()])))

    def test_the_sampler_draws_through_the_adapter(self):
        graph = self.graph()
        apply_id = find(graph, "AnimaControlApply")
        self.assertEqual(graph["3"]["inputs"]["model"], [apply_id, 0])
        lora_id = graph[apply_id]["inputs"]["model"][0]
        self.assertEqual(graph[lora_id]["inputs"]["lora_name"], anima_graph.POSE_ADAPTER)
        self.assertEqual(graph[lora_id]["inputs"]["model"], ["1", 0])
        encode_id = graph[apply_id]["inputs"]["control_latent"][0]
        skeleton = graph[graph[encode_id]["inputs"]["pixels"][0]]
        self.assertEqual(skeleton["class_type"], "YukariPoseSkeleton")
        self.assertEqual((skeleton["inputs"]["width"], skeleton["inputs"]["height"]),
                         (graph["5"]["inputs"]["width"], graph["5"]["inputs"]["height"]))
        self.assertEqual(json.loads(skeleton["inputs"]["points"]), POINTS)

    def test_the_hires_pass_draws_without_the_adapter(self):
        graph = self.graph(hires=2048)
        upscale_id = find(graph, "LatentUpscale")
        (resample,) = [node for node in graph.values()
                       if node["inputs"].get("latent_image") == [upscale_id, 0]]
        self.assertEqual(resample["inputs"]["model"], ["1", 0])

    def test_a_redraw_hires_peels_the_adapter_off(self):
        graph = self.graph()
        result = hires_graph(graph, base_roles(graph), 2048 * 2048, 0.45, "p")
        upscale_id = find(result, "LatentUpscale")
        (resample,) = [node for node in result.values()
                       if node["inputs"].get("latent_image") == [upscale_id, 0]]
        self.assertEqual(resample["inputs"]["model"], ["1", 0])

    def test_no_pose_or_strength_zero_adds_nothing(self):
        unposed = render_spec("coffee", 7, "p")
        for spec in (unposed, apply_patches(unposed, parse_patches([pose_patch(strength=0)]))):
            with self.subTest(pose=spec.pose):
                self.assertEqual(anima_graph.build_graph(spec),
                                 anima_graph.build_graph(unposed))


class SkeletonTest(unittest.TestCase):
    def test_the_source_frame_is_fitted_whole_and_centred(self):
        self.assertEqual(pose_skeleton.frame(1000, 1000, 0.5), (250.0, 0.0, 500.0, 1000.0))
        self.assertEqual(pose_skeleton.frame(1000, 1000, 2.0), (0.0, 250.0, 1000.0, 500.0))

    def test_draws_rgb_on_black_and_skips_unsure_points(self):
        points = [[0.0, 0.0, 0.0]] * 133
        points[0] = [0.5, 0.5, 0.9]
        image = pose_skeleton.draw(points, 1.0, 64, 32)
        self.assertEqual(image.shape, (32, 64, 3))
        self.assertEqual(tuple(image[16, 32]), (255, 153, 51))
        self.assertEqual(int(np.count_nonzero(image[:4, :4])), 0)

    def test_pads_black_to_whole_patches_at_the_bottom_and_right(self):
        skeleton = np.full((1640, 1024, 3), 7, np.uint8)
        padded = pose_skeleton.patch_padded(skeleton)
        self.assertEqual(padded.shape, (1648, 1024, 3))
        self.assertTrue((padded[:1640] == 7).all())
        self.assertFalse(padded[1640:].any())

    def test_needs_every_point(self):
        with self.assertRaises(ValueError):
            pose_skeleton.draw(POINTS[:17], 1.0, 64, 64)
