"""hires_graph: the latent-upscale + same-seed pass spliced into a stored graph."""

from __future__ import annotations

import copy
import unittest

from chagama.infrastructure.comfyui.base_graph import base_roles, hires_graph

PORTRAIT_2048 = 1280 * 2048


def plain_graph(width=1024, height=1640):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
        "3": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0], "seed": 77, "steps": 20, "cfg": 4.5,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
    }


def guided_graph():
    graph = plain_graph()
    graph["10"] = {"class_type": "KSamplerAdvanced", "inputs": {
        "model": ["1", 0], "positive": ["6", 0], "negative": ["7", 0],
        "latent_image": ["5", 0], "add_noise": "enable", "noise_seed": 99,
        "steps": 10, "cfg": 2.0, "sampler_name": "euler",
        "scheduler": "simple", "start_at_step": 0, "end_at_step": 4,
        "return_with_leftover_noise": "enable"}}
    graph["3"] = {"class_type": "KSamplerAdvanced", "inputs": {
        "model": ["1", 0], "positive": ["6", 0], "negative": ["7", 0],
        "latent_image": ["10", 0], "add_noise": "disable", "noise_seed": 99,
        "steps": 10, "cfg": 1.0, "sampler_name": "euler",
        "scheduler": "simple", "start_at_step": 4, "end_at_step": 10000,
        "return_with_leftover_noise": "disable"}}
    return graph


class HiresGraphTest(unittest.TestCase):
    def test_wires_an_upscale_and_a_second_pass_before_the_decode(self):
        graph = plain_graph()
        result = hires_graph(graph, base_roles(graph), PORTRAIT_2048, 0.35, "hires-x")
        self.assertEqual(result["10"], {"class_type": "LatentUpscale", "inputs": {
            "samples": ["3", 0], "upscale_method": "bicubic",
            "width": 1280, "height": 2048, "crop": "disabled"}})
        self.assertEqual(result["11"], {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["10", 0], "seed": 77, "steps": 20, "cfg": 4.5,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 0.35}})
        self.assertEqual(result["8"]["inputs"]["samples"], ["11", 0])
        self.assertEqual(result["9"]["inputs"]["filename_prefix"], "hires-x")
        self.assertEqual(result["5"], graph["5"])

    def test_target_sides_are_rounded_to_multiples_of_eight(self):
        graph = plain_graph(1000, 1500)
        result = hires_graph(graph, base_roles(graph), PORTRAIT_2048, 0.35, "p")
        upscale = result["10"]["inputs"]
        self.assertEqual((upscale["width"], upscale["height"]), (1320, 1984))

    def test_square_canvas_gets_the_same_area_not_the_same_long_side(self):
        graph = plain_graph(1280, 1280)
        result = hires_graph(graph, base_roles(graph), PORTRAIT_2048, 0.35, "p")
        upscale = result["10"]["inputs"]
        self.assertEqual((upscale["width"], upscale["height"]), (1616, 1616))

    def test_landscape_canvas_scales_the_width(self):
        graph = plain_graph(1640, 1024)
        result = hires_graph(graph, base_roles(graph), PORTRAIT_2048, 0.35, "p")
        upscale = result["10"]["inputs"]
        self.assertEqual((upscale["width"], upscale["height"]), (2048, 1280))

    def test_guided_graph_takes_seed_and_cfg_from_the_first_pass(self):
        graph = guided_graph()
        result = hires_graph(graph, base_roles(graph), PORTRAIT_2048, 0.5, "p")
        resample = result["12"]
        self.assertEqual(resample["class_type"], "KSampler")
        self.assertEqual(resample["inputs"]["seed"], 99)
        self.assertEqual(resample["inputs"]["cfg"], 2.0)
        self.assertEqual(resample["inputs"]["steps"], 10)
        self.assertEqual(result["11"]["inputs"]["samples"], ["3", 0])
        self.assertEqual(result["8"]["inputs"]["samples"], ["12", 0])

    def test_rejects_a_graph_that_is_already_hires(self):
        graph = plain_graph()
        graph["10"] = {"class_type": "LatentUpscale", "inputs": {}}
        with self.assertRaisesRegex(ValueError, "hires 済み"):
            hires_graph(graph, base_roles(graph), PORTRAIT_2048, 0.35, "p")

    def test_does_not_mutate_its_input(self):
        graph = plain_graph()
        snapshot = copy.deepcopy(graph)
        hires_graph(graph, base_roles(graph), PORTRAIT_2048, 0.35, "p")
        self.assertEqual(graph, snapshot)

    def test_all_node_ids_stay_decimal_strings(self):
        graph = plain_graph()
        result = hires_graph(graph, base_roles(graph), PORTRAIT_2048, 0.35, "p")
        self.assertTrue(all(key.isdecimal() for key in result))


if __name__ == "__main__":
    unittest.main()
