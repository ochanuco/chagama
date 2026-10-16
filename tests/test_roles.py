"""label_roles: stable node role names in `_meta.title`."""

from __future__ import annotations

import copy
import unittest

from chagama.domain.yukari import delivery_style
from chagama.infrastructure.comfyui.base_graph import base_roles, hires_graph
from chagama.infrastructure.comfyui.deliver_graph import deliver_graph
from chagama.infrastructure.comfyui.roles import label_roles, snake_case

from test_hires_graph import guided_graph, plain_graph
from test_deliver_application import GRAPH_KWARGS


def titles(graph: dict) -> dict:
    return {node_id: node["_meta"]["title"] for node_id, node in graph.items()}


def stripped(graph: dict) -> dict:
    return {node_id: {key: value for key, value in node.items() if key != "_meta"}
            for node_id, node in graph.items()}


class LabelRolesTest(unittest.TestCase):
    def test_base_graph_roles(self):
        graph = plain_graph()
        graph["2"] = {"class_type": "CLIPLoader", "inputs": {}}
        graph["4"] = {"class_type": "VAELoader", "inputs": {}}
        labelled = titles(label_roles(graph))
        self.assertEqual(labelled["1"], "unet_loader")
        self.assertEqual(labelled["2"], "clip_loader")
        self.assertEqual(labelled["4"], "vae_loader")
        self.assertEqual(labelled["5"], "empty_latent_image")
        self.assertEqual(labelled["6"], "positive_prompt")
        self.assertEqual(labelled["7"], "negative_prompt")
        self.assertEqual(labelled["3"], "base_sampler")
        self.assertEqual(labelled["8"], "vae_decode")
        self.assertEqual(labelled["9"], "save_image")

    def test_hires_graph_names_the_upscale_and_second_sampler(self):
        graph = plain_graph()
        result = hires_graph(graph, base_roles(graph), 1280 * 2048, 0.35, "hires-x")
        labelled = titles(label_roles(result))
        self.assertEqual(labelled["3"], "base_sampler")
        self.assertEqual(labelled["10"], "hires_upscale")
        self.assertEqual(labelled["11"], "hires_sampler")
        self.assertEqual(labelled["6"], "positive_prompt")

    def test_two_stage_sampler_chain(self):
        labelled = titles(label_roles(guided_graph()))
        self.assertEqual(labelled["10"], "base_sampler")
        self.assertEqual(labelled["3"], "base_sampler_stage2")

    def test_guided_hires_keeps_stage_names_per_group(self):
        graph = guided_graph()
        result = hires_graph(graph, base_roles(graph), 1280 * 2048, 0.35, "hires-x")
        labelled = titles(label_roles(result))
        self.assertEqual(labelled["10"], "base_sampler")
        self.assertEqual(labelled["3"], "base_sampler_stage2")
        self.assertIn("hires_sampler", labelled.values())

    def test_repeated_classes_get_numbered_suffixes_in_topological_order(self):
        graph = plain_graph()
        graph["20"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0]}}
        graph["21"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["20", 0]}}
        graph["3"]["inputs"]["model"] = ["21", 0]
        labelled = titles(label_roles(graph))
        self.assertEqual(labelled["20"], "lora_loader")
        self.assertEqual(labelled["21"], "lora_loader_2")

    def test_deliver_graph_falls_back_to_snake_case_class(self):
        graph = deliver_graph("src.png", delivery_style.MATTE_MODEL, "dlv-g", **GRAPH_KWARGS)
        labelled = titles(label_roles(graph))
        self.assertEqual(set(labelled), set(graph))
        saves = [labelled[key] for key, node in graph.items()
                 if node["class_type"] == "SaveImage"]
        self.assertEqual(saves[0], "save_image")
        self.assertEqual(len(set(labelled.values())), len(labelled))
        load = next(key for key, node in graph.items() if node["class_type"] == "LoadImage")
        self.assertTrue(labelled[load].startswith("load_image"))

    def test_existing_titles_win_and_nothing_else_changes(self):
        graph = plain_graph()
        graph["3"]["_meta"] = {"title": "mine", "extra": 1}
        before = copy.deepcopy(graph)
        result = label_roles(graph)
        self.assertEqual(graph, before)
        self.assertEqual(result["3"]["_meta"], {"title": "mine", "extra": 1})
        self.assertEqual(stripped(result), stripped(before))

    def test_snake_case(self):
        self.assertEqual(snake_case("VAEDecode"), "vae_decode")
        self.assertEqual(snake_case("KSampler"), "k_sampler")
        self.assertEqual(snake_case("CLIPTextEncode"), "clip_text_encode")


if __name__ == "__main__":
    unittest.main()
