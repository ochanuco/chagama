"""Tests pinning `control_hook`'s graph shape on the repair reroll seam."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from chagama.infrastructure.comfyui.repair_controlnet import control_hook
from chagama.infrastructure.comfyui.repair_graph import repair_graph

FIXTURES = Path(__file__).parent / "fixtures"
RAW = json.loads((FIXTURES / "repair-graph-raw.json").read_text())


class ControlHookTest(unittest.TestCase):
    def _sampler(self, graph):
        return next(node for node in graph.values()
                    if node["class_type"] == "KSampler")

    def _build(self):
        hook = control_hook("lineart", 0.75, "ref.png")
        graph = repair_graph(
            RAW, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=1, denoise=0.6, size=1024,
            prefix="rep-abc-s1", conditioning_hooks=[hook])
        return graph

    def test_unknown_word_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nope"):
            control_hook("nope", 0.8, "ref.png")

    def test_loads_the_reference_image(self):
        graph = self._build()
        loaders = [node for node in graph.values()
                  if node["class_type"] == "LoadImage"
                  and node["inputs"]["image"] == "ref.png"]
        self.assertEqual(len(loaders), 1)

    def test_loads_the_controlnet_model(self):
        graph = self._build()
        loaders = [node for node in graph.values()
                  if node["class_type"] == "ControlNetLoader"]
        self.assertEqual(len(loaders), 1)
        self.assertEqual(loaders[0]["inputs"]["control_net_name"],
                         "noob-lineart-anime-fp16.safetensors")

    def test_applies_the_controlnet_at_the_given_strength(self):
        graph = self._build()
        applies = [node for node in graph.values()
                  if node["class_type"] == "ControlNetApplyAdvanced"]
        self.assertEqual(len(applies), 1)
        self.assertEqual(applies[0]["inputs"]["strength"], 0.75)

    def test_sampler_ends_up_wired_through_the_apply_node(self):
        graph = self._build()
        apply_id = next(key for key, node in graph.items()
                        if node["class_type"] == "ControlNetApplyAdvanced")
        sample = self._sampler(graph)
        self.assertEqual(sample["inputs"]["positive"], [apply_id, 0])
        self.assertEqual(sample["inputs"]["negative"], [apply_id, 1])

    def test_apply_image_input_is_the_loaded_reference_not_the_crop(self):
        graph = self._build()
        apply = next(node for node in graph.values()
                    if node["class_type"] == "ControlNetApplyAdvanced")
        image_node = graph[apply["inputs"]["image"][0]]
        self.assertEqual(image_node["class_type"], "LoadImage")
        self.assertEqual(image_node["inputs"]["image"], "ref.png")


if __name__ == "__main__":
    unittest.main()
