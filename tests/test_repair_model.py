"""Tests for the `model_hooks` entry that reroutes a repair reroll onto Anima."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from chagama.domain.yukari.prompt_style import CFG, SAMPLER, SCHEDULER, STEPS
from chagama.infrastructure.comfyui.anima_graph import CLIP_NAME, VAE_NAME
from chagama.infrastructure.comfyui.repair_graph import repair_graph
from chagama.infrastructure.comfyui.repair_model import anima_model_hook

FIXTURES = Path(__file__).parent / "fixtures"
RAW = json.loads((FIXTURES / "repair-graph-raw.json").read_text())


class AnimaModelHookTest(unittest.TestCase):
    def setUp(self):
        self.graph = repair_graph(
            RAW, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=99, denoise=0.6, size=1024,
            prefix="rep-abc-s99", loras=[("feet-xl-ill.safetensors", 0.8)],
            model_hooks=[anima_model_hook("anima")])

    def _sampler(self):
        return next(node for node in self.graph.values()
                    if node["class_type"] == "KSampler")

    def test_unet_loader_names_the_word_s_checkpoint(self):
        loader = next(node for node in self.graph.values()
                      if node["class_type"] == "UNETLoader")
        self.assertEqual(loader["inputs"]["unet_name"], "sudachiAnima_v10.safetensors")

    def test_clip_and_vae_loaders_match_the_anima_graph_shape(self):
        clip_loader = next(node for node in self.graph.values()
                           if node["class_type"] == "CLIPLoader")
        self.assertEqual(clip_loader["inputs"], {"clip_name": CLIP_NAME, "type": "qwen_image"})
        vae_loader = next(node for node in self.graph.values()
                          if node["class_type"] == "VAELoader")
        self.assertEqual(vae_loader["inputs"], {"vae_name": VAE_NAME})

    def test_ksampler_bypasses_the_lora_chain_left_on_the_source_model(self):
        lora_id = next(key for key, node in self.graph.items()
                       if node["class_type"] == "LoraLoader")
        sample = self._sampler()
        self.assertNotEqual(sample["inputs"]["model"][0], lora_id)

    def test_ksampler_and_decode_use_the_swapped_vae(self):
        unet_id = next(key for key, node in self.graph.items()
                       if node["class_type"] == "UNETLoader")
        clip_id = next(key for key, node in self.graph.items()
                       if node["class_type"] == "CLIPLoader")
        vae_id = next(key for key, node in self.graph.items()
                      if node["class_type"] == "VAELoader")
        sample = self._sampler()
        self.assertEqual(sample["inputs"]["model"], [unet_id, 0])
        for key in ("positive", "negative"):
            self.assertEqual(self.graph[sample["inputs"][key][0]]["inputs"]["clip"],
                             [clip_id, 0])
        encode = self.graph[sample["inputs"]["latent_image"][0]]["inputs"]
        vae_encode = self.graph[encode["samples"][0]]
        self.assertEqual(vae_encode["inputs"]["vae"], [vae_id, 0])
        decode = next(node for node in self.graph.values()
                      if node["class_type"] == "VAEDecode"
                      and node["inputs"]["samples"][0] in self.graph
                      and self.graph[node["inputs"]["samples"][0]] is sample)
        self.assertEqual(decode["inputs"]["vae"], [vae_id, 0])

    def test_sampler_settings_come_from_yukari_prompt_style(self):
        sample = self._sampler()
        self.assertEqual(sample["inputs"]["steps"], STEPS)
        self.assertEqual(sample["inputs"]["cfg"], CFG)
        self.assertEqual(sample["inputs"]["sampler_name"], SAMPLER)
        self.assertEqual(sample["inputs"]["scheduler"], SCHEDULER)
        self.assertEqual(sample["inputs"]["seed"], 99)
        self.assertEqual(sample["inputs"]["denoise"], 0.6)

    def test_unknown_word_is_rejected(self):
        with self.assertRaises(ValueError):
            anima_model_hook("nope")


if __name__ == "__main__":
    unittest.main()
