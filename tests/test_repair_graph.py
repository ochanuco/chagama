"""Tests pinning repair_graph's transformation on the two reference graphs.

`repair-graph-raw.json` is a plain generation (one SaveImage). `repair-graph-
redraw.json` is a redraw output graph: a second sampling pass on a bigger
canvas whose VAEDecode feeds the one SaveImage. Both were captured from a
real worker submission; see AGENTS.md for how to regenerate
fixtures like these if the node packs' contracts ever change.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from chagama.domain.yukari.recipe import render_spec
from chagama.infrastructure.comfyui import anima_graph
from chagama.infrastructure.comfyui.base_graph import base_roles, source_prompts
from chagama.infrastructure.comfyui.repair_graph import (
    masked_redraw_graph,
    repair_graph,
)

FIXTURES = Path(__file__).parent / "fixtures"
RAW = json.loads((FIXTURES / "repair-graph-raw.json").read_text())
REDRAW = json.loads((FIXTURES / "repair-graph-redraw.json").read_text())

# A layerdiffuse raw: node 12 LayeredDiffusionApply wraps the model, 13
# LayeredDiffusionDecode/14 InvertMask/15 JoinImageWithAlpha compute the RGBA
# picture from VAEDecode 8, and SaveImage 9 reads the Join. Written out by
# hand -- no recipe builds this shape any more, but repair still has to
# handle a historical generation that carries it.
LAYERDIFFUSE = {
    "4": {"class_type": "DiffusersLoader",
         "inputs": {"model_path": "hassaku-il-v22"}},
    "5": {"class_type": "EmptyLatentImage",
         "inputs": {"batch_size": 1, "width": 832, "height": 1664}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "p"}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "n"}},
    "3": {"class_type": "KSampler", "inputs": {
        "model": ["12", 0], "positive": ["6", 0], "negative": ["7", 0],
        "latent_image": ["5", 0], "seed": 7, "steps": 30, "cfg": 5.0,
        "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage",
         "inputs": {"images": ["15", 0], "filename_prefix": "ld-src"}},
    "12": {"class_type": "LayeredDiffusionApply", "inputs": {
        "model": ["4", 0], "config": "SDXL, Conv Injection", "weight": 1.0}},
    "13": {"class_type": "LayeredDiffusionDecode", "inputs": {
        "samples": ["3", 0], "images": ["8", 0],
        "sd_version": "SDXL", "sub_batch_size": 16}},
    "14": {"class_type": "InvertMask", "inputs": {"mask": ["13", 1]}},
    "15": {"class_type": "JoinImageWithAlpha", "inputs": {
        "image": ["13", 0], "alpha": ["14", 0]}},
}


class SourcePromptsTest(unittest.TestCase):
    def test_raw_graph_reads_the_only_prompt_pair(self):
        positive, negative = source_prompts(RAW)
        self.assertEqual(positive, RAW["6"]["inputs"]["text"])
        self.assertEqual(negative, RAW["7"]["inputs"]["text"])

    def test_redraw_graph_reads_the_redraw_prompt_pair_not_pass1(self):
        # The fixture's pass-1 and redraw prompts happen to carry the same
        # text; what matters is which node source_prompts reads them from.
        positive, negative = source_prompts(REDRAW)
        self.assertEqual(positive, REDRAW["15"]["inputs"]["text"])
        self.assertEqual(negative, REDRAW["16"]["inputs"]["text"])


class RepairGraphRawTest(unittest.TestCase):
    def setUp(self):
        self.graph = repair_graph(
            RAW, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=99, denoise=0.6, size=1024,
            prefix="rep-abc-s99")

    def test_keeps_only_the_shared_loaders(self):
        kept_original = {key for key in self.graph if key in RAW}
        self.assertEqual(kept_original, {"4", "10", "9"})

    def test_kept_loader_nodes_are_untouched(self):
        self.assertEqual(self.graph["4"], RAW["4"])
        self.assertEqual(self.graph["10"], RAW["10"])

    def test_the_only_saveimage_is_rewired_to_the_stitch_output_and_prefixed(self):
        save = self.graph["9"]
        self.assertEqual(save["class_type"], "SaveImage")
        self.assertEqual(save["inputs"]["filename_prefix"], "rep-abc-s99")
        stitch_id = save["inputs"]["images"][0]
        self.assertEqual(self.graph[stitch_id]["class_type"], "InpaintStitchImproved")

    def test_no_extra_saveimage_is_added_when_one_directly_consumes_the_stitch(self):
        saves = [node for node in self.graph.values()
                if node["class_type"] == "SaveImage"]
        self.assertEqual(len(saves), 1)

    def test_new_subgraph_wires_crop_encode_sample_decode_stitch(self):
        crop_id, crop = next(
            (key, node) for key, node in self.graph.items()
            if node["class_type"] == "InpaintCropImproved")
        self.assertEqual(crop["inputs"]["output_target_width"], 1024)
        self.assertEqual(crop["inputs"]["output_target_height"], 1024)
        self.assertEqual(crop["inputs"]["output_padding"], "32")
        self.assertIs(crop["inputs"]["preresize"], False)
        self.assertIs(crop["inputs"]["mask_fill_holes"], True)

        load_image = self.graph[crop["inputs"]["image"][0]]
        self.assertEqual(load_image["class_type"], "LoadImage")
        self.assertEqual(load_image["inputs"]["image"], "src.png")

        to_mask = self.graph[crop["inputs"]["mask"][0]]
        self.assertEqual(to_mask["class_type"], "ImageToMask")
        self.assertEqual(to_mask["inputs"]["channel"], "red")
        load_mask = self.graph[to_mask["inputs"]["image"][0]]
        self.assertEqual(load_mask["inputs"]["image"], "mask.png")

        encode_id, encode = next(
            (key, node) for key, node in self.graph.items()
            if node["class_type"] == "VAEEncode")
        self.assertEqual(encode["inputs"]["vae"], ["4", 2])
        self.assertEqual(encode["inputs"]["pixels"], [crop_id, 1])

        noise_mask_id, noise_mask = next(
            (key, node) for key, node in self.graph.items()
            if node["class_type"] == "SetLatentNoiseMask")
        self.assertEqual(noise_mask["inputs"]["samples"], [encode_id, 0])
        self.assertEqual(noise_mask["inputs"]["mask"], [crop_id, 2])

        sample_id, sample = next(
            (key, node) for key, node in self.graph.items()
            if node["class_type"] == "KSampler")
        self.assertEqual(sample["inputs"]["model"], ["10", 0])
        self.assertEqual(sample["inputs"]["latent_image"], [noise_mask_id, 0])
        self.assertEqual(sample["inputs"]["seed"], 99)
        self.assertEqual(sample["inputs"]["denoise"], 0.6)
        self.assertEqual(sample["inputs"]["steps"], RAW["3"]["inputs"]["steps"])
        self.assertEqual(sample["inputs"]["cfg"], RAW["3"]["inputs"]["cfg"])
        self.assertEqual(sample["inputs"]["sampler_name"],
                         RAW["3"]["inputs"]["sampler_name"])
        self.assertEqual(sample["inputs"]["scheduler"],
                         RAW["3"]["inputs"]["scheduler"])

        positive_node = self.graph[sample["inputs"]["positive"][0]]
        negative_node = self.graph[sample["inputs"]["negative"][0]]
        self.assertEqual(positive_node["inputs"]["text"], "p")
        self.assertEqual(positive_node["inputs"]["clip"], ["10", 1])
        self.assertEqual(negative_node["inputs"]["text"], "n")

        # The raw fixture's own VAEDecode ("8") is dropped along with its
        # sampler, so the repair's own decode is the only one left.
        decode_id, decode = next(
            (key, node) for key, node in self.graph.items()
            if node["class_type"] == "VAEDecode")
        self.assertEqual(decode["inputs"]["samples"], [sample_id, 0])
        self.assertEqual(decode["inputs"]["vae"], ["4", 2])

        stitch_id = self.graph["9"]["inputs"]["images"][0]
        stitch_node = self.graph[stitch_id]
        self.assertEqual(stitch_node["class_type"], "InpaintStitchImproved")
        self.assertEqual(stitch_node["inputs"]["stitcher"], [crop_id, 0])
        self.assertEqual(stitch_node["inputs"]["inpainted_image"], [decode_id, 0])

    def test_does_not_mutate_the_source_graph(self):
        before = json.dumps(RAW, sort_keys=True)
        repair_graph(RAW, image_name="x.png", mask_name="m.png",
                    positive="p", negative="n", seed=1, denoise=0.5,
                    size=512, prefix="rep")
        self.assertEqual(json.dumps(RAW, sort_keys=True), before)


class RepairGraphRedrawTest(unittest.TestCase):
    def setUp(self):
        self.graph = repair_graph(
            REDRAW, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=42, denoise=0.6, size=1024,
            prefix="rep-xyz-s42")

    def test_keeps_only_the_shared_loaders_plus_the_save(self):
        kept_original = {key for key in self.graph if key in REDRAW}
        self.assertEqual(kept_original, {"4", "10", "9"})

    def test_pass1_and_old_redraw_sampling_nodes_are_dropped(self):
        dropped = {"3", "5", "6", "7", "8", "11", "13", "15", "16"}
        self.assertFalse(dropped & set(self.graph))

    def test_saveimage_is_rewired_to_the_stitch_and_gets_the_bare_prefix(self):
        save = self.graph["9"]
        self.assertEqual(save["inputs"]["filename_prefix"], "rep-xyz-s42")
        stitch_id = save["inputs"]["images"][0]
        self.assertEqual(self.graph[stitch_id]["class_type"], "InpaintStitchImproved")

    def test_only_one_saveimage_survives(self):
        saves = [node for node in self.graph.values()
                if node["class_type"] == "SaveImage"]
        self.assertEqual(len(saves), 1)

    def test_sampler_settings_come_from_the_redraw_pass_not_pass1(self):
        sample = next(node for node in self.graph.values()
                     if node["class_type"] == "KSampler")
        self.assertEqual(sample["inputs"]["sampler_name"],
                         REDRAW["13"]["inputs"]["sampler_name"])
        self.assertEqual(sample["inputs"]["scheduler"],
                         REDRAW["13"]["inputs"]["scheduler"])
        self.assertEqual(sample["inputs"]["steps"], REDRAW["13"]["inputs"]["steps"])
        self.assertEqual(sample["inputs"]["cfg"], REDRAW["13"]["inputs"]["cfg"])
        self.assertNotEqual(sample["inputs"]["sampler_name"],
                            REDRAW["3"]["inputs"]["sampler_name"])

    def test_vae_ref_matches_the_redraw_decode(self):
        encode = next(node for node in self.graph.values()
                     if node["class_type"] == "VAEEncode")
        self.assertEqual(encode["inputs"]["vae"], REDRAW["14"]["inputs"]["vae"])


class RepairGraphLayerDiffuseTest(unittest.TestCase):
    def setUp(self):
        self.graph = repair_graph(
            LAYERDIFFUSE, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=11, denoise=0.6, size=1024,
            prefix="rep-ld-s11")

    def test_ld_decode_and_invert_mask_are_dropped(self):
        self.assertNotIn("13", self.graph)
        self.assertNotIn("14", self.graph)

    def test_pass1_and_apply_nodes_are_not_pulled_back_in(self):
        dropped = {"3", "5", "6", "7", "8", "12"}
        self.assertFalse(dropped & set(self.graph))

    def test_join_is_rewired_to_the_stitch_and_the_staged_masks_alpha(self):
        join = self.graph["15"]
        self.assertEqual(join["class_type"], "JoinImageWithAlpha")
        stitch_id = join["inputs"]["image"][0]
        self.assertEqual(self.graph[stitch_id]["class_type"], "InpaintStitchImproved")
        alpha_ref = join["inputs"]["alpha"]
        self.assertEqual(alpha_ref[1], 1)
        load_image = self.graph[alpha_ref[0]]
        self.assertEqual(load_image["class_type"], "LoadImage")
        self.assertEqual(load_image["inputs"]["image"], "src.png")

    def test_exactly_one_saveimage_fed_by_the_join_with_the_prefix(self):
        saves = [(key, node) for key, node in self.graph.items()
                if node["class_type"] == "SaveImage"]
        self.assertEqual(len(saves), 1)
        save_id, save = saves[0]
        self.assertEqual(save_id, "9")
        self.assertEqual(save["inputs"]["filename_prefix"], "rep-ld-s11")
        self.assertEqual(save["inputs"]["images"], ["15", 0])

    def test_base_roles_resolves_the_single_save_as_stitched(self):
        roles = base_roles(self.graph)
        self.assertEqual(roles.save_id, "9")
        self.assertTrue(roles.stitched)

    def test_reroll_model_is_not_the_layereddiffusionapply_node(self):
        sample = next(node for node in self.graph.values()
                     if node["class_type"] == "KSampler")
        self.assertEqual(sample["inputs"]["model"], ["4", 0])

    def test_does_not_mutate_the_source_graph(self):
        before = json.dumps(LAYERDIFFUSE, sort_keys=True)
        repair_graph(LAYERDIFFUSE, image_name="x.png", mask_name="m.png",
                    positive="p", negative="n", seed=1, denoise=0.5,
                    size=512, prefix="rep-ld")
        self.assertEqual(json.dumps(LAYERDIFFUSE, sort_keys=True), before)


class MaskedRedrawGraphTest(unittest.TestCase):
    def test_repair_graph_keeps_the_inpaint_crop_defaults(self):
        graph = repair_graph(
            RAW, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=99, denoise=0.6, size=1024,
            prefix="rep-abc-s99")
        crop = next(node for node in graph.values()
                   if node["class_type"] == "InpaintCropImproved")
        self.assertEqual(crop["inputs"]["mask_expand_pixels"], 0)
        self.assertEqual(crop["inputs"]["mask_blend_pixels"], 32)

    def test_mask_padding_and_feather_reach_the_inpaint_crop(self):
        graph = masked_redraw_graph(
            RAW, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=99, denoise=0.48,
            mask_padding=24, mask_feather=8, size=768, prefix="mrd-abc-s99")
        crop = next(node for node in graph.values()
                   if node["class_type"] == "InpaintCropImproved")
        self.assertEqual(crop["inputs"]["mask_expand_pixels"], 24)
        self.assertEqual(crop["inputs"]["mask_blend_pixels"], 8)
        self.assertEqual(crop["inputs"]["output_target_width"], 768)
        self.assertEqual(crop["inputs"]["output_target_height"], 768)

    def test_otherwise_shaped_like_repair_graph(self):
        graph = masked_redraw_graph(
            RAW, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=99, denoise=0.6,
            mask_padding=0, mask_feather=32, size=1024, prefix="mrd-abc-s99")
        save = graph["9"]
        self.assertEqual(save["class_type"], "SaveImage")
        self.assertEqual(save["inputs"]["filename_prefix"], "mrd-abc-s99")
        sample = next(node for node in graph.values()
                     if node["class_type"] == "KSampler")
        self.assertEqual(sample["inputs"]["seed"], 99)
        self.assertEqual(sample["inputs"]["denoise"], 0.6)

    def test_does_not_mutate_the_source_graph(self):
        before = json.dumps(RAW, sort_keys=True)
        masked_redraw_graph(
            RAW, image_name="x.png", mask_name="m.png", positive="p",
            negative="n", seed=1, denoise=0.5, mask_padding=10,
            mask_feather=20, size=512, prefix="mrd")
        self.assertEqual(json.dumps(RAW, sort_keys=True), before)


class RepairGraphLorasTest(unittest.TestCase):
    def test_no_loraloader_is_added_when_loras_is_empty(self):
        graph = repair_graph(
            RAW, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=99, denoise=0.6, size=1024,
            prefix="rep-abc-s99", loras=())
        added = [node for key, node in graph.items() if key not in RAW]
        self.assertFalse(
            any(node["class_type"] == "LoraLoader" for node in added))

    def test_one_lora_feeds_the_ksampler_model_and_both_clip_text_encodes(self):
        graph = repair_graph(
            RAW, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=99, denoise=0.6, size=1024,
            prefix="rep-abc-s99", loras=[("feet-xl-ill.safetensors", 0.8)])
        added_loaders = [(key, node) for key, node in graph.items()
                         if key not in RAW and node["class_type"] == "LoraLoader"]
        self.assertEqual(len(added_loaders), 1)
        loader_id, loader = added_loaders[0]
        self.assertEqual(loader["inputs"]["model"], ["10", 0])
        self.assertEqual(loader["inputs"]["clip"], ["10", 1])
        self.assertEqual(loader["inputs"]["lora_name"], "feet-xl-ill.safetensors")
        self.assertEqual(loader["inputs"]["strength_model"], 0.8)
        self.assertEqual(loader["inputs"]["strength_clip"], 0.8)

        sample = next(node for node in graph.values()
                     if node["class_type"] == "KSampler")
        self.assertEqual(sample["inputs"]["model"], [loader_id, 0])
        positive_node = graph[sample["inputs"]["positive"][0]]
        negative_node = graph[sample["inputs"]["negative"][0]]
        self.assertEqual(positive_node["inputs"]["clip"], [loader_id, 1])
        self.assertEqual(negative_node["inputs"]["clip"], [loader_id, 1])

    def test_two_loras_chain_in_order(self):
        graph = repair_graph(
            RAW, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=99, denoise=0.6, size=1024,
            prefix="rep-abc-s99",
            loras=[("feet-xl-ill.safetensors", 0.8),
                  ("hands-xl-ill.safetensors", 0.5)])
        added_loaders = [(key, node) for key, node in graph.items()
                         if key not in RAW and node["class_type"] == "LoraLoader"]
        self.assertEqual(len(added_loaders), 2)
        first_id, first = next(
            (key, node) for key, node in added_loaders
            if node["inputs"]["lora_name"] == "feet-xl-ill.safetensors")
        second_id, second = next(
            (key, node) for key, node in added_loaders
            if node["inputs"]["lora_name"] == "hands-xl-ill.safetensors")
        self.assertEqual(first["inputs"]["model"], ["10", 0])
        self.assertEqual(first["inputs"]["clip"], ["10", 1])
        self.assertEqual(second["inputs"]["model"], [first_id, 0])
        self.assertEqual(second["inputs"]["clip"], [first_id, 1])

        sample = next(node for node in graph.values()
                     if node["class_type"] == "KSampler")
        self.assertEqual(sample["inputs"]["model"], [second_id, 0])
        positive_node = graph[sample["inputs"]["positive"][0]]
        self.assertEqual(positive_node["inputs"]["clip"], [second_id, 1])


class RerollHooksTest(unittest.TestCase):
    def _sampler(self, graph):
        return next(node for node in graph.values()
                    if node["class_type"] == "KSampler"
                    and node["inputs"]["denoise"] == 0.6)

    def test_model_hook_reroutes_model_clip_vae_and_sampler_settings(self):
        seen = {}

        def swap(graph, allocate, refs):
            seen["positive"] = refs.positive
            loader = allocate()
            graph[loader] = {"class_type": "UNETLoader", "inputs": {
                "unet_name": "other.safetensors"}}
            return replace(refs, model=[loader, 0], positive_clip=[loader, 1],
                           negative_clip=[loader, 1], vae=[loader, 2],
                           sampler={"steps": 12, "cfg": 3.5})

        graph = repair_graph(
            RAW, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=99, denoise=0.6, size=1024,
            prefix="rep-abc-s99", loras=[("feet-xl-ill.safetensors", 0.8)],
            model_hooks=[swap])
        self.assertIsNone(seen["positive"])
        loader_id = next(key for key, node in graph.items()
                         if node["class_type"] == "UNETLoader")
        sample = self._sampler(graph)
        self.assertEqual(sample["inputs"]["model"], [loader_id, 0])
        self.assertEqual(sample["inputs"]["steps"], 12)
        self.assertEqual(sample["inputs"]["cfg"], 3.5)
        self.assertEqual(sample["inputs"]["seed"], 99)
        self.assertEqual(sample["inputs"]["denoise"], 0.6)
        for key in ("positive", "negative"):
            self.assertEqual(graph[sample["inputs"][key][0]]["inputs"]["clip"],
                             [loader_id, 1])
        encode = graph[graph[sample["inputs"]["latent_image"][0]]
                       ["inputs"]["samples"][0]]
        self.assertEqual(encode["inputs"]["vae"], [loader_id, 2])
        decode = next(node for node in graph.values()
                      if node["class_type"] == "VAEDecode"
                      and node["inputs"]["samples"][0] in graph
                      and graph[node["inputs"]["samples"][0]] is sample)
        self.assertEqual(decode["inputs"]["vae"], [loader_id, 2])

    def test_conditioning_hook_sees_text_encodes_and_reroutes_them(self):
        seen = {}

        def control(graph, allocate, refs):
            seen["refs"] = refs
            apply = allocate()
            graph[apply] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
                "positive": refs.positive, "negative": refs.negative,
                "image": refs.cropped_image}}
            return replace(refs, positive=[apply, 0], negative=[apply, 1])

        graph = repair_graph(
            RAW, image_name="src.png", mask_name="mask.png",
            positive="p", negative="n", seed=99, denoise=0.6, size=1024,
            prefix="rep-abc-s99", conditioning_hooks=[control])
        refs = seen["refs"]
        self.assertEqual(graph[refs.positive[0]]["inputs"]["text"], "p")
        self.assertEqual(graph[refs.negative[0]]["inputs"]["text"], "n")
        self.assertEqual(graph[refs.cropped_image[0]]["class_type"],
                         "InpaintCropImproved")
        apply_id = next(key for key, node in graph.items()
                        if node["class_type"] == "ControlNetApplyAdvanced")
        sample = self._sampler(graph)
        self.assertEqual(sample["inputs"]["positive"], [apply_id, 0])
        self.assertEqual(sample["inputs"]["negative"], [apply_id, 1])

    def test_hooks_reach_masked_redraw(self):
        calls = []

        def note(graph, allocate, refs):
            calls.append(refs)
            return refs

        masked_redraw_graph(
            RAW, image_name="src.png", mask_name="mask.png", positive="p",
            negative="n", seed=1, denoise=0.45, mask_padding=0,
            mask_feather=20, size=512, prefix="mrd",
            model_hooks=[note], conditioning_hooks=[note])
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0].positive)
        self.assertIsNotNone(calls[1].positive)


class RepairGraphErrorsTest(unittest.TestCase):
    def test_a_graph_with_no_final_decode_raises(self):
        broken = {"3": {"class_type": "KSampler", "inputs": {
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0], "seed": 1, "steps": 1, "cfg": 1,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1}}}
        with self.assertRaises(ValueError):
            repair_graph(broken, image_name="i", mask_name="m",
                        positive="p", negative="n", seed=1, denoise=0.5,
                        size=512, prefix="rep")

    def test_a_graph_with_two_final_decodes_raises(self):
        broken = {
            "4": {"class_type": "DiffusersLoader", "inputs": {"model_path": "m"}},
            "8a": {"class_type": "VAEDecode", "inputs": {
                "samples": ["3a", 0], "vae": ["4", 2]}},
            "8b": {"class_type": "VAEDecode", "inputs": {
                "samples": ["3b", 0], "vae": ["4", 2]}},
            "3a": {"class_type": "KSampler", "inputs": {
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["5", 0], "seed": 1, "steps": 1, "cfg": 1,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1}},
            "3b": {"class_type": "KSampler", "inputs": {
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["5", 0], "seed": 1, "steps": 1, "cfg": 1,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "n"}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {}},
            "9a": {"class_type": "SaveImage", "inputs": {
                "images": ["8a", 0], "filename_prefix": "a"}},
            "9b": {"class_type": "SaveImage", "inputs": {
                "images": ["8b", 0], "filename_prefix": "b"}},
        }
        with self.assertRaises(ValueError):
            repair_graph(broken, image_name="i", mask_name="m",
                        positive="p", negative="n", seed=1, denoise=0.5,
                        size=512, prefix="rep")


class GuidedStepsBaseTest(unittest.TestCase):
    """repair/masked_redraw must redraw a guided two-stage Anima base at its
    guided cfg (the first stage's own), not the final stage's own 1.0."""

    def setUp(self):
        spec = render_spec("coffee", 42, "p")
        self.spec = spec
        self.base = anima_graph.build_graph(spec)

    def test_repair_graph_reports_the_guided_cfg_not_the_final_stages_one(self):
        graph = repair_graph(
            self.base, image_name="i.png", mask_name="m.png",
            positive="p", negative="n", seed=99, denoise=0.6, size=512,
            prefix="rep")
        sample = next(
            node for node in graph.values()
            if node["class_type"] == "KSampler" and node["inputs"]["denoise"] == 0.6)
        self.assertEqual(sample["inputs"]["cfg"], self.spec.cfg)
        self.assertEqual(sample["inputs"]["steps"], self.spec.steps)


if __name__ == "__main__":
    unittest.main()
