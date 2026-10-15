"""Unit tests for declarative request-time RenderSpec patches.

Pure domain logic plus one round trip through the ComfyUI graph encoder;
no network.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from chagama.domain.generation.models import PromptPair, RenderSpec
from chagama.domain.generation.patches import (
    Patch,
    apply_patches,
    parse_patches,
)
from chagama.domain.yukari.dials import DIALS as ANIMA_DIALS
from chagama.domain.yukari.recipe import render_spec as anima_render_spec
from chagama.infrastructure.comfyui import anima_graph


def _patch(**fields):
    base = {"target": "prompt.positive", "op": "append", "value": " extra",
            "reason": "test"}
    base.update(fields)
    return base


class ParsePatchesTest(unittest.TestCase):
    def test_rejects_non_list(self):
        with self.assertRaises(ValueError):
            parse_patches({"target": "render.cfg"})

    def test_rejects_non_dict_element(self):
        with self.assertRaises(ValueError):
            parse_patches(["not a dict"])

    def test_rejects_unknown_target(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="prompt.expression")])

    def test_rejects_unknown_key(self):
        with self.assertRaises(ValueError):
            parse_patches([{**_patch(), "extra": "nope"}])

    def test_rejects_missing_reason(self):
        patch = _patch()
        del patch["reason"]
        with self.assertRaises(ValueError):
            parse_patches([patch])

    def test_rejects_empty_reason(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(reason="")])

    def test_rejects_replace_missing_old(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(op="replace", value="new")])

    def test_rejects_remove_with_value(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(op="remove", old="x", value="y")])

    def test_rejects_append_with_old(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(op="append", value="x", old="y")])

    def test_rejects_set_on_text_target(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(op="set", value="x")])

    def test_rejects_append_on_number_target(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.cfg", op="append", value=1)])

    def test_rejects_steps_zero(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.steps", op="set", value=0,
                                  reason="test")])

    def test_rejects_steps_bool(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.steps", op="set", value=True,
                                  reason="test")])

    def test_rejects_denoise_over_one(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="hires.denoise", op="set", value=1.5,
                                  reason="test")])

    def test_rejects_cfg_zero(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.cfg", op="set", value=0,
                                  reason="test")])

    def test_rejects_width_non_int(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.width", op="set",
                                  value=1280.5, reason="test")])

    def test_rejects_width_below_64(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.width", op="set", value=32,
                                  reason="test")])

    def test_rejects_width_not_multiple_of_8(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.width", op="set", value=100,
                                  reason="test")])

    def test_rejects_height_non_int(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.height", op="set",
                                  value=2048.5, reason="test")])

    def test_rejects_height_below_64(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.height", op="set", value=32,
                                  reason="test")])

    def test_rejects_height_not_multiple_of_8(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.height", op="set", value=100,
                                  reason="test")])

    def test_rejects_non_set_op_on_string_target(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.model", op="append",
                                  value="x", reason="test")])

    def test_rejects_empty_string_target_value(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.sampler", op="set", value="",
                                  reason="test")])

    def test_accepts_each_op_as_patch_tuple(self):
        raw = [
            _patch(target="prompt.positive", op="append", value=" a",
                   reason="r1"),
            _patch(target="prompt.negative", op="prepend", value="b ",
                   reason="r2"),
            _patch(target="prompt.hires.positive", op="replace", old="x",
                   value="y", reason="r3"),
            {"target": "prompt.hires.negative", "op": "remove", "old": "z",
             "reason": "r4"},
            _patch(target="render.cfg", op="set", value=4.5, reason="r5"),
            _patch(target="render.steps", op="set", value=20, reason="r6"),
            _patch(target="hires.denoise", op="set", value=0.5, reason="r7"),
        ]
        parsed = parse_patches(raw)
        self.assertEqual(len(parsed), len(raw))
        self.assertTrue(all(isinstance(p, Patch) for p in parsed))
        self.assertEqual(parsed[0], Patch("prompt.positive", "append", " a",
                                          None, "r1"))
        self.assertEqual(parsed[4], Patch("render.cfg", "set", 4.5, None,
                                          "r5"))

    def test_number_target_word_resolves_through_dials(self):
        dials = {"render.width": {"draft": 1024, "full": 1280}}
        parsed = parse_patches([_patch(
            target="render.width", op="set", value="full",
            reason="r")], dials)
        self.assertEqual(parsed[0], Patch("render.width", "set", 1280,
                                          None, "r"))

    def test_number_target_unknown_word_names_the_word(self):
        dials = {"render.width": {"draft": 1024, "full": 1280}}
        with self.assertRaisesRegex(ValueError, "punchy"):
            parse_patches([_patch(
                target="render.width", op="set", value="punchy",
                reason="r")], dials)

    def test_number_target_word_with_no_dials_for_that_target_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(
                target="render.width", op="set", value="full",
                reason="r")])

    def test_number_target_word_still_enforces_its_own_range(self):
        dials = {"render.width": {"tiny": 40}}
        with self.assertRaises(ValueError):
            parse_patches([_patch(
                target="render.width", op="set", value="tiny",
                reason="r")], dials)

    def test_anima_render_width_word_resolves_through_its_own_dials(self):
        dials = ANIMA_DIALS["patches"]
        parsed = parse_patches([_patch(
            target="render.width", op="set", value="full",
            reason="r")], dials)
        self.assertEqual(parsed[0], Patch("render.width", "set", 1280,
                                          None, "r"))

    def test_anima_render_width_word_applies_to_an_anima_spec(self):
        spec = anima_render_spec("stand", 42, "p")
        patches = parse_patches([_patch(
            target="render.width", op="set", value="full",
            reason="r")], ANIMA_DIALS["patches"])
        result = apply_patches(spec, patches)
        self.assertEqual(result.width, 1280)

    def test_loras_accepts_a_list_of_name_strength_pairs(self):
        parsed = parse_patches([_patch(
            target="render.loras", op="set",
            value=[["a.safetensors", 0.8], ["b.safetensors", 1.2]],
            reason="r")])
        self.assertEqual(parsed[0], Patch(
            "render.loras", "set",
            (("a.safetensors", 0.8), ("b.safetensors", 1.2)), None, "r"))

    def test_loras_rejects_non_set_op(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(
                target="render.loras", op="append",
                value=[["a.safetensors", 0.8]], reason="r")])

    def test_loras_rejects_empty_list(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(target="render.loras", op="set", value=[],
                                  reason="r")])

    def test_loras_rejects_a_name_not_ending_in_safetensors(self):
        with self.assertRaisesRegex(ValueError, "safetensors"):
            parse_patches([_patch(
                target="render.loras", op="set", value=[["a.ckpt", 0.8]],
                reason="r")])

    def test_loras_rejects_an_empty_name(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(
                target="render.loras", op="set", value=[["", 0.8]], reason="r")])

    def test_loras_rejects_a_strength_out_of_range(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(
                target="render.loras", op="set",
                value=[["a.safetensors", 2.1]], reason="r")])
        with self.assertRaises(ValueError):
            parse_patches([_patch(
                target="render.loras", op="set",
                value=[["a.safetensors", -0.1]], reason="r")])

    def test_loras_rejects_a_pair_that_is_not_length_two(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(
                target="render.loras", op="set",
                value=[["a.safetensors"]], reason="r")])


class ApplyPatchesTest(unittest.TestCase):
    def setUp(self):
        self.spec = anima_render_spec("coffee", 555666777, "prefix")
        self.hires_spec = anima_render_spec(
            "coffee", 555666777, "prefix", hires=2048)

    def test_append_and_prepend_positive(self):
        patches = parse_patches([
            _patch(target="prompt.positive", op="append", value=" tail",
                   reason="r"),
            _patch(target="prompt.positive", op="prepend", value="head ",
                   reason="r"),
        ])
        result = apply_patches(self.spec, patches)
        self.assertEqual(
            result.prompts.positive,
            "head " + self.spec.prompts.positive + " tail")

    def test_append_negative(self):
        patches = parse_patches([
            _patch(target="prompt.negative", op="append", value=", extra",
                   reason="r")])
        result = apply_patches(self.spec, patches)
        self.assertEqual(
            result.prompts.negative, self.spec.prompts.negative + ", extra")

    def test_replace_existing_needle(self):
        needle = "(wide hips:1.2)"
        self.assertIn(needle, self.spec.prompts.positive)
        patches = parse_patches([
            _patch(target="prompt.positive", op="replace", old=needle,
                   value="(wide hips:1.25)", reason="r")])
        result = apply_patches(self.spec, patches)
        self.assertNotIn(needle, result.prompts.positive)
        self.assertIn("(wide hips:1.25)", result.prompts.positive)

    def test_replace_missing_needle_raises(self):
        patches = parse_patches([
            _patch(target="prompt.positive", op="replace",
                   old="no such text in the prompt", value="x", reason="r")])
        with self.assertRaises(ValueError):
            apply_patches(self.spec, patches)

    def test_remove_missing_needle_raises(self):
        patches = parse_patches([
            {"target": "prompt.negative", "op": "remove",
             "old": "no such text in the prompt", "reason": "r"}])
        with self.assertRaises(ValueError):
            apply_patches(self.spec, patches)

    def test_hires_negative_append(self):
        patches = parse_patches([
            _patch(target="prompt.hires.negative", op="append", value=", x",
                   reason="r")])
        result = apply_patches(self.hires_spec, patches)
        self.assertEqual(
            result.hires.negative, self.hires_spec.hires.negative + ", x")

    def test_hires_positive_materializes_from_patched_pass1(self):
        self.assertIsNone(self.hires_spec.hires.positive)
        patches = parse_patches([
            _patch(target="prompt.positive", op="append", value=" pass1tail",
                   reason="r"),
            _patch(target="prompt.hires.positive", op="append",
                   value=" pass2tail", reason="r"),
        ])
        result = apply_patches(self.hires_spec, patches)
        expected = self.hires_spec.prompts.positive + " pass1tail pass2tail"
        self.assertEqual(result.hires.positive, expected)

    def test_hires_targets_require_hires_spec(self):
        for patch_dict in (
            _patch(target="prompt.hires.positive", op="append", value=" x",
                  reason="r"),
            _patch(target="prompt.hires.negative", op="append", value=" x",
                  reason="r"),
            _patch(target="hires.denoise", op="set", value=0.5, reason="r"),
        ):
            with self.subTest(patch_dict=patch_dict):
                patches = parse_patches([patch_dict])
                with self.assertRaises(ValueError):
                    apply_patches(self.spec, patches)

    def test_number_sets(self):
        patches = parse_patches([
            _patch(target="render.cfg", op="set", value=4.5, reason="r"),
            _patch(target="render.steps", op="set", value=20, reason="r"),
            _patch(target="hires.denoise", op="set", value=0.4, reason="r"),
        ])
        result = apply_patches(self.hires_spec, patches)
        self.assertEqual(result.cfg, 4.5)
        self.assertEqual(result.steps, 20)
        self.assertEqual(result.hires.denoise, 0.4)

    def test_string_sets(self):
        patches = parse_patches([
            _patch(target="render.model", op="set", value="other.safetensors",
                  reason="r"),
            _patch(target="render.sampler", op="set", value="euler",
                  reason="r"),
            _patch(target="render.scheduler", op="set", value="normal",
                  reason="r"),
        ])
        result = apply_patches(self.spec, patches)
        self.assertEqual(result.model_path, "other.safetensors")
        self.assertEqual(result.sampler_name, "euler")
        self.assertEqual(result.scheduler, "normal")

    def test_width_and_height_sets(self):
        patches = parse_patches([
            _patch(target="render.width", op="set", value=1280, reason="r"),
            _patch(target="render.height", op="set", value=2048, reason="r"),
        ])
        result = apply_patches(self.spec, patches)
        self.assertEqual(result.width, 1280)
        self.assertEqual(result.height, 2048)

    def test_loras_replaces_spec_loras(self):
        spec = replace(self.spec, loras=(("a.safetensors", 0.5),))
        patches = parse_patches([_patch(
            target="render.loras", op="set",
            value=[["c.safetensors", 0.9], ["d.safetensors", 1.4]],
            reason="r")])
        result = apply_patches(spec, patches)
        self.assertEqual(result.loras, (("c.safetensors", 0.9),
                                        ("d.safetensors", 1.4)))

    def test_empty_patches_returns_equal_spec(self):
        result = apply_patches(self.spec, ())
        self.assertEqual(result, self.spec)

    def test_original_spec_is_unchanged(self):
        original_positive = self.spec.prompts.positive
        patches = parse_patches([
            _patch(target="prompt.positive", op="append", value=" tail",
                   reason="r")])
        apply_patches(self.spec, patches)
        self.assertEqual(self.spec.prompts.positive, original_positive)

    def test_encoded_graph_reflects_cfg_and_hires_denoise_patches(self):
        spec = replace(self.hires_spec, guided_steps=None)
        patches = parse_patches([
            _patch(target="render.cfg", op="set", value=4.5, reason="r"),
            _patch(target="hires.denoise", op="set", value=0.4, reason="r"),
        ])
        patched = apply_patches(spec, patches)
        graph = anima_graph.build_graph(patched)
        self.assertEqual(graph["3"]["inputs"]["cfg"], 4.5)
        self.assertEqual(graph["11"]["inputs"]["denoise"], 0.4)

    def test_unpatched_encoding_matches_direct_build(self):
        spec = anima_render_spec("coffee", 555666777, "prefix")
        patched = apply_patches(spec, parse_patches([]))
        self.assertEqual(
            anima_graph.build_graph(patched), anima_graph.build_graph(spec))

    def test_render_loras_patch_encodes_a_lora_loader_model_only_chain(self):
        spec = anima_render_spec("coffee", 7, "prefix")
        patches = parse_patches([_patch(
            target="render.loras", op="set",
            value=[["anima-sketch-style-chosen.safetensors", 0.8]],
            reason="r")])
        patched = apply_patches(spec, patches)
        graph = anima_graph.build_graph(patched)
        loader_ids = [node_id for node_id, node in graph.items()
                     if node.get("class_type") == "LoraLoaderModelOnly"]
        self.assertEqual(len(loader_ids), 1)
        self.assertEqual(
            graph[loader_ids[0]]["inputs"],
            {"model": ["1", 0],
             "lora_name": "anima-sketch-style-chosen.safetensors",
             "strength_model": 0.8})
        self.assertEqual(graph["3"]["inputs"]["model"], [loader_ids[0], 0])


class PartTargetPatchTest(unittest.TestCase):
    def setUp(self):
        self.anima_spec = anima_render_spec("coffee", 7, "prefix")
        self.no_parts_spec = RenderSpec(
            model_path="m", prompts=PromptPair("p", "n"),
            width=8, height=8, seed=1, steps=1, cfg=1.0,
            sampler_name="s", scheduler="k", denoise=1.0,
            filename_prefix="prefix")

    def test_parts_join_back_into_the_whole_positive(self):
        joined = "".join(text for _, text in self.anima_spec.positive_parts)
        self.assertEqual(joined, self.anima_spec.prompts.positive)

    def test_a_recipe_without_parts_has_an_empty_positive_parts(self):
        self.assertEqual(self.no_parts_spec.positive_parts, ())

    def test_append_edits_only_the_named_component(self):
        patches = parse_patches([_patch(
            target="prompt.positive.framing_tags", op="append",
            value="(overcast:1.1), ", reason="r")])
        result = apply_patches(self.anima_spec, patches)
        parts = dict(result.positive_parts)
        self.assertTrue(parts["framing_tags"].endswith("(overcast:1.1), "))
        for name, text in dict(self.anima_spec.positive_parts).items():
            if name != "framing_tags":
                self.assertEqual(parts[name], text)
        self.assertEqual(
            result.prompts.positive,
            "".join(text for _, text in result.positive_parts))

    def test_replace_and_remove_keep_separators_clean(self):
        patches = parse_patches([_patch(
            target="prompt.positive.face", op="replace",
            old="(tareme:1.2)", value="(tareme:1.3)", reason="r")])
        result = apply_patches(self.anima_spec, patches)
        self.assertIn("(tareme:1.3), (thick eyelashes:1.3)",
                      result.prompts.positive)
        self.assertNotIn(",,", result.prompts.positive)
        self.assertNotIn("  ", result.prompts.positive)

        patches = parse_patches([
            {"target": "prompt.positive.pose", "op": "remove",
             "old": "(straw in mouth:1.25), ", "reason": "r"}])
        result = apply_patches(anima_render_spec("coffee", 7, "p"), patches)
        self.assertNotIn("(straw in mouth:1.25)", result.prompts.positive)
        self.assertNotIn(", , ", result.prompts.positive)
        self.assertNotIn(",,", result.prompts.positive)

    def test_prepend_to_a_part(self):
        patches = parse_patches([_patch(
            target="prompt.positive.gesture", op="prepend",
            value="(grin:1.1), ", reason="r")])
        result = apply_patches(self.anima_spec, patches)
        self.assertTrue(dict(result.positive_parts)["gesture"]
                        .startswith("(grin:1.1), "))

    def test_unknown_part_raises_and_names_component_and_legacy_names(self):
        patches = parse_patches([_patch(
            target="prompt.positive.nope", op="append", value="x",
            reason="r")])
        with self.assertRaises(ValueError) as ctx:
            apply_patches(self.anima_spec, patches)
        message = str(ctx.exception)
        self.assertIn("quality", message)
        self.assertIn("style", message)
        self.assertIn("scene", message)

    def test_legacy_scene_replace_of_a_framing_tag_hits_framing_tags(self):
        patches = parse_patches([_patch(
            target="prompt.positive.scene", op="replace",
            old="(cowboy shot:1.3), ", value="(dynamic cowboy shot:1.3), ",
            reason="r")])
        result = apply_patches(self.anima_spec, patches)
        parts = dict(result.positive_parts)
        self.assertEqual(parts["framing_tags"], "(dynamic cowboy shot:1.3), ")
        before = dict(self.anima_spec.positive_parts)
        self.assertEqual(parts["leg_display"], before["leg_display"])

    def test_legacy_pose_replace_hits_action(self):
        patches = parse_patches([_patch(
            target="prompt.positive.pose", op="replace",
            old="(drinking:1.3), ", value="(sipping:1.3), ", reason="r")])
        result = apply_patches(self.anima_spec, patches)
        self.assertTrue(
            dict(result.positive_parts)["action"].startswith("(sipping:1.3), "))

    def test_legacy_costume_replace_of_a_legwear_tag_hits_legwear(self):
        opaque_spec = anima_render_spec("coffee", 7, "prefix", costume="outing",
                                         legwear="opaque")
        patches = parse_patches([_patch(
            target="prompt.positive.costume", op="replace",
            old="(black pantyhose:1.5), ", value="(navy pantyhose:1.5), ",
            reason="r")])
        result = apply_patches(opaque_spec, patches)
        parts = dict(result.positive_parts)
        self.assertIn("(navy pantyhose:1.5), ", parts["legwear"])
        before = dict(opaque_spec.positive_parts)
        self.assertEqual(parts["costume"], before["costume"])

    def test_legacy_scene_replace_spanning_two_members_raises_split_error(self):
        before = dict(self.anima_spec.positive_parts)
        spanning = before["framing_tags"] + before["leg_display"]
        patches = parse_patches([_patch(
            target="prompt.positive.scene", op="replace",
            old=spanning, value="x", reason="r")])
        with self.assertRaises(ValueError) as ctx:
            apply_patches(self.anima_spec, patches)
        message = str(ctx.exception)
        self.assertIn("framing_tags", message)
        self.assertIn("leg_display", message)

    def test_append_and_prepend_on_a_legacy_group(self):
        patches = parse_patches([
            _patch(target="prompt.positive.scene", op="append",
                  value="(overcast:1.1), ", reason="r"),
            _patch(target="prompt.positive.scene", op="prepend",
                  value="(golden hour:1.1), ", reason="r"),
        ])
        result = apply_patches(self.anima_spec, patches)
        parts = dict(result.positive_parts)
        self.assertTrue(parts["leg_display"].endswith("(overcast:1.1), "))
        self.assertTrue(parts["framing_tags"].startswith("(golden hour:1.1), "))

    def test_part_target_on_a_recipe_without_parts_raises(self):
        patches = parse_patches([_patch(
            target="prompt.positive.face", op="append", value="x",
            reason="r")])
        with self.assertRaises(ValueError):
            apply_patches(self.no_parts_spec, patches)

    def test_dotted_sub_part_is_an_unknown_target(self):
        with self.assertRaises(ValueError):
            parse_patches([_patch(
                target="prompt.positive.face.extra", op="append",
                value="x", reason="r")])

    def test_replace_missing_needle_in_a_part_raises(self):
        patches = parse_patches([_patch(
            target="prompt.positive.face", op="replace",
            old="no such text here", value="x", reason="r")])
        with self.assertRaises(ValueError):
            apply_patches(self.anima_spec, patches)


if __name__ == "__main__":
    unittest.main()
