"""Tests for turning a request row's JSON `options` into use-case kwargs."""

from __future__ import annotations

import unittest

from comfyui_recipes.application.deliver import RECIPE_DEFAULT
from comfyui_recipes.application.request_options import (
    deliver_arguments,
    dof_arguments,
    lens_arguments,
    photo_arguments,
    masked_redraw_arguments,
    redraw_arguments,
    repair_arguments,
)
from comfyui_recipes.domain.yukari.delivery_style import Light
from comfyui_recipes.domain.repair.controlnet import DEFAULT_CONTROL_STRENGTH
from comfyui_recipes.domain.repair.loras import DEFAULT_PART_LORA_WEIGHT

# Synthetic dial vocabularies for exercising resolve_dial()'s word lookup --
# not any recipe's own published words.
_REPAIR_DIALS = {
    "denoise": {"keep": 0.6},
    "lora": {"on": DEFAULT_PART_LORA_WEIGHT},
}
_DENOISE_ONLY_DIALS = {"denoise": {"keep": 0.45}}


class RedrawArgumentsTest(unittest.TestCase):
    def test_a_method_is_required_and_known(self):
        for options in ({}, {"method": None}, {"method": "deliver"}):
            with self.subTest(options=options), self.assertRaisesRegex(
                    ValueError, "method"):
                redraw_arguments(options)
        with self.assertRaisesRegex(ValueError, "object"):
            redraw_arguments([])

    def test_canvas_defaults_leave_the_recipe_values_to_the_use_case(self):
        self.assertEqual(redraw_arguments({"method": "canvas"}), {
            "method": "canvas", "denoise": None, "size": None,
            "latent_route": None, "finalizer": None, "upscale": None,
            "keep_regions": [], "keep_strength": 0.25})

    def test_canvas_options_parse(self):
        arguments = redraw_arguments({
            "method": "canvas", "denoise": 0.5, "size": 2048, "route": "latent",
            "finalizer": "model", "upscale": "lanczos",
            "keep_regions": [[0, 0, 0.5, 0.5]], "keep_strength": 0.3})
        self.assertEqual(arguments, {
            "method": "canvas", "denoise": 0.5, "size": 2048,
            "latent_route": True, "finalizer": "model", "upscale": "lanczos",
            "keep_regions": [[0.0, 0.0, 0.5, 0.5]], "keep_strength": 0.3})
        self.assertIs(redraw_arguments(
            {"method": "canvas", "route": "pixel"})["latent_route"], False)

    def test_canvas_denoise_resolves_dial_words(self):
        self.assertEqual(redraw_arguments(
            {"method": "canvas", "denoise": "keep"}, _DENOISE_ONLY_DIALS)["denoise"],
            0.45)
        with self.assertRaisesRegex(ValueError, "unknown denoise word"):
            redraw_arguments({"method": "canvas", "denoise": "tidy"},
                             _DENOISE_ONLY_DIALS)

    def test_canvas_option_checks(self):
        for options in ({"route": "sideways"}, {"upscale": "sharp"},
                        {"denoise": 0}, {"denoise": 1.5}, {"size": "big"},
                        {"keep_strength": 1}, {"keep_regions": [[0, 0, 2, 1]]},
                        {"finalizer": 3}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                redraw_arguments({"method": "canvas", **options})

    def test_hires_takes_its_pixels_and_defaults_the_denoise(self):
        self.assertEqual(redraw_arguments({"method": "hires", "hires": 2048}), {
            "method": "hires", "hires": 2048, "denoise": 0.45})
        self.assertEqual(
            redraw_arguments({"method": "hires", "hires": 2048,
                              "denoise": 0.3})["denoise"], 0.3)
        for options in ({}, {"hires": 0}, {"hires": "x"}, {"hires": True},
                        {"hires": 2048, "denoise": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                redraw_arguments({"method": "hires", **options})

    def test_hires_takes_no_dial_word(self):
        with self.assertRaisesRegex(ValueError, "number"):
            redraw_arguments({"method": "hires", "hires": 2048, "denoise": "keep"},
                             _DENOISE_ONLY_DIALS)

    def test_light_takes_a_scene_and_a_direction(self):
        self.assertEqual(redraw_arguments({"method": "light", "scene": "moon"}), {
            "method": "light", "light": Light("moon", "nw")})
        self.assertEqual(
            redraw_arguments({"method": "light", "scene": "moon",
                              "from": "se"})["light"], Light("moon", "se"))
        for options in ({}, {"scene": "noon"}, {"scene": "moon", "from": "up"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                redraw_arguments({"method": "light", **options})

    def test_unknown_keys_are_rejected_per_method(self):
        cases = {
            "canvas": ("hires", "scene", "from", "repin", "light"),
            "hires": ("size", "route", "scene", "hires_denoise", "light"),
            "light": ("denoise", "size", "hires", "stroke_light"),
        }
        valid = {"canvas": {}, "hires": {"hires": 2048},
                 "light": {"scene": "moon"}}
        for method, keys in cases.items():
            for key in keys:
                with self.subTest(method=method, key=key), self.assertRaisesRegex(
                        ValueError, key):
                    redraw_arguments({"method": method, **valid[method], key: 1})


class DeliverArgumentsTest(unittest.TestCase):
    def test_defaults_are_the_recipes_delivery(self):
        self.assertEqual(deliver_arguments({}), {
            "repin": True, "skin": False, "recolor": False, "keep_legwear": None,
            "keep_scene": False, "transparent": False, "backdrop": "dots",
            "stroke_light": RECIPE_DEFAULT, "deliver_size": None,
            "outlines": [{"color": "#ffffff", "width": 0.8},
                         {"color": "#885b80", "width": 1.04}],
            "light": None})

    def test_unknown_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "denoise"):
            deliver_arguments({"denoise": 0.4})
        with self.assertRaisesRegex(ValueError, "deliver_only"):
            deliver_arguments({"deliver_only": True})

    def test_options_must_be_an_object(self):
        with self.assertRaisesRegex(ValueError, "object"):
            deliver_arguments([])

    def test_transparent_drops_the_default_backdrop(self):
        arguments = deliver_arguments({"transparent": True})
        self.assertIsNone(arguments["backdrop"])
        self.assertIs(arguments["transparent"], True)

    def test_an_explicit_backdrop_survives_and_is_validated(self):
        self.assertEqual(deliver_arguments({"backdrop": "#aabbcc"})["backdrop"],
                         "#aabbcc")
        self.assertIsNone(deliver_arguments({"backdrop": None})["backdrop"])
        with self.assertRaisesRegex(ValueError, "backdrop"):
            deliver_arguments({"backdrop": "plaid"})

    def test_null_backdrop_delivers_transparent(self):
        arguments = deliver_arguments({"backdrop": None})
        self.assertIs(arguments["transparent"], True)
        self.assertIsNone(arguments["backdrop"])

    def test_null_backdrop_with_transparent_false_stays_opaque(self):
        arguments = deliver_arguments({"backdrop": None, "transparent": False})
        self.assertIs(arguments["transparent"], False)
        self.assertIsNone(arguments["backdrop"])

    def test_omitted_backdrop_stays_opaque_dots(self):
        arguments = deliver_arguments({})
        self.assertIs(arguments["transparent"], False)
        self.assertEqual(arguments["backdrop"], "dots")

    def test_keep_scene_overrides_null_backdrop(self):
        arguments = deliver_arguments({"keep_scene": True, "backdrop": None})
        self.assertIs(arguments["transparent"], False)

    def test_keep_scene_overrides_transparent(self):
        arguments = deliver_arguments({"keep_scene": True, "transparent": True})
        self.assertIs(arguments["transparent"], False)

    def test_recolor_wins_over_repin(self):
        arguments = deliver_arguments({"recolor": True})
        self.assertIs(arguments["recolor"], True)
        self.assertIs(arguments["repin"], False)

    def test_repin_can_be_turned_off(self):
        self.assertIs(deliver_arguments({"repin": False})["repin"], False)

    def test_keep_legwear_resolves_dial_words_and_true(self):
        dials = {"keep_legwear": {"on": 0.5}}
        self.assertEqual(
            deliver_arguments({"keep_legwear": "on"}, dials)["keep_legwear"], 0.5)
        self.assertEqual(
            deliver_arguments({"keep_legwear": True})["keep_legwear"], 0.62)
        with self.assertRaisesRegex(ValueError, "unknown keep_legwear word"):
            deliver_arguments({"keep_legwear": "off"}, dials)

    def test_stroke_light_follows_the_light_direction(self):
        arguments = deliver_arguments({"light": {"scene": "moon", "from": "se"}})
        self.assertEqual(arguments["light"], Light("moon", "se"))
        self.assertEqual(arguments["stroke_light"], "se")

    def test_a_stroke_light_that_disagrees_with_light_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "light の from"):
            deliver_arguments({"light": {"scene": "moon", "from": "se"},
                               "stroke_light": "n"})
        self.assertEqual(
            deliver_arguments({"light": {"scene": "moon", "from": "se"},
                               "stroke_light": "even"})["stroke_light"],
            "even")

    def test_stroke_light_none_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stroke_light"):
            deliver_arguments({"stroke_light": "none"})

    def test_dof_and_viewfinder_are_no_longer_deliver_options(self):
        for key in ("dof", "viewfinder"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                deliver_arguments({key: {"focus": [0.5, 0.5]}})

    def test_outlines_take_a_list_of_bands_or_none(self):
        bands = [{"color": "#FFFFFF", "width": 0.8}, {"color": "#885b80", "width": 2}]
        self.assertEqual(deliver_arguments({"outlines": bands})["outlines"],
                         [{"color": "#ffffff", "width": 0.8},
                          {"color": "#885b80", "width": 2.0}])
        self.assertEqual(deliver_arguments({"outlines": []})["outlines"], [])
        self.assertEqual(deliver_arguments({"outlines": None})["outlines"],
                         deliver_arguments({})["outlines"])
        self.assertEqual(len(deliver_arguments(
            {"outlines": [{"color": "#000000", "width": 5}] * 6})["outlines"]), 6)

    def test_bad_outlines_are_rejected(self):
        band = {"color": "#885b80", "width": 1.0}
        for bad in ({"color": "#885b80"}, [band] * 7, [{"color": "885b80", "width": 1}],
                    [{"color": "#885b8", "width": 1}], [{"color": "#885b80", "width": 0}],
                    [{"color": "#885b80", "width": 5.1}],
                    [{"color": "#885b80", "width": True}],
                    [{"color": "#885b80", "width": 1, "extra": 1}], ["#885b80"]):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "outlines"):
                deliver_arguments({"outlines": bad})

    def test_deliver_size_must_be_a_positive_integer(self):
        self.assertEqual(deliver_arguments({"deliver_size": 1536})["deliver_size"], 1536)
        for bad in (0, "big", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                deliver_arguments({"deliver_size": bad})


class DofArgumentsTest(unittest.TestCase):
    def test_defaults_take_the_catalog_values(self):
        self.assertEqual(dof_arguments({"focus": [0.5, 0.25]}), {
            "focus": (0.5, 0.25), "f_number": 2.8, "viewfinder": "off",
            "scope": {"figure": True, "outline": True, "backdrop": True}})

    def test_a_partial_scope_keeps_the_rest_on(self):
        arguments = dof_arguments({"focus": [0, 1], "f_number": 1.4,
                                   "scope": {"backdrop": False}, "viewfinder": "both"})
        self.assertEqual(arguments["scope"],
                         {"figure": True, "outline": True, "backdrop": False})
        self.assertEqual((arguments["f_number"], arguments["viewfinder"]), (1.4, "both"))

    def test_every_layer_off_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scope"):
            dof_arguments({"focus": [0.5, 0.5], "scope": {
                "figure": False, "outline": False, "backdrop": False}})

    def test_bad_options_are_rejected(self):
        for bad, key in (({}, "focus"), ({"focus": [0.5]}, "focus"),
                         ({"focus": [0.5, 1.2]}, "focus"),
                         ({"focus": [0.5, True]}, "focus"),
                         ({"focus": [0.5, 0.5], "f_number": 1.2}, "f_number"),
                         ({"focus": [0.5, 0.5], "f_number": 23}, "f_number"),
                         ({"focus": [0.5, 0.5], "scope": "all"}, "scope"),
                         ({"focus": [0.5, 0.5], "scope": {"rim": True}}, "scope"),
                         ({"focus": [0.5, 0.5], "scope": {"figure": 1}}, "scope"),
                         ({"focus": [0.5, 0.5], "viewfinder": "yes"}, "viewfinder"),
                         ({"focus": [0.5, 0.5], "blur": 1}, "blur")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, key):
                dof_arguments(bad)
        with self.assertRaisesRegex(ValueError, "object"):
            dof_arguments([])


class LensArgumentsTest(unittest.TestCase):
    def test_defaults_take_the_catalog_values(self):
        self.assertEqual(lens_arguments({}), {"center": (0.5, 0.5), "strength": 0.7,
                                              "shape": "fisheye", "radius": None})
        self.assertEqual(lens_arguments({"center": [0, 1], "strength": -1}),
                         {"center": (0.0, 1.0), "strength": -1.0,
                          "shape": "fisheye", "radius": None})

    def test_a_sphere_takes_a_radius_and_a_fisheye_refuses_one(self):
        self.assertEqual(lens_arguments({"shape": "sphere"})["radius"], 0.3)
        self.assertEqual(lens_arguments({"shape": "sphere", "radius": 1})["radius"], 1.0)
        for bad in ({"shape": "sphere", "radius": 0.01},
                    {"shape": "sphere", "radius": 1.5},
                    {"shape": "sphere", "radius": "big"},
                    {"radius": 0.3}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "radius"):
                lens_arguments(bad)
        with self.assertRaisesRegex(ValueError, "shape"):
            lens_arguments({"shape": "ball"})

    def test_bad_options_are_rejected(self):
        for bad, key in (({"center": [0.5]}, "center"),
                         ({"center": [0.5, 1.2]}, "center"),
                         ({"center": [0.5, True]}, "center"),
                         ({"center": "middle"}, "center"),
                         ({"strength": 1.5}, "strength"),
                         ({"strength": -1.01}, "strength"),
                         ({"strength": 0}, "strength"),
                         ({"strength": True}, "strength"),
                         ({"zoom": 2}, "zoom")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, key):
                lens_arguments(bad)
        with self.assertRaisesRegex(ValueError, "object"):
            lens_arguments([])


class PhotoArgumentsTest(unittest.TestCase):
    def test_defaults_take_the_catalog_values(self):
        self.assertEqual(photo_arguments({"photo": "pic"}), {
            "photo_id": "pic", "place": (0.5, 0.95), "height": 0.7,
            "stand_at": None, "nearer": 0.0, "focus": "subject", "f_number": 2.8,
            "tilt": 0.0})
        arguments = photo_arguments({"photo": "pic", "focus": [0.2, 0.3],
                                     "stand_at": [0.8, 0.4], "nearer": 0.5})
        self.assertEqual((arguments["focus"], arguments["stand_at"], arguments["nearer"]),
                         ((0.2, 0.3), (0.8, 0.4), 0.5))
        self.assertEqual(photo_arguments({"photo": "p", "place": [0.3, 1.4]})["place"],
                         (0.3, 1.4))

    def test_bad_options_are_rejected(self):
        for bad, key in (({}, "photo"), ({"photo": ""}, "photo"),
                         ({"photo": "p", "place": [0.5]}, "place"),
                         ({"photo": "p", "place": [0.5, 1.6]}, "place"),
                         ({"photo": "p", "place": [1.2, 0.5]}, "place"),
                         ({"photo": "p", "stand_at": [0.5, 1.2]}, "stand_at"),
                         ({"photo": "p", "height": 0.01}, "height"),
                         ({"photo": "p", "height": 2}, "height"),
                         ({"photo": "p", "stand_at": "sign"}, "stand_at"),
                         ({"photo": "p", "nearer": -0.1}, "nearer"),
                         ({"photo": "p", "focus": "sign"}, "focus"),
                         ({"photo": "p", "focus": None}, "focus"),
                         ({"photo": "p", "f_number": 1}, "f_number"),
                         ({"photo": "p", "tilt": 1.5}, "tilt"),
                         ({"photo": "p", "blur": 1}, "blur")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, key):
                photo_arguments(bad)
        with self.assertRaisesRegex(ValueError, "object"):
            photo_arguments([])


class RepairArgumentsTest(unittest.TestCase):
    def test_defaults(self):
        arguments = repair_arguments({})
        self.assertEqual(arguments, {
            "parts": ["hands", "feet"], "regions": [], "denoise": 0.6,
            "seeds": [1, 2, 3, 4], "size": 1024, "pad": 1.0, "lora": None,
            "model": None,
            "control": None, "control_strength": DEFAULT_CONTROL_STRENGTH,
        })

    def test_not_a_mapping_is_rejected(self):
        with self.assertRaises(ValueError):
            repair_arguments([])

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            repair_arguments({"nope": True})
        self.assertIn("nope", str(ctx.exception))

    def test_parts_must_be_hands_or_feet(self):
        with self.assertRaisesRegex(ValueError, "parts"):
            repair_arguments({"parts": ["elbows"]})

    def test_parts_empty_list_is_valid_with_regions(self):
        arguments = repair_arguments({"parts": [], "regions": [[0, 0, 1, 1]]})
        self.assertEqual(arguments["parts"], [])

    def test_parts_and_regions_both_empty_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "parts or regions"):
            repair_arguments({"parts": [], "regions": []})

    def test_regions_must_be_four_numbers_in_0_1(self):
        with self.assertRaisesRegex(ValueError, "region"):
            repair_arguments({"regions": [[0, 0, 1]]})
        with self.assertRaisesRegex(ValueError, "region"):
            repair_arguments({"regions": [[0, 0, 1, 1.5]]})
        with self.assertRaisesRegex(ValueError, "region"):
            repair_arguments({"regions": [["a", 0, 1, 1]]})

    def test_regions_pass_through_as_floats(self):
        arguments = repair_arguments({"regions": [[0, 0.25, 1, 0.75]]})
        self.assertEqual(arguments["regions"], [[0.0, 0.25, 1.0, 0.75]])

    def test_denoise_must_be_in_0_1(self):
        with self.assertRaisesRegex(ValueError, "denoise"):
            repair_arguments({"denoise": 0})
        with self.assertRaisesRegex(ValueError, "denoise"):
            repair_arguments({"denoise": 1.5})
        with self.assertRaisesRegex(ValueError, "denoise"):
            repair_arguments({"denoise": "0.5"})

    def test_denoise_one_is_valid(self):
        self.assertEqual(repair_arguments({"denoise": 1})["denoise"], 1.0)

    def test_seeds_must_be_a_non_empty_list_of_ints(self):
        with self.assertRaisesRegex(ValueError, "seeds"):
            repair_arguments({"seeds": []})
        with self.assertRaisesRegex(ValueError, "seeds"):
            repair_arguments({"seeds": [1.5]})
        with self.assertRaisesRegex(ValueError, "seeds"):
            repair_arguments({"seeds": [True]})

    def test_size_must_be_a_multiple_of_8_at_least_256(self):
        with self.assertRaisesRegex(ValueError, "size"):
            repair_arguments({"size": 200})
        with self.assertRaisesRegex(ValueError, "size"):
            repair_arguments({"size": 1001})
        with self.assertRaisesRegex(ValueError, "size"):
            repair_arguments({"size": 1024.0})

    def test_pad_must_be_between_half_and_three(self):
        with self.assertRaisesRegex(ValueError, "pad"):
            repair_arguments({"pad": 0.4})
        with self.assertRaisesRegex(ValueError, "pad"):
            repair_arguments({"pad": 3.1})

    def test_pad_bounds_are_inclusive(self):
        self.assertEqual(repair_arguments({"pad": 0.5})["pad"], 0.5)
        self.assertEqual(repair_arguments({"pad": 3})["pad"], 3.0)

    def test_lora_default_null(self):
        self.assertIsNone(repair_arguments({})["lora"])

    def test_lora_true_becomes_the_default_weight(self):
        self.assertEqual(
            repair_arguments({"lora": True})["lora"], DEFAULT_PART_LORA_WEIGHT)

    def test_lora_number_passes_through(self):
        self.assertEqual(repair_arguments({"lora": 0.5})["lora"], 0.5)

    def test_lora_out_of_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lora"):
            repair_arguments({"lora": 0})
        with self.assertRaisesRegex(ValueError, "lora"):
            repair_arguments({"lora": 2.5})

    def test_lora_a_string_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lora"):
            repair_arguments({"lora": "on"})

    def test_denoise_word_resolves_through_dials(self):
        dials = _REPAIR_DIALS
        self.assertEqual(repair_arguments({"denoise": "keep"}, dials)["denoise"], 0.6)

    def test_lora_word_resolves_through_dials(self):
        dials = _REPAIR_DIALS
        self.assertEqual(
            repair_arguments({"lora": "on"}, dials)["lora"], DEFAULT_PART_LORA_WEIGHT)

    def test_unknown_word_names_the_key_and_word(self):
        dials = _REPAIR_DIALS
        with self.assertRaises(ValueError) as ctx:
            repair_arguments({"denoise": "fuzzy"}, dials)
        self.assertIn("denoise", str(ctx.exception))
        self.assertIn("fuzzy", str(ctx.exception))

    def test_model_default_null(self):
        self.assertIsNone(repair_arguments({})["model"])

    def test_model_passes_through(self):
        self.assertEqual(repair_arguments({"model": "anima"})["model"], "anima")

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model"):
            repair_arguments({"model": "nope"})

    def test_control_default_null(self):
        self.assertIsNone(repair_arguments({})["control"])

    def test_control_lineart_passes_through(self):
        self.assertEqual(repair_arguments({"control": "lineart"})["control"], "lineart")

    def test_control_unknown_word_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "control"):
            repair_arguments({"control": "nope"})

    def test_control_not_a_string_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "control"):
            repair_arguments({"control": True})

    def test_control_strength_default(self):
        self.assertEqual(
            repair_arguments({})["control_strength"], DEFAULT_CONTROL_STRENGTH)

    def test_control_strength_number_passes_through(self):
        self.assertEqual(
            repair_arguments({"control_strength": 0.5})["control_strength"], 0.5)

    def test_control_strength_out_of_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "control_strength"):
            repair_arguments({"control_strength": 0})
        with self.assertRaisesRegex(ValueError, "control_strength"):
            repair_arguments({"control_strength": 2.5})


class MaskedRedrawArgumentsTest(unittest.TestCase):
    def _valid(self, **overrides):
        options = {"regions": [[0.1, 0.1, 0.5, 0.5]], "prompt_patch": "a dress"}
        options.update(overrides)
        return options

    def test_defaults(self):
        arguments = masked_redraw_arguments(self._valid())
        self.assertEqual(arguments, {
            "regions": [[0.1, 0.1, 0.5, 0.5]], "prompt_patch": "a dress",
            "denoise": 0.45, "mask_padding": 0.0, "mask_feather": 32.0,
            "size": 1024, "seeds": [1, 2, 3, 4],
        })

    def test_not_a_mapping_is_rejected(self):
        with self.assertRaises(ValueError):
            masked_redraw_arguments([])

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            masked_redraw_arguments(self._valid(nope=True))
        self.assertIn("nope", str(ctx.exception))

    def test_regions_empty_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "region"):
            masked_redraw_arguments(self._valid(regions=[]))

    def test_regions_out_of_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "region"):
            masked_redraw_arguments(self._valid(regions=[[0, 0, 1, 1.5]]))

    def test_overlapping_regions_are_not_rejected(self):
        # _regions_argument only validates shape/range; chimera already
        # enforces ordering and non-overlap before the row is queued.
        arguments = masked_redraw_arguments(self._valid(
            regions=[[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]]))
        self.assertEqual(
            arguments["regions"], [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]])

    def test_prompt_patch_empty_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prompt_patch"):
            masked_redraw_arguments(self._valid(prompt_patch=""))

    def test_prompt_patch_not_a_string_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prompt_patch"):
            masked_redraw_arguments(self._valid(prompt_patch=1))

    def test_prompt_patch_over_4096_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prompt_patch"):
            masked_redraw_arguments(self._valid(prompt_patch="x" * 4097))

    def test_prompt_patch_exactly_4096_is_valid(self):
        arguments = masked_redraw_arguments(self._valid(prompt_patch="x" * 4096))
        self.assertEqual(len(arguments["prompt_patch"]), 4096)

    def test_denoise_upper_bound_is_0_75_not_1(self):
        arguments = masked_redraw_arguments(self._valid(denoise=0.75))
        self.assertEqual(arguments["denoise"], 0.75)
        with self.assertRaisesRegex(ValueError, "denoise"):
            masked_redraw_arguments(self._valid(denoise=0.76))

    def test_mask_padding_bounds(self):
        arguments = masked_redraw_arguments(self._valid(mask_padding=512))
        self.assertEqual(arguments["mask_padding"], 512.0)
        with self.assertRaisesRegex(ValueError, "mask_padding"):
            masked_redraw_arguments(self._valid(mask_padding=513))

    def test_mask_feather_bounds(self):
        arguments = masked_redraw_arguments(self._valid(mask_feather=256))
        self.assertEqual(arguments["mask_feather"], 256.0)
        with self.assertRaisesRegex(ValueError, "mask_feather"):
            masked_redraw_arguments(self._valid(mask_feather=257))

    def test_size_must_be_a_multiple_of_8_at_least_256(self):
        arguments = masked_redraw_arguments(self._valid(size=256))
        self.assertEqual(arguments["size"], 256)
        with self.assertRaisesRegex(ValueError, "size"):
            masked_redraw_arguments(self._valid(size=200))
        with self.assertRaisesRegex(ValueError, "size"):
            masked_redraw_arguments(self._valid(size=1001))

    def test_seeds_up_to_16_are_valid_17_is_rejected(self):
        arguments = masked_redraw_arguments(self._valid(seeds=list(range(16))))
        self.assertEqual(len(arguments["seeds"]), 16)
        with self.assertRaisesRegex(ValueError, "seeds"):
            masked_redraw_arguments(self._valid(seeds=list(range(17))))

    def test_seeds_must_be_a_non_empty_list_of_ints(self):
        with self.assertRaisesRegex(ValueError, "seeds"):
            masked_redraw_arguments(self._valid(seeds=[]))
        with self.assertRaisesRegex(ValueError, "seeds"):
            masked_redraw_arguments(self._valid(seeds=[1.5]))
        with self.assertRaisesRegex(ValueError, "seeds"):
            masked_redraw_arguments(self._valid(seeds=[True]))

    def test_denoise_word_resolves_through_the_repair_scope_dials(self):
        dials = _REPAIR_DIALS
        self.assertEqual(
            masked_redraw_arguments(self._valid(denoise="keep"), dials)["denoise"], 0.6)

    def test_unknown_denoise_word_names_the_key_and_word(self):
        dials = _REPAIR_DIALS
        with self.assertRaises(ValueError) as ctx:
            masked_redraw_arguments(self._valid(denoise="blurry"), dials)
        self.assertIn("denoise", str(ctx.exception))
        self.assertIn("blurry", str(ctx.exception))


