"""Tests for turning a request row's JSON `options` into use-case kwargs."""

from __future__ import annotations

import unittest

from chagama.application.deliver import RECIPE_DEFAULT
from chagama.application.request_options import (
    deliver_arguments,
    dof_arguments,
    lens_arguments,
    lighting_arguments,
    photo_arguments,
    stage_arguments,
    redraw_arguments,
)
from chagama.domain.yukari.delivery_style import Light

# Synthetic dial vocabularies for exercising resolve_dial()'s word lookup --
# not any recipe's own published words.
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
            "keep_regions": [], "keep_strength": 0.25, "cut": False})

    def test_canvas_options_parse(self):
        arguments = redraw_arguments({
            "method": "canvas", "denoise": 0.5, "size": 2048, "route": "latent",
            "finalizer": "model", "upscale": "lanczos",
            "keep_regions": [[0, 0, 0.5, 0.5]], "keep_strength": 0.3})
        self.assertEqual(arguments, {
            "method": "canvas", "denoise": 0.5, "size": 2048,
            "latent_route": True, "finalizer": "model", "upscale": "lanczos",
            "keep_regions": [[0.0, 0.0, 0.5, 0.5]], "keep_strength": 0.3, "cut": False})
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
            "method": "hires", "hires": 2048, "denoise": 0.45, "cut": False})
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
            "method": "light", "light": Light("moon", "nw"), "cut": False})
        self.assertEqual(
            redraw_arguments({"method": "light", "scene": "moon",
                              "from": "se"})["light"], Light("moon", "se"))
        for options in ({}, {"scene": "noon"}, {"scene": "moon", "from": "up"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                redraw_arguments({"method": "light", **options})

    def test_every_method_takes_cut(self):
        for options in ({"method": "canvas"}, {"method": "hires", "hires": 2048},
                        {"method": "light", "scene": "moon"}):
            with self.subTest(method=options["method"]):
                self.assertIs(redraw_arguments(options)["cut"], False)
                self.assertIs(redraw_arguments({**options, "cut": True})["cut"], True)
                with self.assertRaises(ValueError):
                    redraw_arguments({**options, "cut": "yes"})

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

    def test_see_through_is_a_gain_from_one_to_five(self):
        self.assertEqual(deliver_arguments({"see_through": 3})["see_through"], 3.0)
        self.assertNotIn("see_through", deliver_arguments({"see_through": None}))
        self.assertNotIn("see_through", deliver_arguments({"see_through": 1}))
        for bad in (0.5, 6, True, "3"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "see_through"):
                deliver_arguments({"see_through": bad})

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
            "scope": {"figure": True, "outline": True, "backdrop": True},
            "photo_blur": None})

    def test_photo_keys_fill_the_photo_blur_with_defaults(self):
        arguments = dof_arguments({"focus": [0.5, 0.25], "tilt": 0.5})
        self.assertEqual(arguments["photo_blur"], {
            "stand_at": None, "nearer": 0.0, "tilt": 0.5, "bokeh": 0.0,
            "focal_length": "85mm"})
        with self.assertRaises(ValueError):
            dof_arguments({"focus": [0.5, 0.25], "focal_length": "50mm"})
        with self.assertRaises(ValueError):
            dof_arguments({"focus": [0.5, 0.25], "bokeh": 2})

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


class StageArgumentsTest(unittest.TestCase):
    def test_a_photo_takes_place_and_height(self):
        self.assertEqual(stage_arguments({"photo": "pic"}), {
            "photo_id": "pic", "place": (0.5, 0.95), "height": 0.7})
        self.assertEqual(stage_arguments({"photo": "pic", "place": [0.6, 1.2],
                                          "height": 1.1})["place"], (0.6, 1.2))

    def test_a_pattern_takes_backdrop_and_shadow(self):
        self.assertEqual(stage_arguments({}), {
            "backdrop": "dots", "shadow_from": None, "inherit_shadow": True,
            "light": None, "inherit_light": True, "frame": None})
        self.assertEqual(stage_arguments({"backdrop": "#102030", "shadow_from": None,
                                          "light": None}), {
            "backdrop": "#102030", "shadow_from": None, "inherit_shadow": False,
            "light": None, "inherit_light": False, "frame": None})
        self.assertEqual(stage_arguments({"light": {"scene": "moon"}})["light"],
                         Light("moon", "nw"))
        self.assertEqual(stage_arguments({"shadow_from": "nw"})["shadow_from"], "nw")

    def test_a_frame_takes_its_box_and_fills_the_catalog_defaults(self):
        frame = stage_arguments({"frame": {"box": [0.1, 0.25, 1, 1],
                                           "sides": {"left": "out"}}})["frame"]
        self.assertEqual(stage_arguments({"frame": {"box": [0, 0, 1, 1], "sides": {
            "bottom": "behind"}}})["frame"]["sides"]["bottom"], "behind")
        self.assertEqual(frame, {
            "box": [0.1, 0.25, 1.0, 1.0],
            "sides": {"top": "out", "right": "cut", "bottom": "cut", "left": "out"},
            "line": {"color": "#4a3152", "width": 0.6, "pattern": "solid"},
            "rim": {"color": "#2e2236", "width": 0.25}, "paper": "#ffffff",
            "canvas": None, "expand": False, "strokes": []})
        frame = stage_arguments({"frame": {
            "box": [0, 0, 1, 1], "line": {"color": "#AABBCC", "width": 0}, "rim": None,
            "canvas": [-0.1, -0.1, 1.1, 1.1],
            "strokes": [{"mode": "restore", "width": 3, "points": [[0.5, 1.2]]}]}})["frame"]
        self.assertIsNone(frame["line"])
        self.assertIsNone(frame["rim"])
        self.assertEqual(frame["canvas"], [-0.1, -0.1, 1.1, 1.1])
        self.assertEqual(frame["strokes"],
                         [{"mode": "restore", "width": 3.0, "points": [[0.5, 1.2]]}])
        frame = stage_arguments({"frame": {"box": [0, 0, 1, 1], "line": {"width": 1},
                                           "rim": {"color": "#000000"}}})["frame"]
        self.assertEqual(frame["line"], {"color": "#4a3152", "width": 1.0, "pattern": "solid"})
        self.assertEqual(stage_arguments({"frame": {"box": [0, 0, 1, 1], "line": {
            "pattern": "slash"}}})["frame"]["line"]["pattern"], "slash")
        self.assertEqual(frame["rim"], {"color": "#000000", "width": 0.25})
        frame = stage_arguments({"frame": {"box": [0, 0, 1, 1], "canvas": [None, -0.1, None, None],
                                           "expand": {"left": True}}})["frame"]
        self.assertEqual(frame["canvas"], [None, -0.1, None, None])
        self.assertEqual(frame["expand"],
                         {"top": False, "right": False, "bottom": False, "left": True})
        self.assertIsNone(stage_arguments({"frame": {
            "box": [0, 0, 1, 1], "canvas": [None, None, None, None]}})["frame"]["canvas"])

    def test_bad_frames_fail(self):
        box = [0, 0, 1, 1]
        for bad in ({}, {"box": [0, 0, 1]}, {"box": [0.5, 0, 0.4, 1]},
                    {"box": [-1.5, 0, 1, 1]}, {"box": box, "sides": {"middle": "cut"}},
                    {"box": box, "sides": {"top": "in"}},
                    {"box": box, "sides": {"top": "over"}}, {"box": box, "line": {"color": "red", "width": 1}},
                    {"box": box, "line": {"color": "#000000", "width": 6}},
                    {"box": box, "rim": {"color": "#000000", "alpha": 1}},
                    {"box": box, "line": {"pattern": "dots"}},
                    {"box": box, "rim": {"pattern": "slash"}},
                    {"box": box, "paper": "white"}, {"box": box, "zoom": 1},
                    {"box": box, "expand": "no"}, {"box": box, "expand": {"up": False}},
                    {"box": box, "expand": {"top": 0}},
                    {"box": box, "canvas": [0.5, None, 0.4, None]},
                    {"box": box, "canvas": [None, None, None]},
                    {"box": box, "strokes": [{"mode": "smudge", "width": 1, "points": [[0, 0]]}]},
                    {"box": box, "strokes": [{"mode": "erase", "width": 0, "points": [[0, 0]]}]},
                    {"box": box, "strokes": [{"mode": "erase", "width": 1, "points": []}]},
                    {"box": box, "strokes": [{"mode": "erase", "width": 1, "points": [[0, 3]]}]},
                    {"box": box, "strokes": [{"mode": "erase", "width": 1,
                                              "points": [[0, 0]] * 2001}]},
                    {"box": box, "strokes": [{"mode": "erase", "width": 1,
                                              "points": [[0, 0]] * 2000}] * 11}):
            with self.subTest(bad=str(bad)[:80]), self.assertRaises(ValueError):
                stage_arguments({"frame": bad})
        with self.assertRaises(ValueError):
            stage_arguments({"photo": "p", "frame": {"box": box}})

    def test_the_two_sets_do_not_mix_and_bad_values_fail(self):
        for bad in ({"photo": "p", "backdrop": "dots"}, {"place": [0.5, 0.5]},
                    {"backdrop": "nope"}, {"shadow_from": "up"}, {"photo": ""},
                    {"photo": "p", "height": 3}, {"zoom": 1},
                    {"photo": "p", "light": None}, {"light": {"scene": "noon"}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                stage_arguments(bad)


class LightingArgumentsTest(unittest.TestCase):
    def test_defaults_and_bounds(self):
        self.assertEqual(lighting_arguments({}),
                         {"direction": None, "strength": 0.7, "scene": None})
        self.assertEqual(lighting_arguments({"direction": "ne", "strength": 1, "scene": "moon"}),
                         {"direction": "ne", "strength": 1.0, "scene": "moon"})
        for bad in ({"direction": "up"}, {"strength": 1.5}, {"scene": "noon"}, {"glow": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                lighting_arguments(bad)


class PhotoArgumentsTest(unittest.TestCase):
    def test_defaults_take_the_catalog_values(self):
        self.assertEqual(photo_arguments({"photo": "pic"}), {
            "photo_id": "pic", "place": (0.5, 0.95), "height": 0.7,
            "stand_at": None, "nearer": 0.0, "focus": "subject", "f_number": 2.8,
            "tilt": 0.0, "bokeh": 0.0, "focal_length": "85mm"})
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
                         ({"photo": "p", "bokeh": 1.5}, "bokeh"),
                         ({"photo": "p", "blur": 1}, "blur")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, key):
                photo_arguments(bad)
        with self.assertRaisesRegex(ValueError, "object"):
            photo_arguments([])
