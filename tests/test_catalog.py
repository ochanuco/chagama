"""Tests for the published recipe catalog.

All collaborators are fakes or pure functions: this suite never opens a
network socket.
"""

from __future__ import annotations

import base64
import io
import re
import unittest

from PIL import Image

from comfyui_recipes.application.catalog import build_catalog, publish_catalog
from comfyui_recipes.application.generate import validate_request
from comfyui_recipes.application.request_options import (
    _KNOWN_DELIVER_OPTIONS,
    _KNOWN_REDRAW_OPTIONS,
    _KNOWN_REPAIR_OPTIONS,
    deliver_arguments,
    redraw_arguments,
    repair_arguments,
)
from comfyui_recipes.domain.generation.patches import (
    NUMBER_TARGETS,
    STRING_TARGETS,
    TEXT_TARGETS,
    parse_patches,
)
from comfyui_recipes.domain.yukari.delivery_style import BACKDROP_LABELS
from comfyui_recipes.infrastructure.imaging.backdrops import PATTERNS as BACKDROP_PATTERNS

GIT = {"commit": "abc123", "branch": "dev/catalog-publish", "dirty": False}

_DIAL_WORD = re.compile(r"^[a-z][a-z0-9-]*$")

# scope -> the real option/target keys a dial in that scope may name.
_DIAL_SCOPE_KEYS = {
    "deliver": _KNOWN_DELIVER_OPTIONS,
    "redraw": _KNOWN_REDRAW_OPTIONS["canvas"],
    "repair": _KNOWN_REPAIR_OPTIONS,
    "patches": set(NUMBER_TARGETS),
}


def _request(recipe: str, parameters: dict) -> dict:
    return {
        "schema_version": 1,
        "request": {"count": 1, "instruction": "test"},
        "generation": {"recipe": recipe, "parameters": parameters},
        "semantic": {"summary": "test arm"},
    }


# One legal dummy value per KNOWN_PARAMETERS key, used to probe whether
# validate_request accepts or rejects it for a given recipe.
_DUMMY_VALUES = {
    "hires": 1024, "denoise": 0.5, "costume": "default", "character": "yukari",
    "character_id": "char-1", "arm": "a", "expression": "doya",
    "legwear": "sheer-gloss", "legwear_state": "removing",
}


class BuildCatalogTest(unittest.TestCase):
    def test_schema_version_and_git_metadata(self):
        catalog = build_catalog(GIT)
        self.assertEqual(catalog["schema_version"], 3)
        self.assertEqual(catalog["git_commit"], "abc123")
        self.assertEqual(catalog["git_branch"], "dev/catalog-publish")
        self.assertIs(catalog["git_dirty"], False)
        self.assertIn("generated_at", catalog)

    def test_is_pure_and_takes_no_io(self):
        # Calling twice must not raise and must agree on everything but the
        # timestamp -- build_catalog does no I/O of its own.
        first = build_catalog(GIT)
        second = build_catalog(GIT)
        first.pop("generated_at")
        second.pop("generated_at")
        self.assertEqual(first, second)

    def test_every_recipe_has_every_pose_with_non_empty_prompts_and_canvas(self):
        catalog = build_catalog(GIT)
        names = {recipe["name"] for recipe in catalog["recipes"]}
        self.assertEqual(names, {"yukari"})
        for recipe in catalog["recipes"]:
            self.assertTrue(recipe["poses"], recipe["name"])
            for pose in recipe["poses"]:
                with self.subTest(recipe=recipe["name"], pose=pose["name"]):
                    self.assertTrue(pose["positive"])
                    self.assertTrue(pose["negative"])
                    canvas = pose["canvas"]
                    self.assertEqual(len(canvas), 2)
                    for side in canvas:
                        self.assertIsInstance(side, int)
                        self.assertGreater(side, 0)

    def test_yukari_poses_carry_an_expression(self):
        catalog = build_catalog(GIT)
        by_name = {recipe["name"]: recipe for recipe in catalog["recipes"]}
        for pose in by_name["yukari"]["poses"]:
            self.assertIn("expression", pose)
            self.assertIsInstance(pose["expression"], str)

    def test_yukari_recipe_lists_its_expressions(self):
        catalog = build_catalog(GIT)
        by_name = {recipe["name"]: recipe for recipe in catalog["recipes"]}
        self.assertIn("expressions", by_name["yukari"])

    def test_recipe_parameters_agree_with_validate_request(self):
        catalog = build_catalog(GIT)
        for recipe in catalog["recipes"]:
            name = recipe["name"]
            pose = recipe["poses"][0]["name"]
            for key in recipe["parameters"]["rejected"]:
                with self.subTest(recipe=name, key=key, expect="rejected"):
                    request = _request(
                        name, {"pose": pose, key: _DUMMY_VALUES[key]})
                    with self.assertRaises(SystemExit):
                        validate_request(request)
            for key in recipe["parameters"]["allowed"]:
                if key == "pose":
                    continue
                with self.subTest(recipe=name, key=key, expect="allowed"):
                    request = _request(
                        name, {"pose": pose, key: _DUMMY_VALUES[key]})
                    validate_request(request)  # must not raise

    def test_patches_block_lists_every_target(self):
        catalog = build_catalog(GIT)
        patches = catalog["patches"]
        self.assertEqual(set(patches["text"]["targets"]), set(TEXT_TARGETS))
        self.assertEqual(set(patches["number"]), set(NUMBER_TARGETS))
        self.assertEqual(set(patches["string"]), set(STRING_TARGETS))
        for target in NUMBER_TARGETS:
            self.assertEqual(patches["number"][target]["op"], "set")
            self.assertIsInstance(patches["number"][target]["constraints"], str)
        for target in STRING_TARGETS:
            self.assertEqual(patches["string"][target]["op"], "set")
            self.assertIsNone(patches["string"][target]["values"])

    def test_patches_block_carries_the_part_target_and_identity_override(self):
        catalog = build_catalog(GIT)
        patches = catalog["patches"]
        self.assertEqual(patches["text"]["part_target"], "prompt.positive.<part>")
        self.assertIn("identity_override", patches["overrides"])
        self.assertTrue(patches["overrides"]["identity_override"])

    def test_yukari_publishes_legwear_state_words_and_pose_defaults(self):
        catalog = build_catalog(GIT)
        by_name = {recipe["name"]: recipe for recipe in catalog["recipes"]}
        yukari = by_name["yukari"]
        self.assertEqual(set(yukari["legwear_state"]), {"worn", "removing", "off"})
        by_pose = {pose["name"]: pose for pose in yukari["poses"]}
        self.assertEqual(by_pose["stand"]["legwear_state"], "worn")
        self.assertIsNone(by_pose["bust"]["legwear_state"])

    def test_yukari_poses_carry_parts_that_join_into_positive(self):
        catalog = build_catalog(GIT)
        by_name = {recipe["name"]: recipe for recipe in catalog["recipes"]}
        for pose in by_name["yukari"]["poses"]:
            with self.subTest(pose=pose["name"]):
                self.assertIn("parts", pose)
                self.assertTrue(pose["parts"])
                for part in pose["parts"]:
                    self.assertEqual(set(part), {"name", "text"})
                joined = "".join(part["text"] for part in pose["parts"])
                self.assertEqual(joined, pose["positive"])

    def test_recipe_level_parts_and_identity_tags(self):
        catalog = build_catalog(GIT)
        by_name = {recipe["name"]: recipe for recipe in catalog["recipes"]}
        self.assertEqual(
            by_name["yukari"]["parts"],
            ["quality", "count", "character", "series", "artist", "identity",
             "eye_base", "eye_quality", "framing_tags",
             "leg_display", "body_build", "action", "mouth", "mood",
             "gesture", "nails", "costume", "legwear", "cutout", "face", "style"])
        self.assertEqual(
            by_name["yukari"]["part_groups"],
            {"quality": ["quality", "count"],
             "identity": ["character", "series", "artist", "identity"],
             "pose": ["action"], "mouth": ["mouth"], "mood": ["mood"],
             "eyes": ["eye_base", "eye_quality"], "gesture": ["gesture"], "nails": ["nails"],
             "costume": ["costume", "legwear"],
             "scene": ["framing_tags", "leg_display"],
             "body": ["body_build"], "background": ["cutout"],
             "face": ["face"], "style": ["style"]})
        tags = by_name["yukari"]["identity_tags"]
        self.assertTrue(tags)
        self.assertEqual(tags, sorted(set(tags)))
        for tag in tags:
            self.assertNotIn("(", tag)
            self.assertNotIn(":", tag)


class DialsTest(unittest.TestCase):
    def test_every_recipe_publishes_a_dials_block(self):
        catalog = build_catalog(GIT)
        for recipe in catalog["recipes"]:
            with self.subTest(recipe=recipe["name"]):
                self.assertIn("dials", recipe)
                self.assertLessEqual(set(recipe["dials"]), set(_DIAL_SCOPE_KEYS))

    def test_yukari_dial_scopes(self):
        catalog = build_catalog(GIT)
        by_name = {recipe["name"]: recipe for recipe in catalog["recipes"]}
        self.assertEqual(set(by_name["yukari"]["dials"]),
                         {"deliver", "redraw", "patches"})

    def test_dial_keys_are_real_option_keys_of_their_scope(self):
        catalog = build_catalog(GIT)
        for recipe in catalog["recipes"]:
            for scope, options in recipe["dials"].items():
                with self.subTest(recipe=recipe["name"], scope=scope):
                    self.assertLessEqual(set(options), _DIAL_SCOPE_KEYS[scope])

    def test_dial_words_match_the_naming_pattern(self):
        catalog = build_catalog(GIT)
        for recipe in catalog["recipes"]:
            for scope, options in recipe["dials"].items():
                for key, words in options.items():
                    for word in words:
                        with self.subTest(
                                recipe=recipe["name"], scope=scope, key=key, word=word):
                            self.assertRegex(word, _DIAL_WORD)

    def test_dial_values_satisfy_their_option_s_own_range(self):
        # Runs each dial value through the exact validator its option uses at
        # request time -- a value out of range raises there, same as a
        # caller-supplied number out of range would.
        catalog = build_catalog(GIT)
        for recipe in catalog["recipes"]:
            for scope, options in recipe["dials"].items():
                for key, words in options.items():
                    for word in words:
                        with self.subTest(
                                recipe=recipe["name"], scope=scope, key=key, word=word):
                            if scope == "deliver":
                                deliver_arguments({key: word}, options)
                            elif scope == "redraw":
                                redraw_arguments(
                                    {"method": "canvas", key: word}, options)
                            elif scope == "repair":
                                repair_arguments({"parts": ["hands"], key: word}, options)
                            else:
                                parse_patches(
                                    [{"target": key, "op": "set", "value": word,
                                      "reason": "catalog pin"}],
                                    options)


class DeliverSectionTest(unittest.TestCase):
    def section(self) -> dict:
        by_name = {recipe["name"]: recipe for recipe in build_catalog(GIT)["recipes"]}
        return by_name["yukari"]["deliver"]

    def test_defaults_are_the_delivery_recipe_and_valid_options(self):
        defaults = self.section()["defaults"]
        self.assertEqual(defaults, {"repin": True, "stroke_light": "n",
                                    "backdrop": "dots"})
        self.assertLessEqual(set(defaults), _KNOWN_DELIVER_OPTIONS)
        deliver_arguments(defaults)

    def test_publishes_the_outline_default_and_limits_without_a_dof(self):
        section = self.section()
        self.assertEqual(section["outlines"], {
            "default": [{"color": "#ffffff", "width": 0.8},
                        {"color": "#885b80", "width": 1.04}],
            "max_count": 6, "max_width": 5})
        self.assertNotIn("dof", section)
        self.assertNotIn("dof", section["defaults"])
        self.assertNotIn("outlines", section["defaults"])

    def test_publishes_the_lens_section_at_the_catalog_root(self):
        lens = build_catalog(GIT)["lens"]
        self.assertEqual(lens["strength"], {"min": -1.0, "max": 1.0, "default": 0.7})
        self.assertEqual(lens["center"]["default"], [0.5, 0.5])
        self.assertIsInstance(lens["center"]["help"], str)
        self.assertEqual(lens["shape"], {"values": ["fisheye", "sphere"],
                                         "default": "fisheye"})
        self.assertEqual((lens["radius"]["min"], lens["radius"]["max"],
                          lens["radius"]["default"]), (0.05, 1.0, 0.3))

    def test_publishes_the_photo_section_at_the_catalog_root(self):
        photo = build_catalog(GIT)["photo"]
        self.assertEqual((photo["place"]["default"], photo["place"]["max_y"]),
                         ([0.5, 0.95], 1.5))
        self.assertEqual((photo["height"]["min"], photo["height"]["max"],
                          photo["height"]["default"]), (0.05, 1.5, 0.7))
        self.assertIsNone(photo["stand_at"]["default"])
        self.assertEqual(photo["focus"]["default"], "subject")
        self.assertEqual((photo["tilt"]["min"], photo["tilt"]["max"],
                          photo["tilt"]["default"]), (-1.0, 1.0, 0.0))
        self.assertEqual(photo["f_number"]["default"], 2.8)

    def test_publishes_the_dof_section_at_the_catalog_root(self):
        dof = build_catalog(GIT)["dof"]
        self.assertEqual(dof["f_number"], {
            "min": 1.4, "max": 22, "default": 2.8,
            "stops": [1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.5, 4.0, 4.5,
                      5.0, 5.6, 6.3, 7.1, 8.0, 9.0, 10.0, 11.0, 13.0, 14.0,
                      16.0, 18.0, 20.0, 22.0]})
        self.assertEqual(dof["scope"],
                         {"figure": True, "outline": True, "backdrop": True})
        self.assertEqual(dof["viewfinder"],
                         {"values": ["off", "on", "both"], "default": "off"})
        self.assertIsInstance(dof["focus"], str)
        self.assertEqual(dof["guide_radius_per_f"], 0.0417)

    def test_publishes_the_stroke_choices_and_solid_backdrop(self):
        section = self.section()
        self.assertEqual(section["stroke_light"],
                         ["even", "n", "ne", "e", "se", "s", "sw", "w", "nw"])
        self.assertEqual(section["backdrop_color"], "#ffffff")


class BackdropsBlockTest(unittest.TestCase):
    def test_lists_every_pattern_in_registry_order_with_name_label_and_thumbnail(self):
        catalog = build_catalog(GIT)
        backdrops = catalog["backdrops"]
        self.assertEqual([entry["name"] for entry in backdrops], list(BACKDROP_PATTERNS))
        for entry in backdrops:
            with self.subTest(name=entry["name"]):
                self.assertEqual(set(entry), {"name", "label", "thumbnail"})
                self.assertEqual(entry["label"], BACKDROP_LABELS[entry["name"]])
                self.assertTrue(entry["thumbnail"].startswith("data:image/png;base64,"))
                raw = base64.b64decode(entry["thumbnail"].split(",", 1)[1])
                image = Image.open(io.BytesIO(raw))
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (120, 192))


class PublishCatalogTest(unittest.TestCase):
    def test_puts_the_given_document_to_the_branchs_catalog_path(self):
        calls = []

        class ManagementFake:
            def put_catalog(self, recipe_ref, catalog):
                calls.append((recipe_ref, catalog))
                return {"ok": True}

        document = {"schema_version": 1}
        response = publish_catalog(ManagementFake(), GIT, catalog=document)
        self.assertEqual(response, {"ok": True})
        self.assertEqual(calls, [("dev/catalog-publish", document)])

    def test_builds_the_document_itself_when_none_is_given(self):
        calls = []

        class ManagementFake:
            def put_catalog(self, recipe_ref, catalog):
                calls.append((recipe_ref, catalog))
                return {}

        publish_catalog(ManagementFake(), GIT)
        self.assertEqual(len(calls), 1)
        recipe_ref, catalog = calls[0]
        self.assertEqual(recipe_ref, "dev/catalog-publish")
        self.assertEqual(catalog["git_branch"], "dev/catalog-publish")


class RedrawSectionTest(unittest.TestCase):
    def test_the_redraw_section_publishes_light_and_per_method_defaults(self):
        redraw = build_catalog(GIT)["recipes"][0]["redraw"]
        light = redraw["light"]
        self.assertEqual(light["scenes"], ["sunset", "moon"])
        self.assertEqual(light["from"],
                         ["e", "n", "ne", "nw", "s", "se", "sw", "w"])
        self.assertEqual(light["default_from"], "nw")
        self.assertEqual(set(redraw["defaults"]), {"canvas", "hires"})
        self.assertEqual(set(redraw["defaults"]["canvas"]),
                         {"denoise", "size", "route"})
        self.assertEqual(set(redraw["defaults"]["hires"]), {"hires_denoise"})

    def test_only_the_current_request_kinds_are_published(self):
        recipe = build_catalog(GIT)["recipes"][0]
        self.assertIn("deliver", recipe)
        self.assertIn("redraw", recipe)
        self.assertNotIn("finalize", recipe)


if __name__ == "__main__":
    unittest.main()
