"""Akari invariants: she draws every shared pose in her own identity, eyes,
costume and tights, and nothing of yukari's comes along."""

from __future__ import annotations

import unittest

from chagama.application.catalog import build_catalog
from chagama.domain.akari.recipe import RECIPE
from chagama.domain.anima import style as st
from chagama.domain.anima.legwear import LEGWEARS
from chagama.domain.anima.poses import POSES as SHARED_POSES
from chagama.domain.generation.prompt_lint import tags as prompt_tags
from chagama.domain.yukari.recipe import RECIPE as YUKARI_RECIPE

GIT = {"commit": "abc", "branch": "main", "dirty": False}
EVERY_POSE = sorted(SHARED_POSES) + [RECIPE.character.persona_pose]

IDENTITY = frozenset({
    "kizuna akari", "white hair", "very long hair", "twin braids",
    "low twin braids", "deep blue eyes", "pale pink pupils", "tareme",
    "droopy eyes",
})
# Yukari's own words: her name, colours, ジト目 and garments.
YUKARI_WORDS = ("yuzuki yukari", "purple", "violet", "eggplant", "jitome",
                "unamused", "rabbit hood", "hair ornament", "halter")
YUKARI_BANS = ("(black clothes:", "(black hoodie:", "(light purple eyes:",
               "(pink eyes:", "(purple lineart:", "(magenta:")


class AkariIdentityTest(unittest.TestCase):
    def test_identity_guard_covers_every_pose(self):
        for pose in EVERY_POSE:
            with self.subTest(pose=pose):
                self.assertEqual(RECIPE.identity_tags(pose), IDENTITY)

    def test_dropping_the_identity_part_trips_the_guard(self):
        for pose in EVERY_POSE:
            with self.subTest(pose=pose):
                kept = "".join(text for name, text in RECIPE.positive_parts(pose)
                               if name not in ("character", "identity"))
                missing = RECIPE.identity_tags(pose) - set(prompt_tags(kept))
                self.assertTrue({"kizuna akari", "white hair",
                                 "deep blue eyes"} <= missing)


class AkariCarriesNothingOfYukariTest(unittest.TestCase):
    def test_no_yukari_words_in_any_positive(self):
        for pose in EVERY_POSE:
            for expression in [None, *RECIPE.expressions]:
                with self.subTest(pose=pose, expression=expression):
                    text = RECIPE.positive(pose, expression=expression)
                    for word in YUKARI_WORDS:
                        self.assertNotIn(word, text)

    def test_no_yukari_bans_in_any_negative(self):
        for pose in EVERY_POSE:
            with self.subTest(pose=pose):
                text = RECIPE.negative(pose)
                for ban in YUKARI_BANS:
                    self.assertNotIn(ban, text)

    def test_every_pose_wears_her_costume(self):
        for pose in SHARED_POSES:
            with self.subTest(pose=pose):
                self.assertEqual(RECIPE.poses[pose].costume, "half_jacket")
                self.assertIn("(very dark brown jacket:1.4)",
                              dict(RECIPE.positive_parts(pose))["costume"])

    def test_half_jacket_has_the_white_frilled_hem_front_and_back(self):
        frill = ("(frilled dress:1.2), (white frilled hem:1.3), "
                 "(white lace trim on skirt hem:1.2), ")
        for angle in (None, "from_behind"):
            with self.subTest(angle=angle):
                kwargs = {"angle": angle} if angle else {}
                costume = dict(RECIPE.positive_parts("amae", **kwargs))["costume"]
                self.assertIn(frill, costume)


class AkariLegwearTest(unittest.TestCase):
    def test_default_legwear_is_white_with_an_orange_tint(self):
        legwear = dict(RECIPE.positive_parts("stand"))["legwear"]
        self.assertTrue(legwear.startswith("(white pantyhose:1.5), "))
        self.assertIn("(light orange tint:1.1)", legwear)

    def test_every_kind_is_white_with_an_orange_tint(self):
        for kind in LEGWEARS:
            with self.subTest(legwear=kind):
                legwear = dict(RECIPE.positive_parts("stand", legwear=kind))["legwear"]
                self.assertIn("white pantyhose", legwear)
                self.assertIn("light orange tint", legwear)

    def test_negative_keeps_the_tights_off_purple_and_black(self):
        for kind in LEGWEARS:
            with self.subTest(legwear=kind):
                negative = RECIPE.negative("stand", legwear=kind)
                self.assertIn("(purple legwear:", negative)
                self.assertIn("(black pantyhose:", negative)


class AkariStyleTest(unittest.TestCase):
    def test_her_1990s_tag_rides_on_the_shared_style(self):
        style = dict(RECIPE.positive_parts("stand"))["style"]
        self.assertEqual(style, st.STYLE + r", (1990s \(style\):0.7)")

    def test_yukari_style_is_the_shared_style(self):
        self.assertEqual(dict(YUKARI_RECIPE.positive_parts("stand"))["style"],
                         st.STYLE)


class AkariCatalogTest(unittest.TestCase):
    def test_catalog_lists_every_shared_pose_and_her_persona(self):
        akari = {recipe["name"]: recipe
                 for recipe in build_catalog(GIT)["recipes"]}["akari"]
        self.assertEqual([pose["name"] for pose in akari["poses"]],
                         sorted(SHARED_POSES))
        self.assertEqual(akari["base_pose"]["name"], "akari_persona")
        self.assertEqual(akari["costumes"], ["half_jacket"])


if __name__ == "__main__":
    unittest.main()
