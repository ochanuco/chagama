"""Yukari domain, graph and dispatch tests."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from chagama.application.generate import request_graph, validate_request
from chagama.domain.generation.prompt_lint import tags as prompt_tags
from chagama.domain.yukari import prompt_style as ps
from chagama.domain.yukari.costumes import (COSTUME_BAN, COSTUMES,
                                                    OFF_LEGWEAR,
                                                    OFF_LEGWEAR_BAN,
                                                    REMOVING_LEGWEAR,
                                                    REMOVING_LEGWEAR_BAN,
                                                    SHEER_GLOSS_LEGWEAR,
                                                    SHEER_LEGWEAR,
                                                    LegwearState)
from chagama.domain.yukari.expressions import (
    EXPRESSIONS, EYE_QUALITY, Expression, EyeQuality)
from chagama.domain.yukari.poses import POSES
from chagama.domain.yukari.recipe import (
    PART_NAMES, identity_tags, negative, positive, positive_parts,
    render_spec,
)
from chagama.infrastructure.comfyui import anima_graph
from chagama.interfaces import cli

FIXTURES = Path(__file__).parent / "fixtures"
DANCE_FIXTURE = json.loads((FIXTURES / "dance-legwear-t4qwab.json").read_text())
LEGWEAR_SHEER_GLOSS_FINAL = json.loads(
    (FIXTURES / "legwear-sheer-gloss-final.json").read_text())


NAILS_TEXT = "(nail polish:1.3), (light purple nails:1.4), "


def _with_nails(positive_text):
    for costume_start in ("(oversized sweatshirt:1.35)",
                          "(eggplant purple hooded cardigan:1.5)"):
        if costume_start in positive_text:
            return positive_text.replace(
                costume_start, NAILS_TEXT + costume_start, 1)
    raise AssertionError("fixture has no costume block")


def _current_defaults(fixture):
    out = dict(fixture)
    out["positive"] = _with_nails(
        fixture["positive"]
        .replace("masterpiece, best quality, score_7, ", "")
        .replace("hitoshi:0.85)", "hitoshi:1.3)")
        .replace("seven heads tall, ", "seven heads tall, (collarbone:1.2), ")
        .replace("hood down, ", "hood down, (criss-cross halter:1.2), halterneck, "))
    out["negative"] = fixture["negative"].replace(
        "(aged down:1.2)", "(aged down:1.2), (text:1.3), (speech bubble:1.3), (onomatopoeia:1.3), "
        "(english text:1.2), signature, watermark", 1)
    return out


DANCE_FIXTURE = _current_defaults(DANCE_FIXTURE)
LEGWEAR_SHEER_GLOSS_FINAL = {pose: _current_defaults(f)
                             for pose, f in LEGWEAR_SHEER_GLOSS_FINAL.items()}

COFFEE_POSITIVE = (
    "1girl, solo, yuzuki yukari, "
    "vocaloid, voiceroid, (@oshiki hitoshi:1.3), (@yoshikawa hideaki:0.5), "
    "light purple hair, short hair with long locks, very long sidelocks, "
    "(dark greyish purple eyes:1.4), (dark irises:1.2), hair ornament, "
    "(jitome:2.0), (tareme:1.2), (long eyes:1.25), (half-closed "
    "eyes:1.3), (unamused:1.15), (cowboy shot:1.3), (thighs:1.2), (mature "
    "female:1.3), (adult:1.2), (wide hips:1.2), (thick thighs:1.2), (soft "
    "thighs:1.3), (long legs:1.35), (narrow waist:1.25), adult proportions, "
    "long torso, seven heads tall, "
    "(collarbone:1.2), (drinking:1.3), (iced coffee:1.4), "
    "(plastic cup:1.45), (clear cup:1.2), (drinking straw:1.4), (holding "
    "cup:1.35), (straw in mouth:1.25), (standing:1.2), (looking at "
    "viewer:1.1), (oversized sweatshirt:1.35), (white sweatshirt:1.2), "
    "(sleeves past wrists:1.25), (denim shorts:1.3), (sheer black "
    "pantyhose:1.5), (dark violet tint:1.2), (see-through legwear:1.4), (skin "
    "visible through legwear:1.3), (pantyhose:1.4), (subtle sheen on "
    "legwear:1.05), (anime coloring:1.2), simple background, (green "
    "background:1.3), (large eyes:1.2), (round face:1.3), (tareme:1.2), "
    "(thick eyelashes:1.3), (long eyelashes:1.15), (flat color:1.3), "
    "(sketch:1.3), "
    "(traditional media:1.2)"
)

COFFEE_NEGATIVE = (
    "(extra digits:1.5), bad anatomy, bad hands, (detailed:1.3), "
    "(intricate:1.3), (highly detailed:1.3), (fine details:1.2), (colored "
    "lineart:1.4), (colored outline:1.3), (purple lineart:1.2), (skinny:1.3), "
    "(thin legs:1.3), (slender legs:1.2), (slender:1.1), (mug:1.3), (paper "
    "cup:1.2), (hot coffee:1.2), (steam:1.3), (shiny hair:1.4), (hair "
    "highlights:1.2), (watercolor:1.3), (ink wash:1.3), (painterly:1.3), "
    "(gradient:1.5), (soft shading:1.5), (sparkling eyes:1.4), (glitter:1.3), "
    "(multiple highlights:1.3), (gradient eyes:1.2), (speed lines:1.45), "
    "(motion lines:1.4), (emphasis lines:1.4), (light purple eyes:1.3), (bright eyes:1.2), (pink eyes:1.2), (magenta:1.45), (pink "
    "legwear:1.45), (bright purple:1.35), (vivid colors:1.3), (neon:1.3), "
    "(red:1.3), (maroon:1.35), (wine red:1.3), (hood:1.3), (cardigan:1.3), "
    "(opaque legwear:1.3), (latex:1.3), (photorealistic:1.4), "
    "(realistic:1.3), (photo:1.2), (tan skin:1.35), (dark skin:1.3), (brown "
    "legwear:1.4), (brown pantyhose:1.4), (tan:1.2), (beige legwear:1.3), "
    "score_1, score_2, score_3, (fat:1.35), (chubby:1.35), (short legs:1.35), "
    "(muscular:1.3), (toned:1.2), (child:1.3), (loli:1.3), (chibi:1.3), (aged "
    "down:1.2)"
    ", (text:1.3), (speech bubble:1.3), (onomatopoeia:1.3), "
    "(english text:1.2), signature, watermark"
    "(glossy:1.3), (specular highlights:1.3), (reflection:1.2), "
    "(ribbed legwear:1.3), (vertical-striped legwear:1.3), "
)

AMAE_POSITIVE = (
    "1girl, solo, yuzuki yukari, "
    "vocaloid, voiceroid, (@oshiki hitoshi:1.3), (@yoshikawa hideaki:0.5), "
    "light purple hair, short hair with long locks, very long sidelocks, "
    "(dark greyish purple eyes:1.4), (dark irises:1.2), hair ornament, "
    "(jitome:2.0), (tareme:1.2), (long eyes:1.25), (half-closed "
    "eyes:1.3), (unamused:1.15), (cowboy shot:1.3), (thighs:1.2), (mature "
    "female:1.3), (adult:1.2), (wide hips:1.2), (thick thighs:1.2), (soft "
    "thighs:1.3), (long legs:1.35), (narrow waist:1.25), adult proportions, "
    "long torso, seven heads tall, (collarbone:1.2), (standing:1.2), "
    "(v-shaped eyebrows:1.25), (smirk:1.3), (smug:1.2), closed mouth, "
    "(pleading:1.15), (head tilt:1.2), (leaning forward:1.3), "
    "(looking at viewer:1.3), (own hands clasped:1.25), (hands up:1.1), "
    "(oversized sweatshirt:1.35), (white sweatshirt:1.2), (sleeves past "
    "wrists:1.25), (denim shorts:1.3), (sheer black pantyhose:1.5), (dark "
    "violet tint:1.2), (see-through legwear:1.4), (skin visible "
    "through legwear:1.3), (pantyhose:1.4), (subtle sheen on legwear:1.05), (anime "
    "coloring:1.2), simple background, (green background:1.3), (large "
    "eyes:1.2), (round face:1.3), (tareme:1.2), (thick eyelashes:1.3), "
    "(long eyelashes:1.15), (flat color:1.3), (sketch:1.3), "
    "(traditional media:1.2)"
)

AMAE_NEGATIVE = COFFEE_NEGATIVE.replace(
    "(mug:1.3), (paper cup:1.2), (hot coffee:1.2), (steam:1.3), ", "")

STAND_POSITIVE = (
    "1girl, solo, yuzuki yukari, "
    "vocaloid, voiceroid, (@oshiki hitoshi:1.3), (@yoshikawa hideaki:0.5), "
    "light purple hair, short hair with long locks, very long sidelocks, "
    "(dark greyish purple eyes:1.4), (dark irises:1.2), hair ornament, "
    "(jitome:2.0), (tareme:1.2), (long eyes:1.25), (half-closed "
    "eyes:1.3), (unamused:1.15), (from front:1.3), (full body:1.45), (wide "
    "shot:1.3), (thighs:1.1), (mature female:1.3), (adult:1.2), (wide "
    "hips:1.2), (thick thighs:1.2), (soft thighs:1.3), (long legs:1.35), "
    "(narrow waist:1.25), adult proportions, long torso, seven heads tall, "
    "(collarbone:1.2), "
    "(standing:1.5), (own hands together:1.3), (hands up:1.2), (arched "
    "back:1.15), "
    "(v-shaped eyebrows:1.25), (smirk:1.3), (smug:1.2), closed mouth, "
    "(looking at viewer:1.2), "
    "(sneakers:1.3), (white sneakers:1.2), (oversized sweatshirt:1.35), "
    "(white sweatshirt:1.2), (sleeves past wrists:1.25), (denim shorts:1.3), "
    "(sheer black pantyhose:1.5), (dark violet tint:1.2), (see-through "
    "legwear:1.4), (skin visible through legwear:1.3), (pantyhose:1.4), (subtle "
    "sheen on legwear:1.05), (anime coloring:1.2), simple background, (green "
    "background:1.3), (large eyes:1.2), (round face:1.3), (tareme:1.2), "
    "(thick eyelashes:1.3), (long eyelashes:1.15), (flat color:1.3), "
    "(sketch:1.3), "
    "(traditional media:1.2)"
)

STAND_NEGATIVE = (
    "(extra digits:1.5), bad anatomy, bad hands, (detailed:1.3), "
    "(intricate:1.3), (highly detailed:1.3), (fine details:1.2), (colored "
    "lineart:1.4), (colored outline:1.3), (purple lineart:1.2), (skinny:1.3), "
    "(thin legs:1.3), (slender legs:1.2), (slender:1.1), (sitting:1.3), "
    "(cowboy shot:1.2), (upper body:1.2), (shiny hair:1.4), (hair "
    "highlights:1.2), (watercolor:1.3), (ink wash:1.3), (painterly:1.3), "
    "(gradient:1.5), (soft shading:1.5), (sparkling eyes:1.4), (glitter:1.3), "
    "(multiple highlights:1.3), (gradient eyes:1.2), (speed lines:1.45), "
    "(motion lines:1.4), (emphasis lines:1.4), (light purple eyes:1.3), (bright eyes:1.2), (pink eyes:1.2), (magenta:1.45), (pink "
    "legwear:1.45), (bright purple:1.35), (vivid colors:1.3), (neon:1.3), "
    "(red:1.3), (maroon:1.35), (wine red:1.3), (hood:1.3), (cardigan:1.3), "
    "(opaque legwear:1.3), (latex:1.3), (photorealistic:1.4), "
    "(realistic:1.3), (photo:1.2), (tan skin:1.35), (dark skin:1.3), (brown "
    "legwear:1.4), (brown pantyhose:1.4), (tan:1.2), (beige legwear:1.3), "
    "score_1, score_2, score_3, (fat:1.35), (chubby:1.35), (short legs:1.35), "
    "(muscular:1.3), (toned:1.2), (child:1.3), (loli:1.3), (chibi:1.3), (aged "
    "down:1.2)"
    ", (text:1.3), (speech bubble:1.3), (onomatopoeia:1.3), "
    "(english text:1.2), signature, watermark"
    "(glossy:1.3), (specular highlights:1.3), (reflection:1.2), "
    "(ribbed legwear:1.3), (vertical-striped legwear:1.3), "
)

BUST_POSITIVE = (
    "1girl, solo, yuzuki yukari, vocaloid, "
    "voiceroid, (@oshiki hitoshi:1.3), (@yoshikawa hideaki:0.5), "
    "light purple hair, short hair with long locks, very long sidelocks, "
    "(dark greyish purple eyes:1.4), (dark irises:1.2), hair ornament, "
    "(jitome:2.0), (tareme:1.2), (long eyes:1.25), (half-closed eyes:1.35), "
    "(unamused:1.35), (expressionless:1.2), "
    "(from front:1.3), (portrait:1.5), (head and shoulders:1.4), "
    "(upper body:1.35), (face focus:1.3), (mature female:1.3), (adult:1.2), "
    "adult proportions, (collarbone:1.2), (closed mouth:1.2), "
    "(looking at viewer:1.2), (eggplant purple hooded cardigan:1.5), "
    "(dark violet hoodie:1.25), open cardigan, (rabbit hood:1.3), long sleeves, "
    "drawstring, (purple dress:1.25), frills, (sleeves past wrists:1.15), "
    "hood down, (criss-cross halter:1.2), halterneck, "
    "simple background, (green background:1.3), "
    "(large eyes:1.2), (round face:1.3), (tareme:1.2), (thick eyelashes:1.3), "
    "(long eyelashes:1.15), "
    "(flat color:1.3), (sketch:1.3), (traditional media:1.2)"
)

# The accepted render's negative, with the missing ", " between "(aged
# down:1.2)" and "(wavy mouth:1.4)" restored -- the baked negative separates
# them properly, so the two are compared as tag sets, not bytes.
BUST_NEGATIVE_TAG_SOURCE = (
    "(extra digits:1.5), bad anatomy, bad hands, (detailed:1.3), "
    "(intricate:1.3), (highly detailed:1.3), (fine details:1.2), (colored "
    "lineart:1.4), (colored outline:1.3), (purple lineart:1.2), (skinny:1.3),"
    " (thin legs:1.3), (slender legs:1.2), (slender:1.1), (sitting:1.3), "
    "(shiny:1.4), (glossy:1.3), (shiny hair:1.4), (shiny clothes:1.3), "
    "(specular highlights:1.3), (reflection:1.2), (hair highlights:1.2), "
    "(watercolor:1.3), (ink wash:1.3), (painterly:1.3), (gradient:1.5), (soft"
    " shading:1.5), (sparkling eyes:1.4), (glitter:1.3), (multiple "
    "highlights:1.3), (gradient eyes:1.2), (speed lines:1.45), (motion "
    "lines:1.4), (emphasis lines:1.4), (light purple eyes:1.3), (bright eyes:1.2), (pink eyes:1.2), (magenta:1.45), (pink legwear:1.45), "
    "(bright purple:1.35), (vivid colors:1.3), (neon:1.3), (red:1.3), "
    "(maroon:1.35), (wine red:1.3), (black jacket:1.35), (black clothes:1.3),"
    " (black hoodie:1.35), score_1, score_2, score_3, (fat:1.35),"
    " (chubby:1.35), (short legs:1.35), (muscular:1.3), (toned:1.2), "
    "(child:1.3), (loli:1.3), (chibi:1.3), (aged down:1.2), (text:1.3), (speech bubble:1.3), (onomatopoeia:1.3),"
    " (english text:1.2), signature, watermark, (wavy mouth:1.4),"
    " (:3:1.3), (pout:1.3), (pursed lips:1.3), (puckered lips:1.2), "
    "(smile:1.2), "
)

GAO_POSITIVE = (
    "1girl, solo, yuzuki yukari, vocaloid, voiceroid, (@oshiki "
    "hitoshi:1.3), (@yoshikawa hideaki:0.5), light purple hair, short hair "
    "with long locks, very long sidelocks, (dark greyish purple eyes:1.4), "
    "(dark irises:1.2), hair ornament, (jitome:2.0), (tareme:1.2), (long "
    "eyes:1.25), (confident:1.18), (from behind:1.2), (looking back:1.3), "
    "(looking over shoulder:1.2), (upper body:1.3), (mature female:1.3), "
    "(adult:1.2), (narrow waist:1.25), adult proportions, long torso, seven "
    "heads tall, (claw pose:1.45), (hands up:1.25), (open mouth:1.35), "
    "(fang:1.3), (looking at viewer:1.2), (nail polish:1.3), (light purple "
    "nails:1.4), (eggplant purple hooded cardigan:1.5), (dark violet "
    "hoodie:1.25), open cardigan, (rabbit hood:1.3), long sleeves, "
    "drawstring, (purple dress:1.25), frills, (sleeves past wrists:1.15), "
    "hood down, (criss-cross halter:1.2), halterneck, "
    "simple background, (green background:1.3), (large "
    "eyes:1.15), (mature face:1.2), (tareme:1.2), "
    "(thick eyelashes:1.3), (long eyelashes:1.15), (flat color:1.3), "
    "(sketch:1.3), (traditional media:1.2)"
)

GAO_NEGATIVE = (
    "(extra digits:1.5), bad anatomy, bad hands, (detailed:1.3), "
    "(intricate:1.3), (highly detailed:1.3), (fine details:1.2), (colored "
    "lineart:1.4), (colored outline:1.3), (purple lineart:1.2), "
    "(skinny:1.3), (thin legs:1.3), (slender legs:1.2), (slender:1.1), "
    "(sitting:1.3), (shiny:1.4), (glossy:1.3), (shiny hair:1.4), (shiny "
    "clothes:1.3), (specular highlights:1.3), (reflection:1.2), (hair "
    "highlights:1.2), (watercolor:1.3), (ink wash:1.3), (painterly:1.3), "
    "(gradient:1.5), (soft shading:1.5), (sparkling eyes:1.4), "
    "(glitter:1.3), (multiple highlights:1.3), (gradient eyes:1.2), (speed "
    "lines:1.45), (motion lines:1.4), (emphasis lines:1.4), (light purple "
    "eyes:1.3), (bright eyes:1.2), (pink eyes:1.2), (magenta:1.45), (pink "
    "legwear:1.45), (bright purple:1.35), (vivid colors:1.3), (neon:1.3), "
    "(red:1.3), (maroon:1.35), (wine red:1.3), (black jacket:1.35), (black "
    "clothes:1.3), (black hoodie:1.35), score_1, score_2, score_3, "
    "(fat:1.35), (chubby:1.35), (short legs:1.35), (muscular:1.3), "
    "(toned:1.2), (child:1.3), (loli:1.3), (chibi:1.3), (aged down:1.2), "
    "(text:1.3), (speech bubble:1.3), (onomatopoeia:1.3), (english "
    "text:1.2), signature, watermark"
)

COFFEE_POSITIVE = _with_nails(COFFEE_POSITIVE)
AMAE_POSITIVE = _with_nails(AMAE_POSITIVE)
STAND_POSITIVE = _with_nails(STAND_POSITIVE)
BUST_POSITIVE = _with_nails(BUST_POSITIVE)


def _outing_for(pose):
    return {"costume": "outing"} if pose in ("coffee", "step", "amae", "stand") else {}


class PromptTest(unittest.TestCase):
    def test_coffee_positive_matches_the_confirmed_render(self):
        self.assertEqual(positive("coffee", costume="outing", legwear="sheer-gloss"), COFFEE_POSITIVE)

    def test_coffee_negative_matches_the_confirmed_render(self):
        self.assertEqual(negative("coffee", costume="outing", legwear="sheer-gloss"), COFFEE_NEGATIVE)

    def test_amae_positive_matches_the_confirmed_render(self):
        self.assertEqual(positive("amae", costume="outing", legwear="sheer-gloss"), AMAE_POSITIVE)

    def test_amae_negative_drops_the_coffee_vessel_ban(self):
        self.assertEqual(negative("amae", costume="outing", legwear="sheer-gloss"), AMAE_NEGATIVE)

    def test_stand_positive_matches_the_confirmed_render(self):
        self.assertEqual(positive("stand", costume="outing", legwear="sheer-gloss"), STAND_POSITIVE)

    def test_stand_negative_matches_the_confirmed_render(self):
        self.assertEqual(negative("stand", costume="outing", legwear="sheer-gloss"), STAND_NEGATIVE)

    def test_bust_positive_matches_the_confirmed_render(self):
        self.assertEqual(positive("bust", legwear="sheer-gloss"), BUST_POSITIVE)

    def test_bust_negative_tag_set_matches_the_confirmed_render(self):
        self.assertEqual(set(prompt_tags(negative("bust"))),
                         set(prompt_tags(BUST_NEGATIVE_TAG_SOURCE)))

    def test_bust_positive_carries_no_pantyhose_tag(self):
        self.assertNotIn("pantyhose", positive("bust"))

    def test_standard_costume_override_carries_the_gradient_legwear(self):
        text = positive("stand", costume="standard", legwear="opaque")
        self.assertIn("(dark purple pantyhose:1.45), (opaque pantyhose:1.3), "
                      "(gradient legwear:1.2), (purple gradient:1.1), ", text)

    def test_suspender_costume_is_an_orange_tee_and_navy_skirt_over_black_tights(self):
        text = positive("stand", costume="suspender", legwear="opaque")
        self.assertIn(COSTUMES["suspender"], text)
        self.assertIn("(black pantyhose:1.5), (opaque pantyhose:1.4), ", text)
        self.assertIn("(muted orange t-shirt:1.2), (plain t-shirt:1.15), ", text)
        self.assertIn("(navy blue skirt:1.3), (knee-length skirt:1.8), ", text)
        self.assertNotIn("(denim shorts", text)
        neg = negative("stand", costume="suspender")
        self.assertIn(ps.HOOD_BAN, neg)
        self.assertIn(COSTUME_BAN["suspender"], neg)
        self.assertIn("(bright orange:1.2), ", neg)

    def test_costume_bans_stay_with_their_costume(self):
        for costume in COSTUMES:
            if costume == "suspender":
                continue
            self.assertNotIn(COSTUME_BAN["suspender"],
                             negative("stand", costume=costume), costume)

    def test_sheer_gloss_legwear_replaces_the_costume_tights_on_every_costume(self):
        for costume in COSTUMES:
            with self.subTest(costume=costume):
                text = positive("stand", costume=costume, legwear="sheer-gloss")
                self.assertIn(SHEER_GLOSS_LEGWEAR, text)
                self.assertNotIn("(opaque pantyhose", text)
                self.assertIn(COSTUMES[costume], text)

    def test_sheer_gloss_legwear_drops_the_garment_gloss_bans_only(self):
        text = negative("amae", legwear="sheer-gloss")
        for tags in ps.GARMENT_GLOSS_TAGS:
            self.assertNotIn(tags, text)
        self.assertIn("(shiny hair:1.4), (hair highlights:1.2), (watercolor:1.3), ",
                      text)
        self.assertIn(ps.SHEER_BAN + ps.SCORE_BAN, text)
        self.assertIn(ps.GRADIENT_BAN, text)

    def test_sheer_gloss_legwear_leaves_a_bare_leg_pose_alone(self):
        self.assertEqual(positive("bust", legwear="sheer-gloss"), positive("bust"))
        self.assertEqual(negative("bust", legwear="sheer-gloss"), negative("bust"))

    def test_unknown_legwear_is_rejected(self):
        with self.assertRaises(KeyError):
            positive("stand", legwear="fishnet")
        with self.assertRaises(KeyError):
            negative("stand", legwear="fishnet")

    def test_render_spec_carries_legwear_into_both_prompts(self):
        spec = render_spec("amae", 7, "x", legwear="sheer-gloss")
        self.assertEqual(spec.prompts.positive, positive("amae", legwear="sheer-gloss"))
        self.assertEqual(spec.prompts.negative, negative("amae", legwear="sheer-gloss"))

    def test_sheer_legwear_replaces_the_costume_tights_on_every_costume(self):
        for costume in COSTUMES:
            with self.subTest(costume=costume):
                text = positive("stand", costume=costume, legwear="sheer")
                self.assertIn(SHEER_LEGWEAR, text)
                self.assertNotIn("(opaque pantyhose", text)
                self.assertIn(COSTUMES[costume], text)

    def test_sheer_legwear_is_black_purple_without_gradient_words(self):
        self.assertIn("(sheer dark purple pantyhose:1.5), ", SHEER_LEGWEAR)
        self.assertNotIn("gradient", SHEER_LEGWEAR)

    def test_default_legwear_is_sheer(self):
        for pose in ("stand", "amae", "coffee", "dance"):
            with self.subTest(pose=pose):
                self.assertEqual(positive(pose), positive(pose, legwear="sheer"))
                self.assertEqual(negative(pose), negative(pose, legwear="sheer"))

    def test_sheer_legwear_drops_the_garment_gloss_bans_only(self):
        text = negative("amae", legwear="sheer")
        for tags in ps.GARMENT_GLOSS_TAGS:
            self.assertNotIn(tags, text)
        self.assertIn("(shiny hair:1.4), (hair highlights:1.2), (watercolor:1.3), ",
                      text)
        self.assertIn(ps.SHEER_BAN + ps.SHEER_TONE_BAN + ps.SCORE_BAN, text)
        self.assertNotIn(ps.SHEER_TONE_BAN, negative("amae", legwear="sheer-gloss"))
        self.assertIn(ps.GRADIENT_BAN, text)

    def test_sheer_legwear_leaves_a_bare_leg_pose_alone(self):
        self.assertEqual(positive("bust", legwear="sheer"), positive("bust"))
        self.assertEqual(negative("bust", legwear="sheer"), negative("bust"))

    def test_render_spec_carries_sheer_legwear_into_both_prompts(self):
        spec = render_spec("amae", 7, "x", legwear="sheer")
        self.assertEqual(spec.prompts.positive, positive("amae", legwear="sheer"))
        self.assertEqual(spec.prompts.negative, negative("amae", legwear="sheer"))

    def test_standard_costume_negative_drops_the_hood_ban(self):
        self.assertIn("(hood:1.3), (cardigan:1.3), ", negative("stand", costume="outing"))
        self.assertNotIn("(hood:1.3), (cardigan:1.3), ",
                         negative("stand", costume="standard"))

    def test_dance_carries_its_own_expression_and_costume(self):
        text = positive("dance")
        self.assertTrue(text.startswith(ps.QUALITY + ps.CHARACTER + ps.IDENTITY))
        self.assertIn("(:v:1.5)", text)
        self.assertIn(COSTUMES["standard"], text)

    def test_expression_override_swaps_the_eyes_block(self):
        default_text = positive("coffee")
        overridden = positive("coffee", expression="doya")
        self.assertNotEqual(default_text, overridden)
        self.assertIn("(half-closed eyes:1.3), (unamused:1.15), ",
                      overridden)
        self.assertNotIn("(unamused:1.3), (half-closed eyes:1.3), ", overridden)

    def test_gao_positive_matches_the_confirmed_render(self):
        self.assertEqual(positive("gao"), GAO_POSITIVE)

    def test_gao_negative_matches_the_confirmed_render(self):
        self.assertEqual(negative("gao"), GAO_NEGATIVE)

    def test_dance_positive_matches_the_confirmed_render(self):
        self.assertEqual(positive("dance", legwear="sheer-gloss"), DANCE_FIXTURE["positive"])

    def test_dance_negative_matches_the_confirmed_render(self):
        self.assertEqual(negative("dance", legwear="sheer-gloss"), DANCE_FIXTURE["negative"])

    def test_default_prompts_match_the_sheer_gloss_final_look(self):
        for pose, fixture in LEGWEAR_SHEER_GLOSS_FINAL.items():
            with self.subTest(pose=pose):
                self.assertEqual(positive(pose, **_outing_for(pose), legwear="sheer-gloss"),
                                 fixture["positive"])
                self.assertEqual(negative(pose, **_outing_for(pose), legwear="sheer-gloss"),
                                 fixture["negative"])

    def test_eye_shape_leads_the_eyes_part_for_every_expression(self):
        for expression in EXPRESSIONS:
            with self.subTest(expression=expression):
                parts = dict(positive_parts("stand", expression=expression))
                self.assertEqual(parts["eye_base"], ps.EYE_SHAPE)
                self.assertNotIn("jitome", parts["face"])

    def test_face_carries_no_big_eyes(self):
        self.assertNotIn("big eyes", ps.FACE)

    def test_gao_expression_shares_the_smile_eyes(self):
        self.assertEqual(EXPRESSIONS["gao"].eyes, EXPRESSIONS["smile"].eyes)


class EyeQualityTest(unittest.TestCase):
    EXPECTED = {
        "resting": EyeQuality.COLD, "doya": EyeQuality.COLD,
        "v": EyeQuality.COLD, "sleepy": EyeQuality.BLANK,
        "smile": EyeQuality.BLANK, "gao": EyeQuality.BLANK,
        "jitome": EyeQuality.BLANK, "disdain": EyeQuality.BLANK,
        "dubious": EyeQuality.BLANK, "neutral": EyeQuality.BLANK,
        "joy": EyeQuality.BLANK, "amused": EyeQuality.BLANK,
        "anger": EyeQuality.BLANK, "sorrow": EyeQuality.BLANK,
        "surprised": EyeQuality.BLANK,
    }

    def test_eye_quality_mapping_per_expression(self):
        self.assertEqual(
            {name: e.eye_quality for name, e in EXPRESSIONS.items()},
            self.EXPECTED)

    def test_cold_and_blank_tag_text(self):
        self.assertEqual(EYE_QUALITY[EyeQuality.COLD],
                         "(half-closed eyes:1.3), (unamused:1.15), ")
        self.assertEqual(EYE_QUALITY[EyeQuality.BLANK], "")

    def test_eye_quality_component_is_quality_text_plus_extras(self):
        for name, expression in EXPRESSIONS.items():
            with self.subTest(expression=name):
                parts = dict(positive_parts("stand", expression=name))
                self.assertEqual(
                    parts["eye_quality"],
                    EYE_QUALITY[expression.eye_quality] + expression.eyes)

    def test_resting_no_longer_carries_its_own_unamused_weight(self):
        # Unified onto EYE_QUALITY[COLD]: (unamused:1.3) -> (unamused:1.15).
        parts = dict(positive_parts("stand", expression="resting"))
        self.assertEqual(parts["eye_quality"],
                         "(half-closed eyes:1.3), (unamused:1.15), ")

    def test_a_new_expression_defaults_to_cold(self):
        self.assertEqual(Expression(mouth="").eye_quality, EyeQuality.COLD)


class PartsTest(unittest.TestCase):
    def test_parts_concatenate_to_the_confirmed_render_byte_for_byte(self):
        for fixture, pose in ((COFFEE_POSITIVE, "coffee"), (AMAE_POSITIVE, "amae"),
                              (STAND_POSITIVE, "stand"), (BUST_POSITIVE, "bust"),
                              (GAO_POSITIVE, "gao")):
            with self.subTest(pose=pose):
                joined = "".join(text for _, text in positive_parts(
                    pose, **_outing_for(pose), legwear="sheer-gloss"))
                self.assertEqual(joined, fixture)

    def test_every_pose_wears_light_purple_nails_right_after_gesture(self):
        for pose in ("bust", "amae"):
            with self.subTest(pose=pose):
                parts = positive_parts(pose)
                names = [name for name, _ in parts]
                self.assertEqual(names[names.index("gesture") + 1], "nails")
                self.assertEqual(dict(parts)["nails"], ps.NAILS)
                self.assertEqual(
                    positive(pose).count("(light purple nails:1.4)"), 1)

    def test_part_names_match_the_declared_order(self):
        self.assertEqual([name for name, _ in positive_parts("coffee")], [
            "quality", "count", "character", "series", "artist", "identity",
            "eye_base", "eye_quality", "framing_tags",
            "leg_display", "body_build", "action", "mouth", "mood",
            "gesture", "nails", "costume", "legwear", "cutout", "face", "style"])
        self.assertEqual(PART_NAMES, (
            "quality", "identity", "pose", "mouth", "mood", "eyes",
            "gesture", "nails", "costume", "scene", "body", "background", "face",
            "style"))


class IdentityTagsTest(unittest.TestCase):
    EXPECTED = frozenset({
        "light purple hair", "short hair with long locks",
        "very long sidelocks", "dark greyish purple eyes", "hair ornament",
        "tareme", "jitome",
    })

    def test_identity_tags_are_pose_and_costume_independent(self):
        # `tareme` and `jitome` ride in EYE_SHAPE, ahead of every
        # expression's own `e.eyes` -- present for every pose regardless of
        # expression. `outing` carries no cardigan/hood, so those two
        # names never enter the set.
        for pose in ("coffee", "step", "amae", "stand"):
            with self.subTest(pose=pose):
                self.assertEqual(identity_tags(pose, "outing"), self.EXPECTED)

    def test_costume_override_does_not_change_the_identity_set(self):
        self.assertEqual(identity_tags("coffee", "roomwear"), self.EXPECTED)


class PoseTableTest(unittest.TestCase):
    def test_step_pose_defaults(self):
        self.assertIn("step", POSES)
        self.assertEqual(POSES["step"].expression, "resting")
        self.assertEqual(POSES["step"].costume, "standard")

    def test_stand_pose_defaults(self):
        self.assertIn("stand", POSES)
        self.assertEqual(POSES["stand"].expression, "doya")
        self.assertEqual(POSES["stand"].costume, "standard")

    def test_dance_pose_defaults(self):
        self.assertIn("dance", POSES)
        self.assertEqual(POSES["dance"].expression, "v")
        self.assertEqual(POSES["dance"].costume, "standard")
        self.assertEqual(POSES["dance"].legwear_kind, "sheer")

    def test_brush_and_sofa_are_retired(self):
        self.assertNotIn("brush", POSES)
        self.assertNotIn("sofa", POSES)

    def test_bust_pose_defaults(self):
        self.assertIn("bust", POSES)
        self.assertEqual(POSES["bust"].expression, "jitome")
        self.assertEqual(POSES["bust"].costume, "standard")
        self.assertIsNone(POSES["bust"].canvas)
        self.assertFalse(POSES["bust"].legwear)

    def test_gao_pose_defaults(self):
        self.assertIn("gao", POSES)
        self.assertEqual(POSES["gao"].expression, "gao")
        self.assertEqual(POSES["gao"].costume, "standard")
        self.assertEqual(POSES["gao"].canvas, (1024, 1280))
        self.assertFalse(POSES["gao"].legwear)

    def test_avatar_pose_defaults(self):
        self.assertEqual(POSES["avatar"].expression, "neutral")
        self.assertEqual(POSES["avatar"].costume, "standard")
        self.assertFalse(POSES["avatar"].legwear)
        spec = render_spec("avatar", 7, "x")
        self.assertEqual((spec.width, spec.height), POSES["avatar"].canvas)
        self.assertIn("(face focus:1.45)", spec.prompts.positive)
        self.assertIn("(mature face:1.2)", spec.prompts.positive)
        self.assertNotIn("round face", spec.prompts.positive)

    def test_anyo_does_not_depend_on_another_pose(self):
        self.assertEqual(POSES["anyo"].expression, "disdain")
        self.assertEqual(POSES["anyo"].costume, "standard")
        self.assertTrue(POSES["anyo"].legwear)
        spec = render_spec("anyo", 7, "x")
        self.assertEqual((spec.width, spec.height), (1024, 1640))
        self.assertIn("(crossed legs:1.3), (foot out towards viewer:1.4)",
                      spec.prompts.positive)
        self.assertNotIn("claw pose", spec.prompts.positive)
        self.assertNotIn("(sitting:1.3)", spec.prompts.negative)

    def test_pillow_lies_hugging_a_plain_cushion(self):
        self.assertEqual(POSES["pillow"].expression, "dubious")
        self.assertEqual(POSES["pillow"].costume, "standard")
        spec = render_spec("pillow", 3333, "x")
        self.assertEqual((spec.width, spec.height), (896, 1440))
        self.assertIn("(from above:1.35), (dutch angle:1.3), (close-up:1.15), "
                      "(cropped:1.2), ", spec.prompts.positive)
        self.assertNotIn("full body", spec.prompts.positive)
        self.assertTrue(spec.prompts.negative.endswith(
            "watermark, (standing:1.4), (sneakers:1.2), (panties:1.3), "
            "(pantyshot:1.3), (green pillow:1.4), (smile:1.2), "
            "(looking at viewer:1.2), (eyes on pillow:1.45), "
            "(face on cushion:1.45), (stuffed toy:1.4), "
            "(stuffed animal:1.35), (plushie:1.35), (mascot:1.35), "
            "(creature:1.3), (character print:1.35), (dakimakura:1.3)"))

    def test_negative_tail_is_empty_for_other_poses(self):
        for pose, p in POSES.items():
            if pose == "pillow":
                continue
            with self.subTest(pose=pose):
                self.assertEqual(p.negative_tail, "")

    def test_gao_render_spec_uses_the_portrait_canvas(self):
        spec = render_spec("gao", 7, "x")
        self.assertEqual((spec.width, spec.height), (1024, 1280))

    def test_every_pose_defaults_to_sheer_legwear(self):
        for pose, p in POSES.items():
            if not p.legwear:
                continue
            with self.subTest(pose=pose):
                text = positive(pose)
                self.assertIn(SHEER_LEGWEAR, text)
                self.assertNotIn("gradient legwear", text)
                self.assertNotIn("opaque pantyhose", text)


class RenderSpecTest(unittest.TestCase):
    def test_render_spec_fields(self):
        spec = render_spec("coffee", 42, "p")
        self.assertEqual(spec.model_path, ps.MODEL)
        self.assertEqual((spec.width, spec.height), (1024, 1640))
        self.assertEqual(spec.steps, 10)

    def test_pose_canvas_overrides_the_default(self):
        wide = replace(POSES["stand"], canvas=(2048, 1280))
        with patch.dict(POSES, {"_wide": wide}):
            spec = render_spec("_wide", 7, "p")
        self.assertEqual((spec.width, spec.height), (2048, 1280))

    def test_render_spec_default_canvas(self):
        spec = render_spec("stand", 42, "p")
        self.assertEqual(spec.cfg, 2.0)
        self.assertEqual(spec.sampler_name, "euler")
        self.assertEqual(spec.scheduler, "normal")
        self.assertIsNone(spec.hires)

    def test_hires_computes_a_proportional_second_pass(self):
        spec = render_spec("stand", 42, "p", hires=2048)
        self.assertEqual(spec.hires.width, 1280)
        self.assertEqual(spec.hires.height, 2048)
        self.assertEqual(spec.hires.denoise, ps.HIRES_DENOISE)
        self.assertIsNone(spec.hires.positive)
        self.assertEqual(spec.hires.negative, spec.prompts.negative)

    def test_hires_denoise_override(self):
        spec = render_spec("stand", 42, "p", hires=2048, denoise=0.35)
        self.assertEqual(spec.hires.denoise, 0.35)

    def test_denoise_without_hires_is_rejected(self):
        with self.assertRaises(ValueError):
            render_spec("coffee", 42, "p", denoise=0.5)

    def test_hires_on_wide_canvas_pose(self):
        wide = replace(POSES["stand"], canvas=(2048, 1280))
        with patch.dict(POSES, {"_wide": wide}):
            spec = render_spec("_wide", 7, "p", hires=2048)
        self.assertEqual((spec.hires.width, spec.hires.height), (2048, 1280))

    def test_hires_on_square_canvas_pose(self):
        square = replace(POSES["stand"], canvas=(1280, 1280))
        with patch.dict(POSES, {"_square": square}):
            spec = render_spec("_square", 7, "p", hires=2048)
        self.assertEqual((spec.hires.width, spec.hires.height), (2048, 2048))

    def test_bust_render_spec_canvas_and_loras(self):
        spec = render_spec("bust", 7, "p")
        self.assertEqual((spec.width, spec.height), (1280, 2048))
        self.assertEqual(spec.loras, ())

    def test_other_poses_render_spec_loras_stay_empty(self):
        for pose in POSES:
            if pose == "bust":
                continue
            with self.subTest(pose=pose):
                self.assertEqual(render_spec(pose, 42, "p").loras, ())

    def test_dance_render_spec_matches_the_confirmed_render(self):
        spec = render_spec("dance", 42, "p", legwear="sheer-gloss")
        self.assertEqual(spec.prompts.positive, DANCE_FIXTURE["positive"])
        self.assertEqual(spec.prompts.negative, DANCE_FIXTURE["negative"])
        self.assertEqual((spec.width, spec.height),
                         (DANCE_FIXTURE["width"], DANCE_FIXTURE["height"]))


class PlainRenderLegwearDefaultTest(unittest.TestCase):
    """The chimera plain-render path forwards only parameters the request
    sets, so a bare `parameters: {"pose": ...}` must resolve legwear from
    the pose, not from a recipe-wide default."""

    def _plain_positive(self, pose: str, legwear: str | None = None,
                        costume: str | None = None) -> str:
        parameters = {"pose": pose}
        if costume is not None:
            parameters["costume"] = costume
        if legwear is not None:
            parameters["legwear"] = legwear
        generation = {"recipe": "yukari", "parameters": parameters}
        spec = request_graph(generation, 42, "p", render_spec, lambda spec: spec)
        return spec.prompts.positive

    def test_plain_dance_render_defaults_to_sheer_legwear(self):
        self.assertIn(SHEER_LEGWEAR, self._plain_positive("dance"))

    def test_plain_render_defaults_to_sheer_legwear_for_every_pose(self):
        for pose in POSES:
            if not POSES[pose].legwear:
                continue
            with self.subTest(pose=pose):
                self.assertIn(SHEER_LEGWEAR, self._plain_positive(pose))

    def test_plain_stand_render_with_explicit_opaque_legwear_yields_the_opaque_block(self):
        text = self._plain_positive("stand", legwear="opaque", costume="outing")
        self.assertNotIn(SHEER_GLOSS_LEGWEAR, text)
        self.assertIn("(black pantyhose:1.5), (opaque pantyhose:1.4), ", text)


class LegwearStateTest(unittest.TestCase):
    def test_every_pose_defaults_to_worn(self):
        for pose in POSES:
            with self.subTest(pose=pose):
                self.assertEqual(POSES[pose].legwear_state, LegwearState.WORN)

    def test_worn_is_byte_identical_to_the_unstated_default(self):
        for pose in POSES:
            with self.subTest(pose=pose):
                self.assertEqual(positive(pose), positive(pose, legwear_state="worn"))
                self.assertEqual(negative(pose), negative(pose, legwear_state="worn"))

    def test_removing_follows_the_worn_block_with_the_pull_down_tags(self):
        worn = positive("stand", legwear_state="worn")
        removing = positive("stand", legwear_state="removing")
        self.assertIn(SHEER_LEGWEAR, worn)
        self.assertIn(SHEER_LEGWEAR + REMOVING_LEGWEAR, removing)

    def test_off_drops_the_legwear_block_for_bare_legs(self):
        text = positive("stand", legwear_state="off")
        self.assertNotIn("pantyhose", text)
        self.assertIn(OFF_LEGWEAR, text)

    def test_removing_negative_bans_thighhighs_kneehighs_socks(self):
        removing = negative("stand", legwear_state="removing")
        self.assertIn(REMOVING_LEGWEAR_BAN + ps.SCORE_BAN, removing)

    def test_off_negative_bans_every_legwear_kind_and_drops_worn_only_bans(self):
        off = negative("stand", legwear_state="off")
        self.assertIn(OFF_LEGWEAR_BAN + ps.SCORE_BAN, off)

    def test_off_drops_the_sheer_kind_negative_edits(self):
        # `dance`'s legwear_kind is sheer-gloss; OFF means no legwear kind is
        # worn at all, so the sheer-only shine/sheer-ban edits do not apply.
        worn = negative("dance")
        off = negative("dance", legwear_state="off")
        self.assertIn("(shiny:1.4), (glossy:1.3), ", off)
        self.assertNotIn("(shiny:1.4), (glossy:1.3), ", worn)
        self.assertNotIn("(opaque legwear:1.3)", off)

    def test_explicit_legwear_state_overrides_the_pose_default(self):
        # `dance`'s pose default is WORN; an explicit parameter still wins.
        self.assertIn(OFF_LEGWEAR, positive("dance", legwear_state="off"))

    def test_bust_ignores_legwear_state_entirely(self):
        for state in ("worn", "removing", "off"):
            with self.subTest(state=state):
                self.assertEqual(positive("bust"),
                                 positive("bust", legwear_state=state))
                self.assertEqual(negative("bust"),
                                 negative("bust", legwear_state=state))

    def test_unknown_legwear_state_is_rejected(self):
        with self.assertRaises(ValueError):
            positive("stand", legwear_state="half-off")
        with self.assertRaises(ValueError):
            negative("stand", legwear_state="half-off")

    def test_render_spec_carries_legwear_state_into_both_prompts(self):
        spec = render_spec("stand", 7, "x", legwear_state="off")
        self.assertEqual(spec.prompts.positive, positive("stand", legwear_state="off"))
        self.assertEqual(spec.prompts.negative, negative("stand", legwear_state="off"))

    def test_plain_render_path_routes_legwear_state_from_parameters(self):
        generation = {"recipe": "yukari",
                     "parameters": {"pose": "stand", "legwear_state": "off"}}
        spec = request_graph(generation, 42, "p", render_spec, lambda spec: spec)
        self.assertEqual(spec.prompts.positive, positive("stand", legwear_state="off"))

    def test_plain_render_path_defaults_to_worn_when_unstated(self):
        generation = {"recipe": "yukari", "parameters": {"pose": "stand"}}
        spec = request_graph(generation, 42, "p", render_spec, lambda spec: spec)
        self.assertEqual(spec.prompts.positive, positive("stand"))


class GraphTest(unittest.TestCase):
    def test_build_graph_node_shape(self):
        spec = replace(render_spec("coffee", 42, "p"), guided_steps=None)
        graph = anima_graph.build_graph(spec)
        self.assertEqual(graph["1"]["class_type"], "UNETLoader")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], ps.MODEL)
        self.assertEqual(graph["2"]["class_type"], "CLIPLoader")
        self.assertEqual(graph["4"]["class_type"], "VAELoader")
        self.assertEqual(graph["5"]["class_type"], "EmptyLatentImage")
        self.assertEqual(graph["6"]["class_type"], "CLIPTextEncode")
        self.assertEqual(graph["6"]["inputs"]["clip"], ["2", 0])
        self.assertEqual(graph["6"]["inputs"]["text"], spec.prompts.positive)
        self.assertEqual(graph["7"]["inputs"]["text"], spec.prompts.negative)
        self.assertEqual(graph["3"]["class_type"], "KSampler")
        self.assertEqual(graph["3"]["inputs"]["model"], ["1", 0])
        self.assertEqual(graph["3"]["inputs"]["positive"], ["6", 0])
        self.assertEqual(graph["3"]["inputs"]["negative"], ["7", 0])
        self.assertEqual(graph["3"]["inputs"]["latent_image"], ["5", 0])
        self.assertEqual(graph["8"]["class_type"], "VAEDecode")
        self.assertEqual(graph["8"]["inputs"]["samples"], ["3", 0])
        self.assertEqual(graph["8"]["inputs"]["vae"], ["4", 0])
        self.assertEqual(graph["9"]["class_type"], "SaveImage")
        self.assertEqual(graph["9"]["inputs"]["images"], ["8", 0])
        self.assertEqual(graph["9"]["inputs"]["filename_prefix"], "p")

    def test_empty_loras_leaves_the_graph_byte_identical(self):
        spec = render_spec("coffee", 42, "p")
        self.assertEqual(anima_graph.build_graph(spec),
                         anima_graph.build_graph(replace(spec, loras=())))

    def test_loras_chains_lora_loader_model_only_from_the_unet_loader(self):
        spec = replace(render_spec("coffee", 42, "p"),
                       loras=(("anima-sketch-style-chosen.safetensors", 0.8),))
        graph = anima_graph.build_graph(spec)
        self.assertEqual(graph["10"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph["10"]["inputs"], {
            "model": ["1", 0],
            "lora_name": "anima-sketch-style-chosen.safetensors",
            "strength_model": 0.8})
        self.assertEqual(graph["3"]["inputs"]["model"], ["10", 0])

    def test_zero_strength_loras_leave_the_graph_byte_identical(self):
        spec = render_spec("coffee", 42, "p")
        zeroed = replace(spec, loras=(
            ("anima-sketch-style-chosen.safetensors", 0),))
        self.assertEqual(anima_graph.build_graph(spec),
                         anima_graph.build_graph(zeroed))

    def test_zero_strength_lora_drops_out_of_the_chain(self):
        spec = replace(render_spec("coffee", 42, "p"), loras=(
            ("anima-sketch-style-chosen.safetensors", 0),
            ("anima-handdrawn-feel-chosen.safetensors", 0.6)), guided_steps=None)
        graph = anima_graph.build_graph(spec)
        self.assertEqual(graph["10"]["inputs"]["lora_name"],
                         "anima-handdrawn-feel-chosen.safetensors")
        self.assertEqual(graph["10"]["inputs"]["model"], ["1", 0])
        self.assertNotIn("11", graph)

    def test_two_loras_chain_in_order(self):
        spec = replace(render_spec("coffee", 42, "p"), loras=(
            ("anima-sketch-style-chosen.safetensors", 0.8),
            ("anima-handdrawn-feel-chosen.safetensors", 0.6)))
        graph = anima_graph.build_graph(spec)
        self.assertEqual(graph["10"]["inputs"]["model"], ["1", 0])
        self.assertEqual(graph["11"]["inputs"]["model"], ["10", 0])
        self.assertEqual(graph["11"]["inputs"]["lora_name"],
                         "anima-handdrawn-feel-chosen.safetensors")
        self.assertEqual(graph["3"]["inputs"]["model"], ["11", 0])

    def test_hires_appends_a_latent_upscale_and_second_sampler(self):
        spec = replace(render_spec("stand", 42, "p", hires=2048), guided_steps=None)
        graph = anima_graph.build_graph(spec)
        self.assertEqual(graph["10"]["class_type"], "LatentUpscale")
        self.assertEqual(graph["10"]["inputs"], {
            "samples": ["3", 0], "upscale_method": "bicubic",
            "width": spec.hires.width, "height": spec.hires.height,
            "crop": "disabled"})
        self.assertEqual(graph["11"]["class_type"], "KSampler")
        self.assertEqual(graph["11"]["inputs"]["model"],
                         graph["3"]["inputs"]["model"])
        self.assertEqual(graph["11"]["inputs"]["positive"], ["6", 0])
        self.assertEqual(graph["11"]["inputs"]["negative"], ["7", 0])
        self.assertEqual(graph["11"]["inputs"]["latent_image"], ["10", 0])
        self.assertEqual(graph["11"]["inputs"]["denoise"], spec.hires.denoise)
        self.assertEqual(graph["8"]["inputs"]["samples"], ["11", 0])
        self.assertEqual(graph["9"]["inputs"]["images"], ["8", 0])

    def test_hires_with_a_lora_pose_does_not_collide_ids(self):
        spec = replace(render_spec("bust", 7, "p", hires=2048),
                       loras=(("anima-sketch-style-chosen.safetensors", 0.8),),
                       guided_steps=None)
        graph = anima_graph.build_graph(spec)
        self.assertEqual(graph["10"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph["11"]["class_type"], "LatentUpscale")
        self.assertEqual(graph["12"]["class_type"], "KSampler")
        self.assertEqual(graph["3"]["inputs"]["model"], ["10", 0])
        self.assertEqual(graph["12"]["inputs"]["model"], ["10", 0])
        self.assertEqual(graph["11"]["inputs"]["width"], spec.hires.width)
        self.assertEqual(graph["11"]["inputs"]["height"], spec.hires.height)
        self.assertEqual(graph["8"]["inputs"]["samples"], ["12", 0])

class GuidedStepsTest(unittest.TestCase):
    """`render_spec`'s default now draws on SilvermoonMix with `guided_steps`
    set, sampling the leading steps at CFG and the rest at CFG 1.0."""

    def test_default_spec_is_silvermoon_with_guided_steps(self):
        spec = render_spec("coffee", 42, "p")
        self.assertEqual(spec.model_path, ps.MODEL)
        self.assertEqual(spec.model_path,
                         "silvermoon-turbo.safetensors")
        self.assertEqual(spec.guided_steps, 4)

    def test_default_spec_builds_a_two_stage_graph(self):
        spec = render_spec("coffee", 42, "p")
        graph = anima_graph.build_graph(spec)
        self.assertEqual(graph["10"]["class_type"], "KSamplerAdvanced")
        self.assertEqual(graph["10"]["inputs"]["add_noise"], "enable")
        self.assertEqual(graph["10"]["inputs"]["noise_seed"], spec.seed)
        self.assertEqual(graph["10"]["inputs"]["cfg"], spec.cfg)
        self.assertEqual(graph["10"]["inputs"]["start_at_step"], 0)
        self.assertEqual(graph["10"]["inputs"]["end_at_step"], spec.guided_steps)
        self.assertEqual(graph["10"]["inputs"]["return_with_leftover_noise"],
                         "enable")
        self.assertEqual(graph["10"]["inputs"]["latent_image"], ["5", 0])
        self.assertEqual(graph["3"]["class_type"], "KSamplerAdvanced")
        self.assertEqual(graph["3"]["inputs"]["add_noise"], "disable")
        self.assertEqual(graph["3"]["inputs"]["noise_seed"], spec.seed)
        self.assertEqual(graph["3"]["inputs"]["cfg"], 1.0)
        self.assertEqual(graph["3"]["inputs"]["start_at_step"], spec.guided_steps)
        self.assertEqual(graph["3"]["inputs"]["end_at_step"], 10000)
        self.assertEqual(graph["3"]["inputs"]["return_with_leftover_noise"],
                         "disable")
        self.assertEqual(graph["3"]["inputs"]["latent_image"], ["10", 0])
        self.assertEqual(graph["3"]["inputs"]["model"], ["1", 0])
        self.assertEqual(graph["10"]["inputs"]["model"], ["1", 0])
        self.assertEqual(graph["8"]["inputs"]["samples"], ["3", 0])

    def test_guided_steps_none_matches_todays_plain_graph(self):
        spec = render_spec("coffee", 42, "p")
        guided = anima_graph.build_graph(spec)
        plain = anima_graph.build_graph(replace(spec, guided_steps=None))
        self.assertEqual(plain["3"]["class_type"], "KSampler")
        self.assertNotEqual(guided, plain)

    def test_guided_steps_zero_is_byte_identical_to_none(self):
        spec = replace(render_spec("coffee", 42, "p"), guided_steps=None)
        self.assertEqual(anima_graph.build_graph(spec),
                         anima_graph.build_graph(replace(spec, guided_steps=0)))

    def test_guided_steps_at_or_above_total_steps_is_byte_identical_to_none(self):
        spec = replace(render_spec("coffee", 42, "p"), guided_steps=None)
        for guided_steps in (spec.steps, spec.steps + 4):
            with self.subTest(guided_steps=guided_steps):
                self.assertEqual(
                    anima_graph.build_graph(spec),
                    anima_graph.build_graph(replace(spec, guided_steps=guided_steps)))

    def test_guided_steps_with_lora_does_not_collide_ids(self):
        spec = replace(render_spec("coffee", 42, "p"), loras=(
            ("anima-sketch-style-chosen.safetensors", 0.8),))
        graph = anima_graph.build_graph(spec)
        self.assertEqual(graph["10"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph["11"]["class_type"], "KSamplerAdvanced")
        self.assertEqual(graph["11"]["inputs"]["model"], ["10", 0])
        self.assertEqual(graph["11"]["inputs"]["latent_image"], ["5", 0])
        self.assertEqual(graph["3"]["class_type"], "KSamplerAdvanced")
        self.assertEqual(graph["3"]["inputs"]["model"], ["10", 0])
        self.assertEqual(graph["3"]["inputs"]["latent_image"], ["11", 0])

    def test_guided_steps_with_hires_does_not_collide_ids(self):
        spec = render_spec("stand", 42, "p", hires=2048)
        graph = anima_graph.build_graph(spec)
        self.assertEqual(graph["10"]["class_type"], "KSamplerAdvanced")
        self.assertEqual(graph["11"]["class_type"], "LatentUpscale")
        self.assertEqual(graph["11"]["inputs"]["samples"], ["3", 0])
        self.assertEqual(graph["12"]["class_type"], "KSampler")
        self.assertEqual(graph["12"]["inputs"]["latent_image"], ["11", 0])
        self.assertEqual(graph["12"]["inputs"]["cfg"], spec.cfg)
        self.assertEqual(graph["8"]["inputs"]["samples"], ["12", 0])

    def test_guided_steps_with_lora_and_hires_does_not_collide_ids(self):
        spec = replace(render_spec("bust", 7, "p", hires=2048),
                       loras=(("anima-sketch-style-chosen.safetensors", 0.8),))
        graph = anima_graph.build_graph(spec)
        self.assertEqual(graph["10"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph["11"]["class_type"], "KSamplerAdvanced")
        self.assertEqual(graph["12"]["class_type"], "LatentUpscale")
        self.assertEqual(graph["13"]["class_type"], "KSampler")
        self.assertEqual(graph["3"]["inputs"]["model"], ["10", 0])
        self.assertEqual(graph["11"]["inputs"]["model"], ["10", 0])
        self.assertEqual(graph["13"]["inputs"]["model"], ["10", 0])
        self.assertEqual(graph["12"]["inputs"]["samples"], ["3", 0])
        self.assertEqual(graph["13"]["inputs"]["latent_image"], ["12", 0])
        self.assertEqual(graph["8"]["inputs"]["samples"], ["13", 0])

    def test_render_guided_steps_patch_applies(self):
        from chagama.domain.generation.patches import (
            apply_patches, parse_patches,
        )
        spec = render_spec("coffee", 42, "p")
        patches = parse_patches([
            {"target": "render.guided_steps", "op": "set", "value": 0,
             "reason": "disable"}])
        result = apply_patches(spec, patches)
        self.assertEqual(result.guided_steps, 0)
        graph = anima_graph.build_graph(result)
        self.assertEqual(graph["3"]["class_type"], "KSampler")

    def test_render_guided_steps_patch_rejects_negative_and_non_int(self):
        from chagama.domain.generation.patches import parse_patches
        for value in (-1, 2.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_patches([
                        {"target": "render.guided_steps", "op": "set",
                         "value": value, "reason": "r"}])


class ValidateRequestTest(unittest.TestCase):
    def _request(self, **generation):
        return {
            "schema_version": 1,
            "request": {"count": 1, "instruction": "test", "seeds": [42]},
            "generation": {"recipe": "yukari",
                          "parameters": {"pose": "coffee"}, **generation},
            "semantic": {"summary": "test arm"},
        }

    def test_yukari_is_accepted(self):
        validate_request(self._request())

    def test_hires_is_accepted_for_yukari(self):
        request = self._request()
        request["generation"]["parameters"]["hires"] = 2048
        validate_request(request)  # must not raise

    def test_denoise_is_accepted_for_yukari(self):
        request = self._request()
        request["generation"]["parameters"]["hires"] = 2048
        request["generation"]["parameters"]["denoise"] = 0.5
        validate_request(request)  # must not raise

    def test_legwear_state_is_accepted_for_yukari(self):
        request = self._request()
        request["generation"]["parameters"]["legwear_state"] = "removing"
        validate_request(request)  # must not raise

    def test_layerdiffuse_is_rejected_for_yukari(self):
        request = self._request()
        request["generation"]["parameters"]["layerdiffuse"] = True
        with self.assertRaises(SystemExit):
            validate_request(request)

    def test_unsupported_recipe_is_rejected(self):
        request = {
            "schema_version": 1,
            "request": {"count": 1, "instruction": "test", "seeds": [42]},
            "generation": {"recipe": "yukari-sketch",
                          "parameters": {"pose": "lounge", "expression": "doya"}},
            "semantic": {"summary": "test arm"},
        }
        with self.assertRaises(SystemExit):
            validate_request(request)


class CliTest(unittest.TestCase):
    def test_yukari_prompt_json_needs_no_clients(self):
        output = io.StringIO()
        with patch.object(cli, "ChimeraClient") as chimera_class, \
                redirect_stdout(output):
            cli.main(["yukari", "prompt", "--pose", "amae", "--json"])
        chimera_class.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["positive"], positive("amae"))
        self.assertEqual(payload["negative"], negative("amae"))

    def test_yukari_prompt_accepts_legwear_state(self):
        output = io.StringIO()
        with patch.object(cli, "ChimeraClient"), redirect_stdout(output):
            cli.main(["yukari", "prompt", "--pose", "stand",
                     "--legwear-state", "off", "--json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["positive"], positive("stand", legwear_state="off"))
        self.assertEqual(payload["negative"], negative("stand", legwear_state="off"))


class AngleFramingOverrideTest(unittest.TestCase):
    def test_unset_angle_and_framing_keep_every_pose_byte_for_byte(self):
        for pose in POSES:
            with self.subTest(pose=pose):
                self.assertEqual(positive(pose, angle=None, framing=None),
                                 positive(pose))

    def test_angle_replaces_the_camera_and_keeps_the_action_tags(self):
        parts = dict(positive_parts("gao", angle="from_below"))
        self.assertEqual(
            parts["framing_tags"],
            "(from below:1.3), (looking back:1.3), "
            "(looking over shoulder:1.2), (upper body:1.3), ")

    def test_angle_words_join_in_axis_order(self):
        for value in ("dutch,from_above", ["dutch", "from_above"]):
            with self.subTest(value=value):
                parts = dict(positive_parts("stand", angle=value))
                self.assertTrue(parts["framing_tags"].startswith(
                    "(from above:1.35), (dutch angle:1.3), "))

    def test_two_words_on_one_axis_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "軸ごとに 1 語"):
            positive("stand", angle="from_above,from_below")

    def test_unknown_angle_word_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "語彙にありません"):
            positive("stand", angle="from_space")

    def test_framing_swaps_the_shot_tags(self):
        parts = dict(positive_parts("coffee", framing="full"))
        self.assertTrue(parts["framing_tags"].endswith(
            "(full body:1.45), (wide shot:1.3), "))

    def test_framing_takes_its_own_canvas_over_the_pose_canvas(self):
        self.assertEqual((render_spec("gao", 7, "p").width,
                          render_spec("gao", 7, "p").height), (1024, 1280))
        spec = render_spec("gao", 7, "p", framing="full")
        self.assertEqual((spec.width, spec.height), (ps.WIDTH, ps.HEIGHT))
        spec = render_spec("stand", 7, "p", framing="bust")
        self.assertEqual((spec.width, spec.height), (1280, 2048))

    def test_unknown_framing_is_rejected(self):
        with self.assertRaises(ValueError):
            positive("stand", framing="extreme")

    def test_request_parameters_reach_the_prompt(self):
        generation = {"recipe": "yukari", "parameters": {
            "pose": "stand", "angle": "from_below", "framing": "cowboy"}}
        graph = request_graph(generation, 7, "p", render_spec,
                              lambda spec: {"positive": spec.prompts.positive})
        self.assertIn("(from below:1.3), (cowboy shot:1.3), ", graph["positive"])
        self.assertNotIn("(from front:1.3)", graph["positive"])


if __name__ == "__main__":
    unittest.main()
