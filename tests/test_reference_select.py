"""The reference selection rules, pinned on five real pictures.

`fixtures/reference/ref0N.json` holds what the Anima Tagger and d1 answered for
each test picture (candidate scores, every kept tag, d1's P(yes) for the tags
it was asked about); `groups.json` holds the checkpoint's groups those
candidates come from.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from chagama.domain.generation.patches import apply_patches, parse_patches
from chagama.domain.generation.prompt_lint import tags as bare_tags
from chagama.domain.yukari import reference
from chagama.domain.yukari.recipe import identity_tags, render_spec
from chagama.infrastructure.reference.d1 import QUESTION, questions, version_tuple

FIXTURES = Path(__file__).parent / "fixtures" / "reference"
GROUPS = json.loads((FIXTURES / "groups.json").read_text(encoding="utf-8"))


def picture(stem: str) -> dict:
    return json.loads((FIXTURES / f"{stem}.json").read_text(encoding="utf-8"))


def plan_for(stem: str) -> reference.Plan:
    data = picture(stem)
    return reference.plan(data["scores"], data["kept"], GROUPS, data["verdicts"])


EXPECTED = {
    "ref01": ("bust", {
        "framing_tags": ["upper body"],
        "gesture": ["looking at viewer", "hand on own cheek", "head rest", "head tilt"],
        "mouth": ["open mouth", "smile", "blush", ":d"]}),
    "ref02": ("coffee", {
        "framing_tags": ["feet out of frame", "from above"],
        "gesture": ["looking at viewer", "sitting", "between legs",
                    "hand between legs", "looking up"],
        "mouth": ["smile", "closed mouth", "blush", "light blush", "light smile"],
        "legwear": ["pantyhose", "black pantyhose"]}),
    "ref03": ("stand", {
        "framing_tags": ["full body"],
        "gesture": ["looking at viewer", "sitting", "knees up"],
        "mouth": ["blush"],
        "legwear": ["thighhighs", "black thighhighs"]}),
    "ref04": ("stand", {
        "gesture": ["looking at viewer", "standing", "standing on one leg",
                    "hand up", "arm up", "leg up"],
        "mouth": ["one eye closed", "smile", "blush", "closed mouth"],
        "legwear": ["thighhighs", "black thighhighs"]}),
    "ref05": ("coffee", {
        "framing_tags": ["cowboy shot"],
        "gesture": ["hands on own chest", "looking at viewer", "standing",
                    "legs together"]}),
}


class PinnedPicturesTest(unittest.TestCase):
    def test_pose_and_applied_tags(self):
        for stem, (pose, applied) in EXPECTED.items():
            with self.subTest(stem):
                result = plan_for(stem)
                self.assertEqual(result.pose, pose)
                self.assertEqual(result.applied, applied)

    def test_d1_is_asked_only_about_undecided_candidates(self):
        for stem in EXPECTED:
            with self.subTest(stem):
                data = picture(stem)
                asked = reference.questions(data["scores"], data["kept"], GROUPS)
                self.assertEqual(sorted(asked), sorted(data["verdicts"]))
                for tag in asked:
                    self.assertNotIn(tag, data["kept"])
                    self.assertGreaterEqual(data["scores"][tag], reference.MIN_SCORE)

    def test_patches_apply_and_keep_the_identity(self):
        for stem in EXPECTED:
            with self.subTest(stem):
                result = plan_for(stem)
                spec = apply_patches(render_spec(result.pose, 1, "t"),
                                     parse_patches(result.patches))
                parts = dict(spec.positive_parts)
                self.assertEqual(parts["action"], "")
                positive = set(bare_tags(spec.prompts.positive))
                self.assertLessEqual(identity_tags(result.pose), positive)
                for part, tags in result.applied.items():
                    self.assertEqual(parts[part], "".join(
                        f"{reference.prompt_form(part, tag)}, " for tag in tags))
                negative = set(bare_tags(spec.prompts.negative))
                used = {tag for tags in result.applied.values() for tag in tags}
                self.assertFalse(used & negative)

    def test_a_smile_clears_the_cold_eyes(self):
        result = plan_for("ref01")
        spec = apply_patches(render_spec("bust", 1, "t"), parse_patches(result.patches))
        eyes = dict(spec.positive_parts)["eye_quality"]
        self.assertEqual(eyes, "(half-closed eyes:1.35), ")
        self.assertNotIn("(smile:1.2)", spec.prompts.negative)

    def test_without_a_camera_tag_the_pose_keeps_its_framing(self):
        result = plan_for("ref04")
        self.assertFalse(any(patch["target"] == "prompt.positive.framing_tags"
                             for patch in result.patches))

    def test_no_smile_keeps_the_cold_eyes(self):
        result = plan_for("ref05")
        self.assertFalse(any(patch["target"] == "prompt.positive.eye_quality"
                             for patch in result.patches))

    def test_a_bust_drops_legwear(self):
        data = picture("ref01")
        kept = data["kept"] + ["black pantyhose"]
        result = reference.plan(data["scores"], kept, GROUPS, data["verdicts"])
        self.assertEqual(result.pose, "bust")
        self.assertNotIn("legwear", result.applied)
        self.assertIn("black pantyhose", result.dropped)

    def test_dropped_is_every_kept_tag_not_applied(self):
        result = plan_for("ref02")
        self.assertIn("1girl", result.dropped)
        self.assertNotIn("smile", result.dropped)


class RulesTest(unittest.TestCase):
    groups = {"pose": ["sitting", "on floor", "on couch"],
              "viewpoint_angle": ["looking at viewer", "from above"],
              "framing": ["full body", "cowboy shot"],
              "expression": ["smile"], "hair_color": ["purple hair"]}

    def test_denied_and_ungrouped_tags_are_never_candidates(self):
        found = reference.candidates(self.groups)
        self.assertEqual({tag: part for tag, part in found.items()
                          if tag not in reference.EXTRA_GESTURE}, {
            "sitting": "gesture", "full body": "framing_tags",
            "cowboy shot": "framing_tags", "smile": "mouth",
            "looking at viewer": "gesture", "from above": "framing_tags"})

    def test_listed_ungrouped_gestures_are_candidates(self):
        found = reference.candidates(self.groups)
        self.assertEqual(found["hand between legs"], "gesture")
        self.assertEqual(found["hands on own chest"], "gesture")

    def test_acceptance_needs_kept_or_score_and_verdict(self):
        scores = {"sitting": 0.3, "smile": 0.3, "cowboy shot": 0.05}
        accepted = reference.extract(
            scores, {"full body"}, self.groups,
            {"sitting": 0.6, "smile": 0.4, "cowboy shot": 0.9})
        self.assertEqual(accepted["gesture"], ["sitting"])
        self.assertEqual(accepted["mouth"], [])
        self.assertEqual(accepted["framing_tags"], ["full body"])

    def test_pose_follows_the_framing(self):
        self.assertEqual(reference.choose_pose({"full body", "cowboy shot"}), "stand")
        self.assertEqual(reference.choose_pose({"feet out of frame"}), "coffee")
        self.assertEqual(reference.choose_pose({"upper body"}, {"sitting"}), "bust")
        self.assertEqual(reference.choose_pose(set()), "bust")

    def test_a_whole_body_posture_picks_stand_when_framing_is_silent(self):
        self.assertEqual(reference.choose_pose(set(), {"leg up"}), "stand")
        self.assertEqual(reference.choose_pose({"from above"}, {"sitting"}), "stand")
        self.assertEqual(reference.choose_pose(set(), {"head tilt"}), "bust")

    def test_an_empty_part_is_filled_by_append(self):
        data = picture("ref05")
        result = reference.plan(data["scores"], data["kept"] + ["smile"],
                                GROUPS, data["verdicts"])
        mouth = [patch for patch in result.patches
                 if patch["target"] == "prompt.positive.mouth"]
        self.assertEqual(mouth, [{"target": "prompt.positive.mouth", "op": "append",
                                  "value": "smile, ", "reason": reference.REASON}])


class CandidateRowsTest(unittest.TestCase):
    def test_rows_list_every_shown_candidate_with_its_numbers(self):
        data = picture("ref02")
        proposal = plan_for("ref02")
        rows = reference.candidate_rows(data["scores"], data["kept"], GROUPS,
                                        data["verdicts"], proposal)
        by_tag = {row["tag"]: row for row in rows}
        self.assertEqual(by_tag["looking up"]["d1"], data["verdicts"]["looking up"])
        self.assertTrue(by_tag["looking up"]["applied"])
        self.assertIsNone(by_tag["sitting"]["d1"])
        for row in rows:
            self.assertEqual(set(row), {"tag", "part", "score", "kept", "d1", "applied"})
            self.assertNotIn(row["tag"], reference.DENIED)
            self.assertTrue(row["score"] >= reference.MIN_LISTED_SCORE
                            or row["kept"] or row["applied"])
        for part, tags in proposal.applied.items():
            for tag in tags:
                self.assertEqual(by_tag[tag]["part"], part)

    def test_a_bust_lists_legwear_unapplied(self):
        data = picture("ref01")
        kept = data["kept"] + ["black pantyhose"]
        proposal = reference.plan(data["scores"], kept, GROUPS, data["verdicts"])
        rows = reference.candidate_rows(data["scores"], kept, GROUPS,
                                        data["verdicts"], proposal)
        row = next(row for row in rows if row["tag"] == "black pantyhose")
        self.assertEqual((row["part"], row["kept"], row["applied"]),
                         ("legwear", True, False))


class SelectionTest(unittest.TestCase):
    rows = [{"tag": "sitting", "part": "gesture"},
            {"tag": "from above", "part": "framing_tags"}]

    def test_a_listed_selection_passes(self):
        reference.check_selection(self.rows, "stand",
                                  {"gesture": ["sitting"], "framing_tags": ["from above"]})

    def test_a_tag_outside_the_candidates_or_its_part_fails(self):
        with self.assertRaisesRegex(ValueError, "gesture:smile"):
            reference.check_selection(self.rows, "stand", {"gesture": ["smile"]})
        with self.assertRaisesRegex(ValueError, "mouth:sitting"):
            reference.check_selection(self.rows, "stand", {"mouth": ["sitting"]})

    def test_an_unknown_pose_fails(self):
        with self.assertRaisesRegex(ValueError, "pose"):
            reference.check_selection(self.rows, "anyo", {})

    def test_build_keeps_the_given_order_and_weights_framing(self):
        plan = reference.build("coffee", {"framing_tags": ["from above"],
                                          "gesture": ["sitting", "head tilt"]})
        framing = next(patch for patch in plan.patches
                       if patch["target"] == "prompt.positive.framing_tags")
        gesture = next(patch for patch in plan.patches
                       if patch["target"] == "prompt.positive.gesture")
        self.assertEqual(framing["value"], "(from above:1.3), ")
        self.assertEqual(gesture["value"], "sitting, head tilt, ")
        self.assertEqual(plan.applied, {"framing_tags": ["from above"],
                                        "gesture": ["sitting", "head tilt"]})


class D1QuestionTest(unittest.TestCase):
    def test_each_tag_is_one_noul_question(self):
        asked = questions(["sitting"])
        self.assertEqual(asked, {"sitting": {
            "type": "noul", "instructions": QUESTION.format(tag="sitting")}})

    def test_version_tuple(self):
        self.assertEqual(version_tuple("5.16.1"), (5, 16))
        self.assertLess(version_tuple("4.57.0.dev0"), (5, 14))


if __name__ == "__main__":
    unittest.main()


class FusedAttentionTest(unittest.TestCase):
    def test_disable_routes_only_d1_hybrid_modules_off_flash(self):
        import sys
        import types

        from chagama.infrastructure.reference import d1

        hybrid = types.ModuleType("transformers_modules.d1_hyphen_3B.abc.hybrid")
        hybrid._fused = lambda q: True
        other = types.ModuleType("transformers_modules.other.hybrid_extra")
        other._fused = lambda q: True
        sys.modules[hybrid.__name__] = hybrid
        sys.modules[other.__name__] = other
        try:
            d1.disable_fused_attention()
            self.assertFalse(hybrid._fused(object()))
            self.assertTrue(other._fused(object()))
        finally:
            del sys.modules[hybrid.__name__]
            del sys.modules[other.__name__]

    def test_flash_attention_built_reads_torch(self):
        import types

        from chagama.infrastructure.reference.d1 import flash_attention_built

        def fake(available):
            return types.SimpleNamespace(backends=types.SimpleNamespace(cuda=types.SimpleNamespace(
                is_flash_attention_available=lambda: available)))

        self.assertTrue(flash_attention_built(fake(True)))
        self.assertFalse(flash_attention_built(fake(False)))
        self.assertFalse(flash_attention_built(types.SimpleNamespace(
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace()))))
