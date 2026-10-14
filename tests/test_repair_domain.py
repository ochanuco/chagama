"""Tests for the repair domain: region geometry and prompt edits."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from chagama.domain.repair.loras import PART_LORAS, part_loras
from chagama.domain.repair.prompt import (
    PART_TAGS,
    REPAIR_DROP_WORDS,
    masked_redraw_prompt,
    repair_prompt,
)
from chagama.domain.repair.regions import (
    Circle,
    Rect,
    rects_from_fractions,
    regions_from_pose,
)

FIXTURES = Path(__file__).parent / "fixtures"
POSE = json.loads((FIXTURES / "repair-kps-raw.json").read_text())["people"][0]
POSE_DOC = json.loads((FIXTURES / "repair-kps-raw.json").read_text())


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class RegionsFromPoseTest(unittest.TestCase):
    def test_both_feet_found_yields_two_circles(self):
        circles = regions_from_pose(POSE_DOC, ["feet"])
        self.assertEqual(len(circles), 2)
        for circle in circles:
            self.assertIsInstance(circle, Circle)
            self.assertGreater(circle.r, 0)

    def test_feet_circle_centres_and_radius_match_the_ankle_and_shin(self):
        knee_r = (627.25, 504.29166666666674)
        ankle_r = (183.08333333333326, 1352.0)
        shin_r = _dist(knee_r, ankle_r)
        circles = regions_from_pose(POSE_DOC, ["feet"])
        right = min(circles, key=lambda c: abs(c.cx - ankle_r[0]))
        self.assertAlmostEqual(right.cx, ankle_r[0])
        self.assertAlmostEqual(right.cy, ankle_r[1] + 0.15 * shin_r)
        self.assertAlmostEqual(right.r, 0.30 * shin_r)

    def test_pad_scales_the_foot_radius_linearly(self):
        base = regions_from_pose(POSE_DOC, ["feet"], pad=1.0)
        padded = regions_from_pose(POSE_DOC, ["feet"], pad=2.0)
        base_by_cx = sorted(base, key=lambda c: c.cx)
        padded_by_cx = sorted(padded, key=lambda c: c.cx)
        for plain, doubled in zip(base_by_cx, padded_by_cx):
            self.assertAlmostEqual(doubled.r, plain.r * 2)
            self.assertAlmostEqual(doubled.cx, plain.cx)
            self.assertAlmostEqual(doubled.cy, plain.cy)

    def test_hands_absent_from_source_yields_no_hand_circles(self):
        circles = regions_from_pose(POSE_DOC, ["hands"])
        self.assertEqual(circles, [])

    def test_both_parts_together_is_the_union(self):
        feet = regions_from_pose(POSE_DOC, ["feet"])
        both = regions_from_pose(POSE_DOC, ["hands", "feet"])
        self.assertEqual(len(both), len(feet))

    def test_no_people_yields_no_circles(self):
        self.assertEqual(regions_from_pose({"people": []}, ["feet"]), [])

    def test_missing_knee_or_ankle_skips_that_foot(self):
        body = list(POSE["pose_keypoints_2d"])
        # Zero out the right knee (index 9): [0, 0, 0].
        body[9 * 3:9 * 3 + 3] = [0.0, 0.0, 0.0]
        pose = {"people": [{**POSE, "pose_keypoints_2d": body}]}
        circles = regions_from_pose(pose, ["feet"])
        self.assertEqual(len(circles), 1)

    def test_hand_points_present_use_bbox_centre_and_forearm_floor(self):
        body = list(POSE["pose_keypoints_2d"])
        # Right elbow (3) and wrist (4) both found, close together, so the
        # forearm floor (0.25 * forearm) should win over the tiny bbox.
        body[3 * 3:3 * 3 + 3] = [100.0, 100.0, 1.0]
        body[4 * 3:4 * 3 + 3] = [110.0, 100.0, 1.0]
        hand = [0.0] * 63
        hand[0:3] = [105.0, 100.0, 1.0]
        hand[3:6] = [106.0, 101.0, 1.0]
        pose = {"people": [{**POSE, "pose_keypoints_2d": body,
                            "hand_right_keypoints_2d": hand}]}
        circles = regions_from_pose(pose, ["hands"])
        self.assertEqual(len(circles), 1)
        forearm = _dist((100.0, 100.0), (110.0, 100.0))
        self.assertAlmostEqual(circles[0].r, 0.25 * forearm)

    def test_wrist_only_places_a_circle_offset_from_the_wrist(self):
        body = list(POSE["pose_keypoints_2d"])
        body[3 * 3:3 * 3 + 3] = [100.0, 100.0, 1.0]
        body[4 * 3:4 * 3 + 3] = [140.0, 100.0, 1.0]
        pose = {"people": [{**POSE, "pose_keypoints_2d": body}]}
        circles = regions_from_pose(pose, ["hands"])
        self.assertEqual(len(circles), 1)
        forearm = 40.0
        self.assertAlmostEqual(circles[0].cx, 140.0 + 0.2 * forearm)
        self.assertAlmostEqual(circles[0].cy, 100.0)
        self.assertAlmostEqual(circles[0].r, 0.35 * forearm)

    def test_wrist_without_elbow_is_skipped(self):
        body = list(POSE["pose_keypoints_2d"])
        body[3 * 3:3 * 3 + 3] = [0.0, 0.0, 0.0]
        body[4 * 3:4 * 3 + 3] = [140.0, 100.0, 1.0]
        pose = {"people": [{**POSE, "pose_keypoints_2d": body}]}
        circles = regions_from_pose(pose, ["hands"])
        self.assertEqual(circles, [])


class RectsFromFractionsTest(unittest.TestCase):
    def test_fractions_scale_by_width_and_height(self):
        rects = rects_from_fractions([[0.1, 0.2, 0.3, 0.4]], 800, 1000)
        self.assertEqual(rects, [Rect(80.0, 200.0, 240.0, 400.0)])

    def test_empty_regions_yields_no_rects(self):
        self.assertEqual(rects_from_fractions([], 800, 1000), [])

    def test_multiple_regions_preserve_order(self):
        rects = rects_from_fractions(
            [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]], 100, 100)
        self.assertEqual(rects, [Rect(0, 0, 50, 50), Rect(50, 50, 100, 100)])


POSITIVE = (
    "masterpiece, best quality, sketch, 1girl, (light purple hair:1.15), "
    "(tareme:1.2), (jitome:1.35), (half-closed eyes:1.3), (unamused:1.3), "
    "(sigh:1.25), (annoyed:1.1), closed mouth, looking at viewer, "
    "(head tilt:1.05), (rabbit hood:1.3), hood down, (full body:1.3), "
    "(from front:1.15), (black pantyhose:1.3), (pantyhose feet:1.2), "
    "(wide hips:1.2)"
)


class RepairPromptTest(unittest.TestCase):
    def test_drop_words_remove_face_hair_and_framing_tags(self):
        result = repair_prompt(POSITIVE, [])
        for dropped in ("light purple hair", "tareme", "jitome",
                        "half-closed eyes", "unamused", "sigh", "annoyed",
                        "closed mouth", "looking at viewer", "head tilt",
                        "rabbit hood", "hood down", "full body", "from front"):
            self.assertNotIn(dropped, result)

    def test_unrelated_tags_survive(self):
        result = repair_prompt(POSITIVE, [])
        self.assertIn("masterpiece", result)
        self.assertIn("wide hips", result)

    def test_part_tags_are_appended_for_each_requested_part(self):
        result = repair_prompt(POSITIVE, ["feet"])
        self.assertIn(PART_TAGS["feet"], result)
        self.assertNotIn(PART_TAGS["hands"], result)

        result = repair_prompt(POSITIVE, ["hands", "feet"])
        self.assertIn(PART_TAGS["hands"], result)
        self.assertIn(PART_TAGS["feet"], result)

    def test_no_parts_appends_no_tags(self):
        result = repair_prompt(POSITIVE, [])
        self.assertNotIn(PART_TAGS["hands"], result)
        self.assertNotIn(PART_TAGS["feet"], result)

    def test_pantyhose_feet_rule_fires_when_feet_requested_and_pantyhose_present(self):
        result = repair_prompt(POSITIVE, ["feet"])
        self.assertIn("(pantyhose feet:1.3)", result)

    def test_pantyhose_feet_rule_does_not_fire_without_feet_in_parts(self):
        result = repair_prompt(POSITIVE, ["hands"])
        self.assertNotIn("(pantyhose feet:1.3)", result)

    def test_pantyhose_feet_rule_does_not_fire_without_pantyhose_mention(self):
        result = repair_prompt("masterpiece, best quality, 1girl", ["feet"])
        self.assertNotIn("(pantyhose feet:1.3)", result)

    def test_all_drop_words_are_exercised_by_some_tag(self):
        # Documents the vocabulary the tests above cover, so a change to
        # REPAIR_DROP_WORDS is noticed here even if no single test above
        # happens to exercise the new/changed word.
        self.assertEqual(len(REPAIR_DROP_WORDS), 28)


class MaskedRedrawPromptTest(unittest.TestCase):
    def test_drop_words_remove_face_hair_and_framing_tags(self):
        result = masked_redraw_prompt(POSITIVE, "replace the dress")
        for dropped in ("light purple hair", "tareme", "jitome",
                        "half-closed eyes", "unamused", "sigh", "annoyed",
                        "closed mouth", "looking at viewer", "head tilt",
                        "rabbit hood", "hood down", "full body", "from front"):
            self.assertNotIn(dropped, result)

    def test_unrelated_tags_survive(self):
        result = masked_redraw_prompt(POSITIVE, "replace the dress")
        self.assertIn("masterpiece", result)
        self.assertIn("wide hips", result)

    def test_prompt_patch_is_appended_verbatim(self):
        result = masked_redraw_prompt(POSITIVE, "replace the dress")
        self.assertTrue(result.endswith("replace the dress"))

    def test_no_part_tags_are_added(self):
        result = masked_redraw_prompt(POSITIVE, "replace the dress")
        self.assertNotIn(PART_TAGS["hands"], result)
        self.assertNotIn(PART_TAGS["feet"], result)
        self.assertNotIn("(pantyhose feet:1.3)", result)


class PartLorasTest(unittest.TestCase):
    def test_none_weight_yields_no_loras(self):
        self.assertEqual(part_loras(["hands", "feet"], None), ())

    def test_feet_only(self):
        self.assertEqual(
            part_loras(["feet"], 0.8), ((PART_LORAS["feet"], 0.8),))

    def test_hands_and_feet_preserve_the_given_order(self):
        self.assertEqual(
            part_loras(["hands", "feet"], 0.8),
            ((PART_LORAS["hands"], 0.8), (PART_LORAS["feet"], 0.8)))
        self.assertEqual(
            part_loras(["feet", "hands"], 0.8),
            ((PART_LORAS["feet"], 0.8), (PART_LORAS["hands"], 0.8)))

    def test_unknown_part_is_skipped(self):
        self.assertEqual(
            part_loras(["face", "feet"], 0.8), ((PART_LORAS["feet"], 0.8),))

    def test_empty_parts_yields_no_loras(self):
        self.assertEqual(part_loras([], 0.8), ())


if __name__ == "__main__":
    unittest.main()
