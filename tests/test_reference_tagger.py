from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from chagama.infrastructure.reference import anima_tagger as tagger
from chagama.infrastructure.reference import models
from chagama.infrastructure.reference.anima_tagger import TagGroup


def logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


class DecideTest(unittest.TestCase):
    names = ["1girl", "solo", "smile", "full body", "cowboy shot",
             "<none:framing>", "upper body", "2girls"]

    def run_decide(self, probs: dict[str, float], groups,
                   thresholds: float = 0.5) -> dict[str, float]:
        logits = np.array([logit(probs.get(name, 1e-4)) for name in self.names])
        return tagger.decide(logits, self.names,
                             np.full(len(self.names), thresholds), groups)

    def test_tags_over_their_threshold_are_kept(self):
        kept = self.run_decide({"1girl": 0.9, "smile": 0.6, "solo": 0.4}, ())
        self.assertEqual(set(kept), {"1girl", "smile"})
        self.assertAlmostEqual(kept["smile"], 0.6)

    def test_a_softmax_when_solo_group_keeps_only_its_argmax(self):
        framing = TagGroup("framing", "softmax_when_solo",
                           ("full body", "cowboy shot", "upper body"),
                           sentinel=True)
        kept = self.run_decide(
            {"1girl": 0.9, "solo": 0.9, "full body": 0.7, "cowboy shot": 0.8},
            (framing,))
        self.assertIn("cowboy shot", kept)
        self.assertNotIn("full body", kept)

    def test_the_sentinel_winning_empties_the_group(self):
        framing = TagGroup("framing", "softmax_when_solo",
                           ("full body", "cowboy shot"), sentinel=True)
        kept = self.run_decide(
            {"1girl": 0.9, "solo": 0.9, "full body": 0.6,
             "<none:framing>": 0.95}, (framing,))
        self.assertNotIn("full body", kept)
        self.assertNotIn("<none:framing>", kept)

    def test_a_winner_under_its_threshold_is_not_kept(self):
        framing = TagGroup("framing", "softmax_when_solo",
                           ("full body", "cowboy shot"))
        kept = self.run_decide(
            {"1girl": 0.9, "solo": 0.9, "full body": 0.3}, (framing,))
        self.assertNotIn("full body", kept)

    def test_a_solo_group_is_left_to_thresholds_with_two_subjects(self):
        framing = TagGroup("framing", "softmax_when_solo",
                           ("full body", "cowboy shot"))
        kept = self.run_decide(
            {"2girls": 0.9, "full body": 0.7, "cowboy shot": 0.8}, (framing,))
        self.assertTrue({"full body", "cowboy shot"} <= set(kept))

    def test_an_escape_tag_leaves_the_group_to_thresholds(self):
        group = TagGroup("g", "softmax", ("full body", "cowboy shot"),
                         escape=("smile",))
        kept = self.run_decide(
            {"smile": 0.9, "full body": 0.7, "cowboy shot": 0.8}, (group,))
        self.assertTrue({"full body", "cowboy shot"} <= set(kept))

    def test_multilabel_groups_change_nothing(self):
        group = TagGroup("g", "multilabel", ("full body", "cowboy shot"))
        kept = self.run_decide(
            {"1girl": 0.9, "full body": 0.7, "cowboy shot": 0.8}, (group,))
        self.assertTrue({"full body", "cowboy shot"} <= set(kept))


class CheckpointPartsTest(unittest.TestCase):
    def test_parse_groups_skips_the_version_key(self):
        groups = tagger.parse_groups({
            "version": 1,
            "framing": {"mode": "softmax_when_solo", "tags": ["full body"],
                        "sentinel": True},
            "pose": {"mode": "multilabel", "tags": ["sitting"],
                     "escape": ["x"]},
        })
        self.assertEqual(groups, (
            TagGroup("framing", "softmax_when_solo", ("full body",), (), True),
            TagGroup("pose", "multilabel", ("sitting",), ("x",), False)))

    def test_align_falls_back_to_the_renamed_source(self):
        ours, ext = tagger.align(
            ["smile", "nsfw", "missing"], {"smile": 4, "questionable": 9},
            tagger.rename_recovery({"questionable": "nsfw"}))
        self.assertEqual((ours, ext), ([0, 1], [4, 9]))

    def test_is_solo(self):
        self.assertTrue(tagger.is_solo({"1girl", "solo", "smile"}))
        self.assertFalse(tagger.is_solo({"1girl", "2girls"}))
        self.assertFalse(tagger.is_solo({"smile"}))

    def test_unsupported_tags_sit_far_below_any_threshold(self):
        logits = tagger.to_logits(np.array([0.0, 0.7]), np.array([False, True]))
        self.assertEqual(logits[0], tagger.UNSUPPORTED_LOGIT)
        self.assertAlmostEqual(logits[1], logit(0.7))

    def test_preprocess_pads_white_to_a_square(self):
        image = Image.new("RGB", (40, 20), (0, 0, 0))
        x = tagger.preprocess(image, 8)
        self.assertEqual(x.shape, (3, 8, 8))
        self.assertAlmostEqual(float(x[:, 0, 0].mean()), 1.0)
        self.assertAlmostEqual(float(x[:, 4, 4].mean()), 0.0)


class ModelPathTest(unittest.TestCase):
    def test_missing_files_are_named(self):
        with self.assertRaisesRegex(models.ReferenceUnavailable, "config.json"):
            with patch.dict("os.environ", {models.MODELS_ENV: "/nonexistent"}):
                models.model_path(models.TAGGER_HEAD, ("config.json",))


if __name__ == "__main__":
    unittest.main()
