"""Tests for the repair reroll's checkpoint vocabulary."""

from __future__ import annotations

import unittest

from chagama.domain.repair.models import MODELS, resolve_model


class ResolveModelTest(unittest.TestCase):
    def test_anima_resolves_to_the_sudachi_checkpoint(self):
        self.assertEqual(resolve_model("anima"), "sudachiAnima_v10.safetensors")

    def test_every_vocabulary_word_resolves_to_a_safetensors_file(self):
        for word, checkpoint in MODELS.items():
            self.assertEqual(resolve_model(word), checkpoint)
            self.assertTrue(checkpoint.endswith(".safetensors"))

    def test_unknown_word_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nope"):
            resolve_model("nope")


if __name__ == "__main__":
    unittest.main()
