"""Tests for the repair reroll's ControlNet vocabulary."""

from __future__ import annotations

import unittest

from chagama.domain.repair.controlnet import (
    CONTROL_MODELS,
    DEFAULT_CONTROL_STRENGTH,
    control_model,
)


class ControlModelTest(unittest.TestCase):
    def test_lineart_resolves_to_its_model_filename(self):
        self.assertEqual(control_model("lineart"), CONTROL_MODELS["lineart"])

    def test_unknown_word_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nope"):
            control_model("nope")

    def test_default_strength_is_within_bounds(self):
        self.assertGreater(DEFAULT_CONTROL_STRENGTH, 0)
        self.assertLessEqual(DEFAULT_CONTROL_STRENGTH, 2)


if __name__ == "__main__":
    unittest.main()
