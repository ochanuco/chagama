"""Unit tests for the prompt fingerprint helper."""

from __future__ import annotations

import unittest

from chagama.domain.generation.fingerprint import prompt_fingerprint


class PromptFingerprintTest(unittest.TestCase):
    def test_stable_prefix(self):
        digest = prompt_fingerprint("yukari", "lounge", "positive", "negative")
        self.assertTrue(digest.startswith("sha256:"))

    def test_changes_with_positive_prompt(self):
        first = prompt_fingerprint("yukari", "lounge", "positive", "negative")
        second = prompt_fingerprint("yukari", "lounge", "different", "negative")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
