from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from chagama.envcompat import LEGACY, getenv


class GetenvTest(unittest.TestCase):
    def test_new_name_is_read(self):
        with patch.dict(os.environ, {"CHAGAMA_MODEL_DIR": "/new"}, clear=True):
            self.assertEqual(getenv("CHAGAMA_MODEL_DIR"), "/new")

    def test_legacy_name_is_the_fallback(self):
        with patch.dict(os.environ, {"COMFY_RECIPES_MODEL_DIR": "/old"}, clear=True):
            self.assertEqual(getenv("CHAGAMA_MODEL_DIR"), "/old")

    def test_new_name_wins(self):
        env = {"CHAGAMA_MODEL_DIR": "/new", "COMFY_RECIPES_MODEL_DIR": "/old"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(getenv("CHAGAMA_MODEL_DIR"), "/new")

    def test_unset_is_none(self):
        with patch.dict(os.environ, {}, clear=True):
            for name in LEGACY:
                self.assertIsNone(getenv(name))
