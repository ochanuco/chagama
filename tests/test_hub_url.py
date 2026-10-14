"""hub_url's scheme swap -- the only pure-function piece of the hub adapter."""

from __future__ import annotations

import unittest

from chagama.infrastructure.chimera.hub import encode, hub_url


class HubUrlTest(unittest.TestCase):
    def test_https_becomes_wss(self):
        self.assertEqual(
            hub_url("https://chimera.chanu.co"),
            "wss://chimera.chanu.co/api/v1/worker/ws")

    def test_http_becomes_ws(self):
        self.assertEqual(
            hub_url("http://127.0.0.1:8787"),
            "ws://127.0.0.1:8787/api/v1/worker/ws")

    def test_trailing_slash_is_stripped(self):
        self.assertEqual(
            hub_url("https://chimera.chanu.co/"),
            "wss://chimera.chanu.co/api/v1/worker/ws")

    def test_unsupported_scheme_raises(self):
        with self.assertRaises(ValueError):
            hub_url("ftp://example.invalid")


class EncodeTest(unittest.TestCase):
    def test_ping_matches_the_hub_auto_response_pattern(self):
        self.assertEqual(encode({"type": "ping"}), '{"type":"ping"}')


if __name__ == "__main__":
    unittest.main()
