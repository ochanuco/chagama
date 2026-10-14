"""WD tagger preprocessing, payload building and the upload hook."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from chagama.application import ingest
from chagama.application import safety as safety_application
from chagama.infrastructure.imaging import safety


class PreprocessTests(unittest.TestCase):
    def test_pads_to_centered_white_square_in_bgr(self):
        image = Image.new("RGB", (4, 2), (10, 20, 30))
        out = safety.preprocess(image, 4)
        self.assertEqual(out.shape, (1, 4, 4, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out[0, 0, 0].tolist(), [255, 255, 255])
        self.assertEqual(out[0, 1, 0].tolist(), [30, 20, 10])
        self.assertEqual(out[0, 2, 3].tolist(), [30, 20, 10])
        self.assertEqual(out[0, 3, 0].tolist(), [255, 255, 255])

    def test_resizes_to_model_size(self):
        out = safety.preprocess(Image.new("RGB", (8, 8), (0, 0, 0)), 4)
        self.assertEqual(out.shape, (1, 4, 4, 3))

    def test_alpha_composites_onto_white(self):
        image = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
        image.putpixel((0, 0), (255, 0, 0, 255))
        out = safety.preprocess(image, 2)
        self.assertEqual(out[0, 0, 0].tolist(), [0, 0, 255])
        self.assertEqual(out[0, 1, 1].tolist(), [255, 255, 255])

    def test_palette_image_is_accepted(self):
        out = safety.preprocess(Image.new("P", (2, 2)), 2)
        self.assertEqual(out.shape, (1, 2, 2, 3))


class PayloadTests(unittest.TestCase):
    def test_rating_general_cutoff_rounding_and_no_characters(self):
        table = [("general", "9"), ("sensitive", "9"), ("questionable", "9"),
                 ("explicit", "9"), ("1girl", "0"), ("smile", "0"),
                 ("thighhighs", "0"), ("yuzuki_yukari", "4")]
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.98765, 0.04999, 0.05, 0.9]
        payload = safety.build_payload(probabilities, table)
        self.assertEqual(payload["model"], safety.MODEL_ID)
        self.assertEqual(list(payload["rating"]), list(safety.RATING_KEYS))
        self.assertAlmostEqual(payload["rating"]["explicit"], 0.4)
        self.assertEqual(payload["tags"], {"1girl": 0.988, "thighhighs": 0.05})


class UploadHookTests(unittest.TestCase):
    def test_rating_failure_only_emits(self):
        calls, messages = [], []

        class Management:
            def request(self, method, path, payload=None, multipart=None):
                calls.append((method, path))
                return {"id": "gen-1", "canonical_url": "https://example/g"}

        buffer = io.BytesIO()
        Image.new("RGB", (2, 2)).save(buffer, "PNG")
        rendered = ingest.upload_generation(
            Management(), messages.append, "job-1", seed=1, name="a.png",
            data=buffer.getvalue(), index=0)
        self.assertEqual(rendered["id"], "gen-1")
        self.assertTrue(any("safety rating skipped" in m for m in messages))
        self.assertFalse(any(path.endswith("/safety") for _, path in calls))

    def test_put_failure_only_emits(self):
        messages = []

        class Management:
            def request(self, method, path, payload=None, multipart=None):
                if path.endswith("/safety"):
                    raise SystemExit("HTTP 404")
                return {"id": "gen-1", "canonical_url": "https://example/g"}

        original = ingest.rate_image
        ingest.rate_image = lambda data: {"rating": {}}
        try:
            ingest.upload_generation(
                Management(), messages.append, "job-1", seed=1, name="a.png",
                data=b"x", index=0)
        finally:
            ingest.rate_image = original
        self.assertTrue(any("HTTP 404" in m for m in messages))



class RateByIdTests(unittest.TestCase):
    def rate(self, generation):
        fetched, puts = [], []

        class Chimera:
            def request(self, method, path, payload=None):
                return generation

            def fetch_generation_image(self, identifier):
                fetched.append("image")
                return b"original"

            def fetch_generation_preview(self, identifier):
                fetched.append("preview")
                return b"preview"

            def put_safety(self, generation_id, payload):
                puts.append(generation_id)

        original = safety_application.rate_image
        safety_application.rate_image = lambda data: {"rating": {"source": data}}
        try:
            rating = safety_application.rate_generation_by_id(Chimera(), "abc123")
        finally:
            safety_application.rate_image = original
        return rating, fetched, puts

    def test_rates_the_original(self):
        rating, fetched, puts = self.rate({"id": "gen-1", "original_purged_at": None})
        self.assertEqual(fetched, ["image"])
        self.assertEqual(rating, {"source": b"original"})
        self.assertEqual(puts, ["gen-1"])

    def test_rates_the_preview_when_the_original_was_purged(self):
        rating, fetched, puts = self.rate(
            {"id": "gen-1", "original_purged_at": "2026-09-01T00:00:00Z"})
        self.assertEqual(fetched, ["preview"])
        self.assertEqual(rating, {"source": b"preview"})
        self.assertEqual(puts, ["gen-1"])

if __name__ == "__main__":
    unittest.main()
