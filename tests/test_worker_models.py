from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import worker_models
from worker_models import FetchError, Model

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMIT = "c" * 40


class _Response(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status


class FakeHub:
    """Hugging Face's revision, paths-info and resolve endpoints over a dict."""

    def __init__(self, files: dict[str, bytes], *, ranges: bool = True):
        self.files = files
        self.ranges = ranges
        self.downloads: list[tuple[str, str | None]] = []

    def __call__(self, request):
        url = request.full_url
        if "/revision/" in url:
            return _Response(json.dumps({"sha": COMMIT}).encode())
        if "/paths-info/" in url:
            assert COMMIT in url
            path = json.loads(request.data)["paths"][0]
            if path not in self.files:
                return _Response(b"[]")
            body = self.files[path]
            return _Response(json.dumps([{
                "path": path, "size": len(body),
                "lfs": {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)},
            }]).encode())
        assert f"/resolve/{COMMIT}/" in url
        path = url.split(f"/resolve/{COMMIT}/", 1)[1]
        body = self.files[path]
        rng = request.get_header("Range")
        self.downloads.append((path, rng))
        if rng and self.ranges:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            return _Response(body[start:], status=206)
        return _Response(body)


def _model(name: str, **kw) -> Model:
    return Model(dest=f"vae/{name}", repo="org/repo", file=f"vae/{name}",
                 revision="main", **kw)


class WorkerModelsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name) / "models"
        self.lines: list[str] = []

    def tearDown(self):
        self.tmp.cleanup()

    def sync(self, models, hub, **kw):
        return worker_models.sync(models, self.models_dir, opener=hub,
                                  disk_free=lambda _: 10**12,
                                  echo=self.lines.append, **kw)

    def test_the_manifest_loads(self):
        models = worker_models.load_manifest(REPO_ROOT / "manifests" / "worker-models.toml")
        self.assertTrue(models)
        for model in models:
            self.assertEqual(model.dest.split("/")[-1], model.file.split("/")[-1])

    def test_dest_must_stay_under_models(self):
        manifest = Path(self.tmp.name) / "models.toml"
        for dest in ("../evil.bin", "/abs/x.bin", "flat.bin"):
            manifest.write_text(f'[[model]]\ndest = "{dest}"\nrepo = "o/r"\nfile = "x"\n')
            with self.subTest(dest=dest), self.assertRaises(FetchError):
                worker_models.load_manifest(manifest)

    def test_fetches_and_verifies(self):
        hub = FakeHub({"vae/a.bin": b"alpha" * 1000})
        self.assertTrue(self.sync([_model("a.bin")], hub))
        self.assertEqual((self.models_dir / "vae/a.bin").read_bytes(), b"alpha" * 1000)
        self.assertFalse((self.models_dir / "vae/a.bin.part").exists())
        self.assertEqual(self.lines[-1], "changed: True")

    def test_a_present_file_is_left_alone(self):
        hub = FakeHub({"vae/a.bin": b"alpha"})
        self.sync([_model("a.bin")], hub)
        hub.downloads.clear()
        self.assertFalse(self.sync([_model("a.bin")], hub))
        self.assertEqual(hub.downloads, [])

    def test_resumes_a_partial_download(self):
        body = bytes(range(256)) * 100
        part = self.models_dir / "vae/a.bin.part"
        part.parent.mkdir(parents=True)
        part.write_bytes(body[:1000])
        hub = FakeHub({"vae/a.bin": body})
        self.sync([_model("a.bin")], hub)
        self.assertEqual(hub.downloads, [("vae/a.bin", "bytes=1000-")])
        self.assertEqual((self.models_dir / "vae/a.bin").read_bytes(), body)

    def test_a_server_ignoring_range_restarts_the_file(self):
        body = b"x" * 5000
        part = self.models_dir / "vae/a.bin.part"
        part.parent.mkdir(parents=True)
        part.write_bytes(body[:100])
        self.sync([_model("a.bin")], FakeHub({"vae/a.bin": body}, ranges=False))
        self.assertEqual((self.models_dir / "vae/a.bin").read_bytes(), body)

    def test_a_corrupt_partial_is_removed(self):
        body = b"good" * 1000
        part = self.models_dir / "vae/a.bin.part"
        part.parent.mkdir(parents=True)
        part.write_bytes(b"BAD!")
        with self.assertRaises(FetchError):
            self.sync([_model("a.bin")], FakeHub({"vae/a.bin": body}))
        self.assertFalse(part.exists())
        self.assertFalse((self.models_dir / "vae/a.bin").exists())

    def test_missing_upstream_fails_before_any_download(self):
        hub = FakeHub({"vae/a.bin": b"alpha"})
        with self.assertRaisesRegex(FetchError, "vae/b.bin"):
            self.sync([_model("a.bin"), _model("b.bin")], hub)
        self.assertEqual(hub.downloads, [])

    def test_a_pinned_hash_must_match_upstream(self):
        hub = FakeHub({"vae/a.bin": b"alpha"})
        with self.assertRaisesRegex(FetchError, "pinned"):
            self.sync([_model("a.bin", sha256="0" * 64)], hub)

    def test_dry_run_downloads_nothing(self):
        hub = FakeHub({"vae/a.bin": b"alpha"})
        self.assertFalse(self.sync([_model("a.bin")], hub, dry_run=True))
        self.assertEqual(hub.downloads, [])
        self.assertIn("vae/a.bin would fetch 0.00 GB", self.lines)

    def test_not_enough_disk_fails_before_any_download(self):
        hub = FakeHub({"vae/a.bin": b"alpha"})
        with self.assertRaisesRegex(FetchError, "free"):
            worker_models.sync([_model("a.bin")], self.models_dir, opener=hub,
                               disk_free=lambda _: 0, echo=self.lines.append)
        self.assertEqual(hub.downloads, [])


if __name__ == "__main__":
    unittest.main()
