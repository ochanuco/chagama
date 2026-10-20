from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock
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

    def __init__(self, files: dict[str, bytes], *, ranges: bool = True, lfs_from: int = 0):
        self.files = files
        self.lfs_from = lfs_from
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
            if len(body) < self.lfs_from:
                oid = hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()
                return _Response(json.dumps([{
                    "path": path, "size": len(body), "oid": oid}]).encode())
            return _Response(json.dumps([{
                "path": path, "size": len(body), "oid": "f" * 40,
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


class FakeCivitai:
    """Civitai's model-versions and download endpoints over one file body."""

    def __init__(self, body: bytes, *, model_id: int = 7, version: int = 70):
        self.body = body
        self.model_id = model_id
        self.version = version
        self.downloads: list[tuple[str, str | None]] = []

    def __call__(self, request):
        url = request.full_url
        if f"/api/v1/model-versions/{self.version}" in url:
            return _Response(json.dumps({
                "modelId": self.model_id, "name": "v1",
                "files": [
                    {"primary": False, "type": "VAE", "sizeKB": 1.0,
                     "hashes": {"SHA256": "AB" * 32}},
                    {"primary": True, "type": "Model",
                     "sizeKB": len(self.body) / 1024,
                     "hashes": {"SHA256": hashlib.sha256(self.body).hexdigest().upper()}},
                ]}).encode())
        assert f"/api/download/models/{self.version}?token=" in url
        rng = request.get_header("Range")
        self.downloads.append((url, rng))
        if rng:
            return _Response(self.body[int(rng.removeprefix("bytes=").rstrip("-")):],
                             status=206)
        return _Response(self.body)


def _civitai(name: str = "m.safetensors", **kw) -> Model:
    return Model(dest=f"diffusion_models/{name}", civitai_model=7, civitai_version=70, **kw)


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
            if not model.is_civitai:
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

    def manifest(self, body: str) -> Path:
        path = Path(self.tmp.name) / "models.toml"
        path.write_text(body)
        return path

    def test_a_civitai_entry_loads(self):
        path = self.manifest('[[model]]\ndest = "diffusion_models/m.safetensors"\n'
                             "civitai_model = 7\ncivitai_version = 70 # v1\n")
        self.assertEqual(worker_models.load_manifest(path), [_civitai()])

    def test_mixed_or_missing_keys_are_rejected(self):
        head = '[[model]]\ndest = "vae/a.bin"\n'
        for body in ('repo = "o/r"\nfile = "x"\ncivitai_model = 7\ncivitai_version = 70\n',
                     'repo = "o/r"\ncivitai_version = 70\n',
                     "civitai_model = 7\n",
                     "civitai_version = 70\n",
                     'repo = "o/r"\n',
                     ""):
            with self.subTest(body=body), self.assertRaises(FetchError):
                worker_models.load_manifest(self.manifest(head + body))

    def test_fetches_and_verifies_a_civitai_file(self):
        body = b"silver" * 2000
        hub = FakeCivitai(body)
        with mock.patch.dict(os.environ, {"CIVITAI_TOKEN": "sekret"}):
            self.assertTrue(self.sync([_civitai()], hub))
        target = self.models_dir / "diffusion_models/m.safetensors"
        self.assertEqual(target.read_bytes(), body)
        self.assertEqual(Path(str(target) + ".sha256").read_text().strip(),
                         hashlib.sha256(body).hexdigest())
        self.assertIn("civitai 7/70", self.lines)
        self.assertFalse(any("sekret" in line for line in self.lines))

    def test_a_civitai_download_resumes(self):
        body = bytes(range(256)) * 100
        part = self.models_dir / "diffusion_models/m.safetensors.part"
        part.parent.mkdir(parents=True)
        part.write_bytes(body[:1000])
        hub = FakeCivitai(body)
        with mock.patch.dict(os.environ, {"CIVITAI_TOKEN": "t"}):
            self.sync([_civitai()], hub)
        self.assertEqual([r for _, r in hub.downloads], ["bytes=1000-"])
        self.assertEqual((self.models_dir / "diffusion_models/m.safetensors").read_bytes(), body)

    def test_a_missing_token_fails_before_any_download(self):
        hub = FakeCivitai(b"abc")
        with mock.patch.dict(os.environ, clear=True), \
                self.assertRaisesRegex(FetchError, "CIVITAI_TOKEN"):
            self.sync([_civitai()], hub)
        self.assertEqual(hub.downloads, [])

    def test_a_dry_run_needs_no_token(self):
        hub = FakeCivitai(b"abc")
        with mock.patch.dict(os.environ, clear=True):
            self.assertFalse(self.sync([_civitai()], hub, dry_run=True))
        self.assertEqual(hub.downloads, [])
        self.assertIn("diffusion_models/m.safetensors would fetch 0.00 GB", self.lines)

    def test_a_model_id_mismatch_is_rejected(self):
        hub = FakeCivitai(b"abc", model_id=8)
        with self.assertRaisesRegex(FetchError, "belongs to model 8"):
            self.sync([_civitai()], hub)

    def test_a_pinned_hash_must_match_civitai(self):
        with self.assertRaisesRegex(FetchError, "pinned"):
            self.sync([_civitai(sha256="0" * 64)], FakeCivitai(b"abc"))

    def test_same_size_new_content_overwrites_a_fixed_dest(self):
        old, new = b"a" * 5000, b"b" * 5000
        with mock.patch.dict(os.environ, {"CIVITAI_TOKEN": "t"}):
            self.sync([_civitai()], FakeCivitai(old))
            hub = FakeCivitai(new)
            self.assertTrue(self.sync([_civitai()], hub))
        target = self.models_dir / "diffusion_models/m.safetensors"
        self.assertEqual(target.read_bytes(), new)
        self.assertEqual(len(hub.downloads), 1)
        self.assertEqual(Path(str(target) + ".sha256").read_text().strip(),
                         hashlib.sha256(new).hexdigest())

    def test_a_file_without_a_sidecar_is_hashed_and_kept(self):
        body = b"a" * 5000
        target = self.models_dir / "diffusion_models/m.safetensors"
        target.parent.mkdir(parents=True)
        target.write_bytes(body)
        hub = FakeCivitai(body)
        with mock.patch.dict(os.environ, {"CIVITAI_TOKEN": "t"}):
            self.assertFalse(self.sync([_civitai()], hub))
        self.assertEqual(hub.downloads, [])
        self.assertEqual(Path(str(target) + ".sha256").read_text().strip(),
                         hashlib.sha256(body).hexdigest())

    def test_a_sidecarless_file_with_other_content_is_refetched(self):
        target = self.models_dir / "diffusion_models/m.safetensors"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * 5000)
        hub = FakeCivitai(b"y" * 5000)
        with mock.patch.dict(os.environ, {"CIVITAI_TOKEN": "t"}):
            self.assertTrue(self.sync([_civitai()], hub))
        self.assertEqual(target.read_bytes(), b"y" * 5000)

    def test_a_file_outside_lfs_is_checked_by_its_git_blob_sha1(self):
        body = b'{"a": 1}\n'
        hub = FakeHub({"vae/a.json": body}, lfs_from=10**6)
        self.assertTrue(self.sync([_model("a.json")], hub))
        target = self.models_dir / "vae/a.json"
        self.assertEqual(target.read_bytes(), body)
        blob = hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()
        self.assertEqual(Path(str(target) + ".sha256").read_text().strip(), blob)
        hub.downloads.clear()
        self.assertFalse(self.sync([_model("a.json")], hub))
        self.assertEqual(hub.downloads, [])

    def test_a_file_outside_lfs_with_other_content_is_refetched(self):
        target = self.models_dir / "vae/a.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * 10)
        hub = FakeHub({"vae/a.json": b"y" * 10}, lfs_from=10**6)
        self.assertTrue(self.sync([_model("a.json")], hub))
        self.assertEqual(target.read_bytes(), b"y" * 10)

    def test_a_corrupt_download_outside_lfs_is_removed(self):
        class Tampered(FakeHub):
            def __call__(self, request):
                if "/resolve/" in request.full_url:
                    return _Response(b"z" * 10)
                return super().__call__(request)

        hub = Tampered({"vae/a.json": b"y" * 10}, lfs_from=10**6)
        with self.assertRaises(FetchError):
            self.sync([_model("a.json")], hub)
        self.assertFalse((self.models_dir / "vae/a.json").exists())
        self.assertFalse((self.models_dir / "vae/a.json.part").exists())

    def test_the_manifest_pins_every_hugging_face_revision_to_a_commit(self):
        models = worker_models.load_manifest(REPO_ROOT / "manifests" / "worker-models.toml")
        for model in models:
            if not model.is_civitai:
                self.assertRegex(model.revision, r"^[0-9a-f]{40}$", model.dest)

    def test_a_hugging_face_file_gets_a_sidecar_too(self):
        hub = FakeHub({"vae/a.bin": b"alpha"})
        self.sync([_model("a.bin")], hub)
        self.assertTrue((self.models_dir / "vae/a.bin.sha256").exists())


if __name__ == "__main__":
    unittest.main()
