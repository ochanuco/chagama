from __future__ import annotations

import io
import json
import os
import stat
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
from PIL import Image

from chagama.infrastructure.chimera.client import ChimeraClient
from chagama.infrastructure.comfyui.client import ComfyUIClient, as_png
from chagama.infrastructure.notifications.discord import DiscordNotifier
from chagama.infrastructure.persistence.run_state import JsonRunState


class AdapterTest(unittest.TestCase):
    def test_upload_source_is_png_with_the_same_pixels(self):
        rgba = np.random.default_rng(0).integers(0, 256, (16, 12, 4), dtype=np.uint8)
        webp = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(webp, "WEBP", lossless=True, exact=True)
        png = as_png(webp.getvalue())
        image = Image.open(io.BytesIO(png))
        self.assertEqual((image.format, image.mode), ("PNG", "RGBA"))
        self.assertTrue(np.array_equal(np.array(image), rgba))
        self.assertIs(as_png(png), png)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_chimera_cache_permissions_are_restricted(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            cache_directory = repository / ".local"
            cache_directory.mkdir(mode=0o755)
            client = ChimeraClient(repository)
            fields = [SimpleNamespace(stdout="client\n"),
                      SimpleNamespace(stdout="secret\n")]
            with patch("subprocess.run", side_effect=fields):
                client.credentials()
            self.assertEqual(stat.S_IMODE(cache_directory.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(client.token_cache.stat().st_mode), 0o600)

    def test_chimera_retry_does_not_sleep_after_final_attempt(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {}
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("offline")) as urlopen, \
                patch("time.sleep") as sleep, self.assertRaises(SystemExit):
            client.request("GET", "/test")
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(2), call(4)])

    def test_chimera_does_not_retry_a_non_idempotent_post(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {}
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("offline")) as urlopen, \
                patch("time.sleep") as sleep, self.assertRaises(SystemExit):
            client.request("POST", "/api/v1/requests", {"recipe": "yukari"})
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_chimera_retries_a_post_when_it_has_an_idempotency_key(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {}
        response = MagicMock()
        response.read.return_value = b'{"ok": true}'
        response.status = 201
        response.headers = {"Content-Type": "application/json"}
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen",
                   side_effect=[urllib.error.URLError("offline"), response]) as urlopen, \
                patch("time.sleep") as sleep:
            self.assertEqual(
                client.request("POST", "/api/v1/requests",
                               {"idempotency_key": "request-key"}),
                {"ok": True})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_chimera_request_returns_none_on_204(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {}
        response = MagicMock()
        response.read.return_value = b""
        response.status = 204
        response.headers = {}
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response):
            self.assertIsNone(client.request("POST", "/api/v1/requests/claim"))

    def test_chimera_request_returns_none_on_empty_body(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {}
        response = MagicMock()
        response.read.return_value = b""
        response.status = 200
        response.headers = {}
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response):
            self.assertIsNone(client.request("GET", "/api/v1/requests/claim"))

    def test_chimera_fetch_asset_reads_the_role_with_the_service_token(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {"CF-Access-Client-Id": "id"}
        response = MagicMock()
        response.read.return_value = b"asset-bytes"
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(client.fetch_asset("gen-1", "alpha"), b"asset-bytes")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url,
                         "https://example.invalid/g/gen-1/assets/alpha")
        self.assertEqual(request.get_header("Cf-access-client-id"), "id")

    def test_chimera_fetch_asset_is_none_for_a_missing_asset(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {}
        missing = urllib.error.HTTPError(
            "https://example.invalid/g/gen-1/assets/cut", 404, "Not Found", {}, None)
        with patch("urllib.request.urlopen", side_effect=missing):
            self.assertIsNone(client.fetch_asset("gen-1", "cut"))

    def test_chimera_fetch_asset_raises_on_other_errors(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {}
        failing = urllib.error.HTTPError(
            "https://example.invalid/g/gen-1/assets/cut", 403, "Forbidden", {}, None)
        with patch("urllib.request.urlopen", side_effect=failing), \
                self.assertRaises(urllib.error.HTTPError):
            client.fetch_asset("gen-1", "cut")

    def test_chimera_put_catalog_puts_to_the_recipe_refs_catalog_path(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {}
        response = MagicMock()
        response.read.return_value = json.dumps({"ok": True}).encode()
        response.status = 200
        response.headers = {"Content-Type": "application/json"}
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            result = client.put_catalog(
                "dev/catalog-publish", {"schema_version": 1})
        self.assertEqual(result, {"ok": True})
        request = urlopen.call_args[0][0]
        self.assertEqual(
            request.full_url,
            "https://example.invalid/api/v1/catalogs/dev/catalog-publish")
        self.assertEqual(request.get_method(), "PUT")
        self.assertEqual(json.loads(request.data), {"schema_version": 1})

    def test_chimera_put_catalog_escapes_the_recipe_ref(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {}
        response = MagicMock()
        response.read.return_value = b"{}"
        response.status = 200
        response.headers = {"Content-Type": "application/json"}
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            client.put_catalog("dev/catalog#v1", {"schema_version": 1})
        self.assertEqual(
            urlopen.call_args[0][0].full_url,
            "https://example.invalid/api/v1/catalogs/dev/catalog%23v1")

    def test_chimera_record_publication_posts_only_the_given_fields(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {}
        response = MagicMock()
        response.read.return_value = json.dumps({"id": "pub-1"}).encode()
        response.status = 201
        response.headers = {"Content-Type": "application/json"}
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            result = client.record_publication("gen-1", url="https://x.com/post/1")
        self.assertEqual(result, {"id": "pub-1"})
        request = urlopen.call_args[0][0]
        self.assertEqual(
            request.full_url,
            "https://example.invalid/api/v1/generations/gen-1/publications")
        self.assertEqual(request.get_method(), "POST")
        body = json.loads(request.data)
        self.assertEqual(body.pop("url"), "https://x.com/post/1")
        self.assertEqual(set(body), {"idempotency_key"})

    def test_chimera_record_publication_omits_unset_fields(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {}
        response = MagicMock()
        response.read.return_value = json.dumps({"id": "pub-2"}).encode()
        response.status = 201
        response.headers = {"Content-Type": "application/json"}
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            client.record_publication("gen-1", idempotency_key="pub-key")
        request = urlopen.call_args[0][0]
        self.assertEqual(json.loads(request.data), {"idempotency_key": "pub-key"})

    def test_chimera_record_publication_resends_one_generated_key(self):
        client = ChimeraClient(Path("."), base_url="https://example.invalid")
        client._credentials = {}
        response = MagicMock()
        response.read.return_value = json.dumps({"id": "pub-3"}).encode()
        response.status = 200
        response.headers = {"Content-Type": "application/json"}
        response.__enter__.return_value = response
        lost = urllib.error.URLError("connection reset")
        with patch("urllib.request.urlopen", side_effect=[lost, response]) as urlopen, \
                patch("time.sleep"):
            client.record_publication("gen-1")
        keys = [json.loads(sent[0][0].data)["idempotency_key"]
                for sent in urlopen.call_args_list]
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], keys[1])

    def test_comfyui_wait_retries_transport_error_and_returns_empty_success(self):
        client = ComfyUIClient(
            "http://example.invalid", poll_interval=0, poll_timeout=1)
        client.request = MagicMock(side_effect=[
            urllib.error.URLError("temporary"),
            {"prompt": {"status": {"status_str": "success"}, "outputs": {}}},
        ])
        self.assertEqual(client.wait_for("prompt"), [])

    def test_comfyui_knows_true_via_history(self):
        client = ComfyUIClient("http://example.invalid")
        client.request = MagicMock(return_value={"prompt": {"status": {}}})
        self.assertTrue(client.knows("prompt"))

    def test_comfyui_knows_true_via_queue_pending(self):
        client = ComfyUIClient("http://example.invalid")
        client.request = MagicMock(side_effect=[
            {},
            {"queue_running": [], "queue_pending": [[0, "prompt", {}, {}, []]]},
        ])
        self.assertTrue(client.knows("prompt"))

    def test_comfyui_knows_false_when_both_empty(self):
        client = ComfyUIClient("http://example.invalid")
        client.request = MagicMock(side_effect=[
            {},
            {"queue_running": [], "queue_pending": []},
        ])
        self.assertFalse(client.knows("prompt"))

    def test_comfyui_knows_true_on_url_error(self):
        client = ComfyUIClient("http://example.invalid")
        client.request = MagicMock(side_effect=urllib.error.URLError("offline"))
        self.assertTrue(client.knows("prompt"))

    def test_comfyui_upload_image_posts_multipart_and_returns_the_stored_name(self):
        client = ComfyUIClient("http://example.invalid")
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"name": "fin-source.png", "type": "input"}).encode()
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            name = client.upload_image("fin-source.png", b"\x89PNG\r\n\x1a\npng-bytes")
        self.assertEqual(name, "fin-source.png")
        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://example.invalid/upload/image")
        self.assertIn(b"png-bytes", request.data)
        self.assertIn(b'name="image"', request.data)
        self.assertIn(b'name="overwrite"', request.data)
        self.assertIn(b"true", request.data)
        self.assertTrue(
            request.headers["Content-type"].startswith("multipart/form-data"))

    def test_discord_closes_response_and_swallows_transport_errors(self):
        notifier = DiscordNotifier(Path("."))
        response = MagicMock()
        with patch.object(notifier, "_webhook", return_value="https://example"), \
                patch("urllib.request.urlopen", return_value=response):
            notifier.send("content", "image.png", b"image")
        response.__enter__.assert_called_once_with()
        response.__exit__.assert_called_once()

        output = io.StringIO()
        with patch.object(notifier, "_webhook", return_value="https://example"), \
                patch("urllib.request.urlopen",
                      side_effect=urllib.error.URLError("offline")), \
                redirect_stdout(output):
            notifier.send("content", "image.png", b"image")
        self.assertIn("Discord", output.getvalue())

    def test_run_state_save_is_atomic_and_cleans_up_on_failure(self):
        state = JsonRunState()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "state.json"
            state.save(path, {"value": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})
            self.assertEqual(list(root.iterdir()), [path])

            failed = root / "failed.json"
            with patch(
                    "chagama.infrastructure.persistence.run_state.os.replace",
                    side_effect=OSError("no replace")), self.assertRaises(OSError):
                state.save(failed, {"value": 2})
            self.assertEqual(sorted(item.name for item in root.iterdir()),
                             ["state.json"])


if __name__ == "__main__":
    unittest.main()
