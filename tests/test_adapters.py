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

from chagama.domain.yukari.recipe import render_spec
from chagama.infrastructure.chimera.client import ChimeraClient
from chagama.infrastructure.comfyui import anima_graph
from chagama.infrastructure.comfyui.client import ComfyUIClient, as_png
from chagama.infrastructure.comfyui.refinement_graph import redraw_graph
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

    def test_redraw_graph_rejects_a_base_with_no_vaedecode_and_non_numeric_ids(self):
        base = {
            "3": {"class_type": "KSampler",
                  "inputs": {"positive": ["6", 0], "negative": ["7", 0]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
        }
        with self.assertRaises(ValueError):
            redraw_graph(base, 2048, 0.2, "test", canvas=(832, 1664))
        with self.assertRaisesRegex(ValueError, "non-numeric node IDs"):
            redraw_graph({**base, "output": {}}, 2048, 0.2, "test", canvas=(832, 1664))

    def test_redraw_graph_upscales_the_decoded_image_in_pixel_space(self):
        base = {
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 7, "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "DiffusersLoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }
        graph = redraw_graph(base, 2048, 0.45, "fin", canvas=(832, 1664))
        scale = graph["10"]
        self.assertEqual(scale["class_type"], "ImageScale")
        self.assertEqual(scale["inputs"]["upscale_method"], "bicubic")
        self.assertEqual(scale["inputs"]["image"], ["8", 0])
        self.assertEqual(
            (scale["inputs"]["width"], scale["inputs"]["height"]), (1024, 2048))
        encode = graph["11"]
        self.assertEqual(encode["class_type"], "VAEEncode")
        self.assertEqual(encode["inputs"]["pixels"], ["10", 0])
        self.assertEqual(graph["12"]["inputs"]["latent_image"], ["11", 0])
        self.assertEqual(graph["9"]["inputs"]["images"], ["13", 0])

    def test_redraw_graph_redraw_from_source_loads_the_uploaded_picture(self):
        base = {
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 7, "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "DiffusersLoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }
        graph = redraw_graph(base, 2048, 0.45, "fin", canvas=(832, 1664),
                           source_image="mrd-source.png", redraw_from_source=True)
        load = graph["23"]
        self.assertEqual(load, {"class_type": "LoadImage",
                                "inputs": {"image": "mrd-source.png"}})
        scale = graph["10"]
        self.assertEqual(scale["inputs"]["image"], ["23", 0])

    def test_redraw_graph_without_redraw_from_source_uses_the_base_saveimage(self):
        base = {
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 7, "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "DiffusersLoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }
        graph = redraw_graph(base, 2048, 0.45, "fin", canvas=(832, 1664),
                           source_image="mrd-source.png")
        scale = graph["10"]
        self.assertEqual(scale["inputs"]["image"], ["8", 0])
        self.assertNotIn("23", graph)

    def test_redraw_graph_redraw_from_source_requires_source_image(self):
        base = {
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 7, "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "DiffusersLoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }
        with self.assertRaisesRegex(ValueError, "requires source_image"):
            redraw_graph(base, 2048, 0.45, "fin", canvas=(832, 1664), redraw_from_source=True)

    def test_redraw_graph_pixel_route_honours_the_upscale_method(self):
        base = {
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 7, "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "DiffusersLoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }
        graph = redraw_graph(base, 2048, 0.45, "fin", canvas=(832, 1664), upscale="nearest-exact")
        scale = graph["10"]
        self.assertEqual(scale["class_type"], "ImageScale")
        self.assertEqual(scale["inputs"]["upscale_method"], "nearest-exact")

    def test_redraw_graph_rejects_an_unknown_upscale_method(self):
        base = {
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 7, "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "DiffusersLoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }
        with self.assertRaisesRegex(ValueError, "unsupported upscale method"):
            redraw_graph(base, 2048, 0.45, "fin", canvas=(832, 1664), upscale="mitchell")

    def test_redraw_graph_sampler_override_keeps_steps_cfg_and_seed(self):
        base = {
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 7, "steps": 30, "cfg": 5.0,
                             "sampler_name": "dpmpp_2m", "scheduler": "karras",
                             "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "DiffusersLoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }
        graph = redraw_graph(base, 2048, 0.45, "fin", canvas=(832, 1664), sampler=("euler", "normal"))
        sample = graph["12"]
        self.assertEqual(sample["inputs"]["sampler_name"], "euler")
        self.assertEqual(sample["inputs"]["scheduler"], "normal")
        self.assertEqual(sample["inputs"]["steps"], 30)
        self.assertEqual(sample["inputs"]["cfg"], 5.0)
        self.assertEqual(sample["inputs"]["seed"], 7)

    def test_redraw_graph_loader_adds_a_diffusers_loader_and_reroutes(self):
        base = {
            "3": {"class_type": "KSampler",
                  "inputs": {"model": ["1", 0], "seed": 7, "steps": 25,
                             "cfg": 3.5, "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "VAELoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"clip": ["2", 0], "text": "p"}},
            "7": {"class_type": "CLIPTextEncode",
                  "inputs": {"clip": ["2", 0], "text": "n"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 0]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }
        original = json.loads(json.dumps(base))
        graph = redraw_graph(base, 2048, 0.75, "fin", loader="hassaku-il-v22",
                           canvas=(832, 1664))
        loader_id = "20"
        self.assertEqual(graph[loader_id],
                         {"class_type": "DiffusersLoader",
                          "inputs": {"model_path": "hassaku-il-v22"}})
        self.assertEqual(graph["12"]["inputs"]["model"], [loader_id, 0])
        self.assertEqual(graph["14"]["inputs"]["clip"], [loader_id, 1])
        self.assertEqual(graph["15"]["inputs"]["clip"], [loader_id, 1])
        self.assertEqual(graph["11"]["inputs"]["vae"], [loader_id, 2])
        self.assertEqual(graph["13"]["inputs"]["vae"], [loader_id, 2])
        for key in ("3", "4", "5", "6", "7"):
            self.assertEqual(graph[key], original[key])

    def test_redraw_graph_loads_a_single_file_loader_as_a_checkpoint(self):
        base = {
            "3": {"class_type": "KSampler",
                  "inputs": {"model": ["1", 0], "seed": 7, "steps": 25,
                             "cfg": 3.5, "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "VAELoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"clip": ["2", 0], "text": "p"}},
            "7": {"class_type": "CLIPTextEncode",
                  "inputs": {"clip": ["2", 0], "text": "n"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 0]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }
        original = json.loads(json.dumps(base))
        graph = redraw_graph(base, 2048, 0.75, "fin", loader="animagine-xl-4.0-opt.safetensors",
                           canvas=(832, 1664))
        loader_id = "20"
        self.assertEqual(graph[loader_id],
                         {"class_type": "CheckpointLoaderSimple",
                          "inputs": {"ckpt_name": "animagine-xl-4.0-opt.safetensors"}})
        self.assertEqual(graph["12"]["inputs"]["model"], [loader_id, 0])
        self.assertEqual(graph["14"]["inputs"]["clip"], [loader_id, 1])
        self.assertEqual(graph["15"]["inputs"]["clip"], [loader_id, 1])
        self.assertEqual(graph["11"]["inputs"]["vae"], [loader_id, 2])
        self.assertEqual(graph["13"]["inputs"]["vae"], [loader_id, 2])
        for key in ("3", "4", "5", "6", "7"):
            self.assertEqual(graph[key], original[key])

    def test_redraw_graph_sampling_override_sets_steps_and_cfg(self):
        base = {
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 7, "steps": 25, "cfg": 3.5,
                             "sampler_name": "er_sde", "scheduler": "normal",
                             "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "DiffusersLoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }
        original_node_3 = json.loads(json.dumps(base["3"]))
        graph = redraw_graph(base, 2048, 0.75, "fin",
                           sampler=("dpmpp_2m", "karras"),
                           sampling=(30, 5.0), canvas=(832, 1664))
        sample = graph["12"]
        self.assertEqual(sample["inputs"]["steps"], 30)
        self.assertEqual(sample["inputs"]["cfg"], 5.0)
        self.assertEqual(graph["3"], original_node_3)

    def _plain_base(self):
        return {
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 7, "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "DiffusersLoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }

    def test_redraw_graph_latent_route_with_source_image_encodes_it_then_upscales(self):
        graph = redraw_graph(self._plain_base(), 2048, 0.55, "fin", canvas=(832, 1664),
                           latent_route=True, source_image="repaired.png")
        load = graph["23"]
        self.assertEqual(load, {"class_type": "LoadImage",
                                "inputs": {"image": "repaired.png"}})
        encode = graph["11"]
        self.assertEqual(encode["class_type"], "VAEEncode")
        self.assertEqual(encode["inputs"]["pixels"], ["23", 0])
        self.assertEqual(encode["inputs"]["vae"], ["4", 2])
        scale = graph["10"]
        self.assertEqual(scale["class_type"], "LatentUpscale")
        self.assertEqual(scale["inputs"]["samples"], ["11", 0])
        sample = graph["12"]
        self.assertEqual(sample["inputs"]["latent_image"], ["10", 0])
        self.assertEqual(sample["inputs"]["denoise"], 0.55)

    def test_redraw_graph_latent_route_without_source_image_upscales_the_base_latent(self):
        graph = redraw_graph(self._plain_base(), 2048, 0.55, "fin", canvas=(832, 1664),
                           latent_route=True)
        scale = graph["10"]
        self.assertEqual(scale["class_type"], "LatentUpscale")
        self.assertEqual(scale["inputs"]["samples"], ["3", 0])
        self.assertNotIn("23", graph)

    def test_redraw_graph_keep_mask_wires_between_latent_source_and_sampler(self):
        graph = redraw_graph(self._plain_base(), 2048, 0.55, "fin", canvas=(832, 1664),
                           keep_mask_image="keep.png")
        load = graph["24"]
        self.assertEqual(load, {"class_type": "LoadImage", "inputs": {"image": "keep.png"}})
        to_mask = graph["25"]
        self.assertEqual(to_mask, {"class_type": "ImageToMask",
                                   "inputs": {"image": ["24", 0], "channel": "red"}})
        noise_mask = graph["26"]
        self.assertEqual(noise_mask["class_type"], "SetLatentNoiseMask")
        self.assertEqual(noise_mask["inputs"]["mask"], ["25", 0])
        # Pixel route: the noise mask sits between the VAEEncode and the sampler.
        self.assertEqual(noise_mask["inputs"]["samples"], ["11", 0])
        sample = graph["12"]
        self.assertEqual(sample["inputs"]["latent_image"], ["26", 0])

    def test_redraw_graph_keep_mask_wires_onto_the_latent_route_too(self):
        graph = redraw_graph(self._plain_base(), 2048, 0.55, "fin", canvas=(832, 1664),
                           latent_route=True, keep_mask_image="keep.png")
        noise_mask = graph["26"]
        self.assertEqual(noise_mask["inputs"]["samples"], ["10", 0])
        sample = graph["12"]
        self.assertEqual(sample["inputs"]["latent_image"], ["26", 0])

    def test_redraw_graph_keep_mask_wires_onto_the_source_image_latent_route(self):
        graph = redraw_graph(self._plain_base(), 2048, 0.55, "fin", canvas=(832, 1664),
                           latent_route=True, source_image="repaired.png",
                           keep_mask_image="keep.png")
        source_load_id = next(key for key, node in graph.items()
                              if node.get("inputs", {}).get("image") == "repaired.png")
        encode = graph["11"]
        self.assertEqual(encode["inputs"]["pixels"], [source_load_id, 0])
        scale = graph["10"]
        self.assertEqual(scale["inputs"]["samples"], ["11", 0])
        noise_mask = graph["26"]
        self.assertEqual(noise_mask["class_type"], "SetLatentNoiseMask")
        self.assertEqual(noise_mask["inputs"]["samples"], ["10", 0])
        sample = graph["12"]
        self.assertEqual(sample["inputs"]["latent_image"], ["26", 0])

    def test_redraw_graph_keep_mask_omitted_adds_nothing(self):
        with_none = redraw_graph(self._plain_base(), 2048, 0.55, "fin", canvas=(832, 1664),
                               keep_mask_image=None)
        without_kwarg = redraw_graph(self._plain_base(), 2048, 0.55, "fin", canvas=(832, 1664))
        self.assertEqual(with_none, without_kwarg)
        self.assertFalse(
            any(node.get("class_type") in ("LoadImage", "ImageToMask", "SetLatentNoiseMask")
                for node in with_none.values()))

    def test_redraw_graph_keep_mask_omitted_reproduces_the_pre_existing_graph_exactly(self):
        # Pinned by hand against the shape redraw_graph has always built for a
        # plain pixel-route pass -- if this ever changes without a
        # keep_mask_image argument in play, something broke the no-op case.
        graph = redraw_graph(self._plain_base(), 2048, 0.55, "fin", canvas=(832, 1664))
        self.assertEqual(graph, {
            "3": {"class_type": "KSampler",
                 "inputs": {"seed": 7, "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "DiffusersLoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                 "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
            "8": {"class_type": "VAEDecode",
                 "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                 "inputs": {"images": ["13", 0], "filename_prefix": "fin"}},
            "10": {"class_type": "ImageScale", "inputs": {
                "image": ["8", 0], "upscale_method": "bicubic",
                "width": 1024, "height": 2048, "crop": "disabled"}},
            "11": {"class_type": "VAEEncode",
                  "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}},
            "12": {"class_type": "KSampler", "inputs": {
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["11", 0], "seed": 7, "steps": 30, "cfg": 5.0,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.55}},
            "13": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["12", 0], "vae": ["4", 2]}},
        })

    def test_redraw_graph_rejects_a_saved_image_that_is_not_decoded(self):
        base = {
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 7, "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "DiffusersLoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
            "8": {"class_type": "ImageScale", "inputs": {}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }
        with self.assertRaisesRegex(ValueError, "must be fed by a VAEDecode"):
            redraw_graph(base, 2048, 0.45, "fin", canvas=(832, 1664))

    def _simple_base(self):
        return {
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 7, "positive": ["6", 0], "negative": ["7", 0]}},
            "4": {"class_type": "DiffusersLoader", "inputs": {}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1664}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "n"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
        }

    def test_redraw_graph_canvas_drives_sizes_not_the_bases_empty_latent_image(self):
        base = self._simple_base()
        # The base's own EmptyLatentImage stays 832x1664; a caller-given
        # canvas of a different shape is what the redraw is sized from.
        graph = redraw_graph(base, 2048, 0.45, "fin", canvas=(1000, 1000))
        scale = graph["10"]
        self.assertEqual(
            (scale["inputs"]["width"], scale["inputs"]["height"]), (2048, 2048))

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


class RedrawGraphGuidedStepsBaseTest(unittest.TestCase):
    """`redraw_graph` must redraw a guided two-stage Anima base at its guided
    cfg (the first stage's own), not the final stage's own cfg of 1.0."""

    def setUp(self):
        self.spec = render_spec("coffee", 42, "p")
        self.base = anima_graph.build_graph(self.spec)

    def test_pixel_route_reports_the_guided_cfg(self):
        graph = redraw_graph(self.base, 2048, 0.45, "fin",
                           canvas=(self.spec.width, self.spec.height))
        sample = next(node for node in graph.values()
                     if node["class_type"] == "KSampler")
        self.assertEqual(sample["inputs"]["cfg"], self.spec.cfg)
        self.assertNotEqual(sample["inputs"]["cfg"], 1.0)
        self.assertEqual(sample["inputs"]["steps"], self.spec.steps)
        self.assertEqual(sample["inputs"]["seed"], self.spec.seed)

    def test_latent_route_reports_the_guided_cfg(self):
        graph = redraw_graph(self.base, 2048, 0.45, "fin", latent_route=True,
                           canvas=(self.spec.width, self.spec.height))
        sample = next(node for node in graph.values()
                     if node["class_type"] == "KSampler")
        self.assertEqual(sample["inputs"]["cfg"], self.spec.cfg)
        self.assertNotEqual(sample["inputs"]["cfg"], 1.0)
        self.assertEqual(sample["inputs"]["steps"], self.spec.steps)
        self.assertEqual(sample["inputs"]["seed"], self.spec.seed)


if __name__ == "__main__":
    unittest.main()
