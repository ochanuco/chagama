"""Tests for the redraw graphs and use case.

All collaborators are fakes: nothing here opens a socket or talks to ComfyUI.
The fake ComfyUI answers `wait_for` with the SaveImage prefixes of the graph
it was handed, so what the use case uploads follows what the graph saves.
"""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from chagama.application.cut_assets import current_cut
from chagama.application.redraw import HIRES_DENOISE, hires_pixels
from chagama.application.ingest import classify_redraw_outputs
from chagama.application.redraw import RedrawServices, redraw
from chagama.domain.yukari import delivery_style
from chagama.domain.yukari.delivery_style import Light
from chagama.infrastructure.comfyui.refinement_graph import DEPTH_NODE
from PIL import Image

ANIMA_GRAPH = {
    "1": {"class_type": "UNETLoader", "inputs": {}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
    "5": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 1024, "height": 1640, "batch_size": 1}},
    "3": {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "seed": 5, "steps": 10, "cfg": 2,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        "latent_image": ["5", 0], "positive": ["6", 0], "negative": ["7", 0]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "p"}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "n"}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage",
          "inputs": {"images": ["8", 0], "filename_prefix": "base"}},
}
SDXL_GRAPH = {key: node for key, node in ANIMA_GRAPH.items() if key != "1"}
SDXL_GRAPH["3"] = {**ANIMA_GRAPH["3"], "inputs": {
    **ANIMA_GRAPH["3"]["inputs"], "model": ["4", 0]}}


def classes(graph: dict) -> list[str]:
    return [node["class_type"] for node in graph.values()]


def nodes_of(graph: dict, class_type: str) -> list[tuple[str, dict]]:
    return [(key, node) for key, node in graph.items()
            if node["class_type"] == class_type]


def saved_prefixes(graph: dict) -> list[str]:
    return [node["inputs"]["filename_prefix"] for _, node in
            nodes_of(graph, "SaveImage")]


class ManagementFake:
    """Generations by id: each holds its request and its job's graph."""

    def __init__(self, records, *, assets=None):
        self.records = records
        self.assets = dict(assets or {})
        self.calls = []

    def request(self, method, path, payload=None, multipart=None):
        self.calls.append((method, path, payload, multipart))
        if method == "GET" and path.startswith("/api/v1/generations/"):
            parts = path.split("/")
            return self.records[parts[4]]
        if (method == "PUT" and path.endswith("/resolution")) or (
                method == "POST" and path == "/api/v1/requests"):
            return {"id": "request-id", "short_id": "req"}
        if method == "POST" and path.endswith("/jobs"):
            return {"id": "job-id"}
        if path.endswith("/generations"):
            return {"id": "redrawn", "short_id": "gen",
                    "canonical_url": "https://example/g/1"}
        return {}

    def fetch_generation_image(self, generation_id):
        return b"picked:" + generation_id.encode()

    def list_assets(self, generation_id):
        return [{"role": role} for role in self.assets]

    def fetch_asset(self, generation_id, role):
        return self.assets.get(role)

    def asset_uploads(self):
        return [call for call in self.calls
                if call[0] == "POST" and call[1].endswith("/assets")]

    def job_graph(self):
        patch = next(call for call in self.calls
                     if call[0] == "PATCH" and (call[2] or {}).get("graph"))
        return patch[2]["graph"]


class ComfyFake:
    def __init__(self):
        self.uploaded = []
        self.submitted = []

    def upload_image(self, name, data):
        self.uploaded.append((name, data))
        return f"uploaded-{name}"

    def submit(self, graph):
        self.submitted.append(graph)
        return "prompt-id"

    def knows(self, prompt_id):
        return True

    def wait_for(self, prompt_id):
        return [{"filename": f"{prefix}_00001_.png"}
                for prefix in saved_prefixes(self.submitted[-1])]

    def fetch(self, image):
        return b"bytes:" + image["filename"].encode()


class RecordingNotifier:
    def __init__(self):
        self.calls = []

    def send(self, *args):
        self.calls.append(args)


def record(kind=None, *, graph=None, request_kind="generate", **parameters):
    return {"request": {"id": "r", "recipe": "yukari", "kind": request_kind,
                        "parameters": {**({"kind": kind} if kind else {}),
                                       **parameters}},
            "comfy_job": {"graph": graph} if graph is not None else None}


def services(directory, management, **overrides) -> RedrawServices:
    kwargs = dict(
        management=management, comfyui=ComfyFake(),
        graph_from_png=lambda data: None,
        git_metadata=lambda: {"commit": "commit", "dirty": False},
        notifier=RecordingNotifier(), output_root=Path(directory),
        image_size=lambda data: (832, 1664), emit=lambda message: None)
    kwargs.update(overrides)
    return RedrawServices(**kwargs)


def cut_asset(**overrides) -> bytes:
    return json.dumps({**current_cut(), **overrides}).encode()


class RedrawCase(unittest.TestCase):
    def run_redraw(self, records, source="src", *, assets=None, **kwargs):
        management = ManagementFake(records, assets=assets)
        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, management)
            result = redraw(source, svc, context=records[source], **kwargs)
        return svc, management, result

    def refusal(self, records, pattern, source="src", **kwargs):
        management = ManagementFake(records)
        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, management)
            with self.assertRaisesRegex(SystemExit, pattern):
                redraw(source, svc, context=records[source], **kwargs)
        self.assertEqual(svc.comfyui.submitted, [])


class CanvasTest(RedrawCase):
    def test_the_graph_is_the_redraw_alone_with_one_save(self):
        svc, management, result = self.run_redraw(
            {"src": record(graph=ANIMA_GRAPH)}, method="canvas")
        graph = svc.comfyui.submitted[0]
        self.assertEqual(saved_prefixes(graph), ["rdw-src"])
        self.assertFalse({"YukariDeliver", "BiRefNetRMBG", "YukariMatting",
                          "RemoveBackground", "MaskToImage"} & set(classes(graph)))
        [(_, resample)] = [(k, n) for k, n in nodes_of(graph, "KSampler")
                           if n["inputs"]["denoise"] == delivery_style.REDRAW_DENOISE]
        self.assertEqual(resample["inputs"]["seed"], 5)
        [(_, scale)] = nodes_of(graph, "ImageScale")
        self.assertEqual(max(scale["inputs"]["width"], scale["inputs"]["height"]),
                         delivery_style.REDRAW_SIZE)
        self.assertEqual(management.asset_uploads(), [])
        self.assertEqual(result["generation_ids"], ["redrawn"])

    def test_the_latent_route_upscales_the_latent(self):
        svc, _, _ = self.run_redraw(
            {"src": record(graph=ANIMA_GRAPH)}, method="canvas",
            latent_route=True, denoise=0.5, size=2048)
        graph = svc.comfyui.submitted[0]
        self.assertIn("LatentUpscale", classes(graph))
        self.assertNotIn("ImageScale", classes(graph))
        self.assertEqual(saved_prefixes(graph), ["rdw-src"])

    def test_the_record_carries_the_resolved_options_and_refines_the_source(self):
        management = ManagementFake({"src": record(graph=ANIMA_GRAPH)})
        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, management)
            redraw("src", svc, method="canvas", denoise=0.5, size=2048,
                   latent_route=True, request_id="req-1",
                   key_prefix="request:req-1", context=record(graph=ANIMA_GRAPH))
        resolution = next(call for call in management.calls
                          if call[1] == "/api/v1/requests/req-1/resolution")[2]
        self.assertEqual(resolution["parameters"], {
            "kind": "redraw", "method": "canvas", "base_generation": "src",
            "size": 2048, "denoise": 0.5, "route": "latent",
            "finalizer": delivery_style.REDRAW_MODEL})
        job = next(call for call in management.calls
                   if call[0] == "POST" and call[1].endswith("/jobs"))
        self.assertEqual(job[2]["seed"], 5)
        self.assertEqual(job[2]["source_generation_id"], "src")
        self.assertEqual(saved_prefixes(management.job_graph()), ["rdw-src"])
        upload = next(call for call in management.calls
                      if call[1].endswith("/generations"))
        self.assertEqual(upload[3][2], "rdw-src_00001_.png")

    def test_keep_regions_stage_a_mask_and_are_recorded(self):
        svc, management, _ = self.run_redraw(
            {"src": record(graph=ANIMA_GRAPH)}, method="canvas",
            keep_regions=[[0.1, 0.1, 0.4, 0.4]], keep_strength=0.3)
        graph = svc.comfyui.submitted[0]
        self.assertIn("SetLatentNoiseMask", classes(graph))
        self.assertTrue(any("keep-mask" in name for name, _ in svc.comfyui.uploaded))

    def test_a_repaired_or_redrawn_source_is_redrawn_from_its_pixels(self):
        for kind in ("repair", "masked_redraw", "redraw"):
            with self.subTest(kind=kind):
                records = {
                    "src": record(kind, base_generation="mid"),
                    "mid": record("redraw", base_generation="gen"),
                    "gen": record(graph=ANIMA_GRAPH),
                }
                svc, management, _ = self.run_redraw(records, method="canvas")
                graph = svc.comfyui.submitted[0]
                [(load_id, load)] = nodes_of(graph, "LoadImage")
                self.assertIn("source.png", load["inputs"]["image"])
                [(_, scale)] = nodes_of(graph, "ImageScale")
                self.assertEqual(scale["inputs"]["image"], [load_id, 0])
                [(_, resample)] = [
                    (k, n) for k, n in nodes_of(graph, "KSampler")
                    if n["inputs"]["denoise"] == delivery_style.REDRAW_DENOISE]
                self.assertEqual(resample["inputs"]["seed"], 5)
                self.assertIn("rdw-src", saved_prefixes(graph))

    def test_only_anima_sources_are_redrawn(self):
        self.refusal({"src": record(graph=SDXL_GRAPH)}, "Anima", method="canvas")


class SourceRulesTest(RedrawCase):
    def test_delivered_sources_are_refused_for_every_method(self):
        cases = [
            dict(request_kind="deliver"),
            dict(request_kind="finalize"),
            dict(kind="deliver"),
            dict(kind="hires-chain"),
            dict(kind="repair", deliver_only=True),
        ]
        for method, kwargs in (("canvas", {}), ("hires", {"hires": 2048,
                                                          "denoise": 0.45}),
                               ("light", {"light": Light("moon", "se")})):
            for case in cases:
                with self.subTest(method=method, case=case):
                    self.refusal({"src": record(graph=ANIMA_GRAPH, **case)},
                                 "納品済み", method=method, **kwargs)

    def test_a_layerdiffuse_source_is_refused(self):
        graph = {**ANIMA_GRAPH, "20": {"class_type": "LayeredDiffusionApply",
                                       "inputs": {}}}
        self.refusal({"src": record(graph=graph)}, "LayerDiffuse", method="canvas")

    def test_hires_refuses_repaired_and_redrawn_outputs(self):
        for kind in ("repair", "masked_redraw", "redraw"):
            with self.subTest(kind=kind):
                self.refusal({"src": record(kind, base_generation="gen"),
                              "gen": record(graph=ANIMA_GRAPH)},
                             "repair や masked_redraw で直した絵",
                             method="hires", hires=2048, denoise=0.45)

    def test_hires_refuses_a_stitched_base(self):
        stitched = copy.deepcopy(ANIMA_GRAPH)
        stitched["30"] = {"class_type": "InpaintStitchImproved",
                          "inputs": {"inpainted_image": ["8", 0]}}
        stitched["9"]["inputs"]["images"] = ["30", 0]
        self.refusal({"src": record(graph=stitched)},
                     "repair や masked_redraw で直した絵",
                     method="hires", hires=2048, denoise=0.45)

    def test_hires_refuses_a_non_anima_source(self):
        self.refusal({"src": record(graph=SDXL_GRAPH)}, "Anima",
                     method="hires", hires=2048, denoise=0.45)

    def test_a_source_without_a_graph_is_refused(self):
        self.refusal({"src": record(graph=None)}, "graph が無い",
                     method="hires", hires=2048, denoise=0.45)

    def test_light_accepts_a_repaired_source(self):
        records = {"src": record("repair", base_generation="gen"),
                   "gen": record(graph=ANIMA_GRAPH)}
        svc, _, result = self.run_redraw(
            records, method="light", light=Light("moon", "se"))
        self.assertEqual(result["generation_ids"], ["redrawn"])
        [(_, load)] = [(k, n) for k, n in nodes_of(svc.comfyui.submitted[0], "LoadImage")
                       if "source" in n["inputs"]["image"]]
        self.assertIn("rdw-src-source", load["inputs"]["image"])

    def test_the_lineage_walk_is_bounded(self):
        records = {"src": record("redraw", base_generation="g0")}
        for index in range(12):
            records[f"g{index}"] = record("redraw", base_generation=f"g{index + 1}")
        self.refusal(records, "たどれません", method="canvas")


class HiresTest(RedrawCase):
    def test_the_hires_graph_is_saved_under_the_redraw_prefix(self):
        svc, management, _ = self.run_redraw(
            {"src": record(graph=ANIMA_GRAPH)}, method="hires",
            hires=2048, denoise=0.5)
        graph = svc.comfyui.submitted[0]
        self.assertEqual(saved_prefixes(graph), ["rdw-src"])
        [(_, upscale)] = nodes_of(graph, "LatentUpscale")
        self.assertEqual(
            round(upscale["inputs"]["width"] * upscale["inputs"]["height"]
                  / hires_pixels(2048), 1), 1.0)
        self.assertFalse({"YukariDeliver", "BiRefNetRMBG"} & set(classes(graph)))
        resolution = next(call for call in management.calls
                          if call[0] == "POST" and call[1] == "/api/v1/requests")[2]
        self.assertEqual(resolution["parameters"], {
            "kind": "redraw", "method": "hires", "base_generation": "src",
            "hires": 2048, "denoise": 0.5})

    def test_the_default_denoise_is_the_hires_default(self):
        self.assertEqual(HIRES_DENOISE, 0.45)


class LightTest(RedrawCase):
    LIGHT = Light("moon", "se")

    def test_a_fresh_cut_is_computed_saved_and_attached_to_the_source(self):
        svc, management, result = self.run_redraw(
            {"src": record(graph=ANIMA_GRAPH)}, method="light", light=self.LIGHT)
        graph = svc.comfyui.submitted[0]
        self.assertIn("BiRefNetRMBG", classes(graph))
        self.assertIn("YukariMatting", classes(graph))
        self.assertIn(DEPTH_NODE, classes(graph))
        self.assertEqual(sorted(saved_prefixes(graph)),
                         ["rdw-src", "rdw-src-alpha", "rdw-src-depth"])
        [(matting_id, _)] = nodes_of(graph, "YukariMatting")
        [(_, lit)] = nodes_of(graph, "YukariLight")
        self.assertEqual(lit["inputs"]["matte"], [matting_id, 0])
        roles = [call[3][0]["role"] for call in management.asset_uploads()]
        self.assertEqual(roles, ["alpha", "depth", "cut"])
        for call in management.asset_uploads():
            self.assertEqual(call[1], "/api/v1/generations/src/assets")
        self.assertEqual(json.loads(management.asset_uploads()[2][3][3]),
                         current_cut())
        self.assertEqual(result["generation_ids"], ["redrawn"])
        generation_uploads = [call for call in management.calls
                              if call[1].endswith("/generations")]
        self.assertEqual([call[3][2] for call in generation_uploads],
                         ["rdw-src_00001_.png"])

    def test_a_reusable_cut_is_loaded_and_nothing_is_computed(self):
        svc, management, _ = self.run_redraw(
            {"src": record(graph=ANIMA_GRAPH)}, method="light", light=self.LIGHT,
            assets={"alpha": b"alpha-png", "depth": b"depth-png",
                    "cut": cut_asset()})
        graph = svc.comfyui.submitted[0]
        self.assertFalse({"BiRefNetRMBG", "YukariMatting", DEPTH_NODE,
                          "RemoveBackground"} & set(classes(graph)))
        self.assertEqual(saved_prefixes(graph), ["rdw-src"])
        loads = {node["inputs"]["image"]: key for key, node in nodes_of(graph, "LoadImage")}
        [(to_mask_id, to_mask)] = nodes_of(graph, "ImageToMask")
        self.assertEqual(to_mask["inputs"]["channel"], "red")
        [(_, lit)] = nodes_of(graph, "YukariLight")
        self.assertEqual(lit["inputs"]["matte"], [to_mask_id, 0])
        depth_load = next(key for image, key in loads.items() if "depth-in" in image)
        self.assertEqual(lit["inputs"]["depth"], [depth_load, 0])
        self.assertEqual(management.asset_uploads(), [])

    def test_a_stale_alpha_is_recomputed_and_the_depth_reused(self):
        stale = {**current_cut()["alpha"], "trimap_px": 3}
        svc, management, _ = self.run_redraw(
            {"src": record(graph=ANIMA_GRAPH)}, method="light", light=self.LIGHT,
            assets={"alpha": b"old", "depth": b"depth-png",
                    "cut": cut_asset(alpha=stale)})
        graph = svc.comfyui.submitted[0]
        self.assertIn("YukariMatting", classes(graph))
        self.assertNotIn(DEPTH_NODE, classes(graph))
        self.assertEqual([call[3][0]["role"] for call in management.asset_uploads()],
                         ["alpha", "cut"])
        self.assertEqual(json.loads(management.asset_uploads()[1][3][3]),
                         current_cut())

    def test_the_light_is_recorded_and_seed_and_prompt_come_from_the_graph(self):
        management = ManagementFake({"src": record(graph=ANIMA_GRAPH)})
        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, management)
            redraw("src", svc, method="light", light=self.LIGHT,
                   request_id="req-1", key_prefix="request:req-1",
                   context=record(graph=ANIMA_GRAPH))
        resolution = next(call for call in management.calls
                          if call[1] == "/api/v1/requests/req-1/resolution")[2]
        self.assertEqual(resolution["parameters"], {
            "kind": "redraw", "method": "light", "base_generation": "src",
            "scene": "moon", "from": "se"})
        job = next(call for call in management.calls
                   if call[0] == "POST" and call[1].endswith("/jobs"))
        self.assertEqual(job[2]["seed"], 5)
        self.assertEqual(job[2]["source_generation_id"], "src")


def png(mode: str, colour) -> bytes:
    output = io.BytesIO()
    Image.new(mode, (4, 4), colour).save(output, "PNG")
    return output.getvalue()


class CutComfyFake(ComfyFake):
    """Answers the cut's alpha and figure colour with real PNGs."""

    def fetch(self, image):
        if image["filename"].startswith("rdw-src-out-alpha"):
            return png("L", 200)
        if image["filename"].startswith("rdw-src-out-foreground"):
            return png("RGB", (10, 20, 30))
        return super().fetch(image)


class CutTest(unittest.TestCase):
    def run_cut(self, **kwargs):
        records = {"src": record(graph=ANIMA_GRAPH)}
        management = ManagementFake(records)
        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, management, comfyui=CutComfyFake())
            result = redraw("src", svc, context=records["src"], cut=True,
                            key_prefix="request:r1", **kwargs)
        return svc, management, result

    def test_the_redrawn_picture_is_cut_and_its_cut_attached_to_it(self):
        svc, management, result = self.run_cut(method="canvas")
        self.assertEqual(len(svc.comfyui.submitted), 2)
        cut = svc.comfyui.submitted[1]
        self.assertTrue({"BiRefNetRMBG", "YukariMatting", DEPTH_NODE,
                         "YukariForeground"} <= set(classes(cut)))
        self.assertEqual(sorted(saved_prefixes(cut)), [
            "rdw-src-out-alpha", "rdw-src-out-depth", "rdw-src-out-foreground"])
        [(_, load)] = nodes_of(cut, "LoadImage")
        self.assertEqual(svc.comfyui.uploaded[-1][1], b"bytes:rdw-src_00001_.png")
        self.assertTrue(load["inputs"]["image"].startswith("uploaded-rdw-src-out-in-"))
        uploads = management.asset_uploads()
        self.assertEqual([call[3][0]["role"] for call in uploads],
                         ["alpha", "depth", "cut", "cutout"])
        self.assertEqual({call[1] for call in uploads},
                         {"/api/v1/generations/redrawn/assets"})
        self.assertEqual(json.loads(uploads[2][3][3]), current_cut())
        self.assertEqual(uploads[0][3][0]["idempotency_key"],
                         "request:r1:output:job:0:asset:alpha")
        cutout = Image.open(io.BytesIO(uploads[3][3][3]))
        self.assertEqual(cutout.mode, "RGBA")
        self.assertEqual(cutout.getpixel((1, 1)), (10, 20, 30, 200))
        resolution = next(call[2] for call in management.calls
                          if call[1] == "/api/v1/requests")
        self.assertTrue(resolution["parameters"]["cut"])
        self.assertEqual(result["generation_ids"], ["redrawn"])

    def test_a_light_redraw_cuts_its_source_and_its_result_apart(self):
        _, management, _ = self.run_cut(method="light", light=Light("moon", "se"))
        targets = [(call[1], call[3][0]["role"]) for call in management.asset_uploads()]
        self.assertEqual(targets, [
            ("/api/v1/generations/src/assets", "alpha"),
            ("/api/v1/generations/src/assets", "depth"),
            ("/api/v1/generations/src/assets", "cut"),
            ("/api/v1/generations/redrawn/assets", "alpha"),
            ("/api/v1/generations/redrawn/assets", "depth"),
            ("/api/v1/generations/redrawn/assets", "cut"),
            ("/api/v1/generations/redrawn/assets", "cutout")])
        keys = [call[3][0]["idempotency_key"] for call in management.asset_uploads()]
        self.assertEqual(len(set(keys)), len(keys))

    def test_without_cut_nothing_is_cut(self):
        records = {"src": record(graph=ANIMA_GRAPH)}
        management = ManagementFake(records)
        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, management)
            redraw("src", svc, context=records["src"], method="canvas")
        self.assertEqual(len(svc.comfyui.submitted), 1)
        self.assertEqual(management.asset_uploads(), [])
        resolution = next(call[2] for call in management.calls
                          if call[1] == "/api/v1/requests")
        self.assertNotIn("cut", resolution["parameters"])


class OutputsTest(unittest.TestCase):
    def test_outputs_are_classified_by_role(self):
        outputs = [{"filename": f"rdw-g{suffix}_00001_.png"} for suffix in
                   ("", "-alpha", "-depth")]
        found = classify_redraw_outputs(outputs)
        self.assertEqual({role: [out["filename"] for out in outs]
                          for role, outs in found.items()}, {
            "alpha": ["rdw-g-alpha_00001_.png"],
            "depth": ["rdw-g-depth_00001_.png"],
            "picture": ["rdw-g_00001_.png"]})

    def test_a_missing_picture_output_fails_loudly(self):
        class Silent(ComfyFake):
            def wait_for(self, prompt_id):
                return []

        management = ManagementFake({"src": record(graph=ANIMA_GRAPH)})
        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, management, comfyui=Silent())
            with self.assertRaisesRegex(SystemExit, "picture"):
                redraw("src", svc, method="canvas",
                       context=record(graph=ANIMA_GRAPH))


if __name__ == "__main__":
    unittest.main()
