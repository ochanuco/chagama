"""Tests for the photo use case. All collaborators are fakes; the fake ComfyUI
answers `wait_for` with the SaveImage prefixes of the graph it was handed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chagama.application.cut_assets import current_cut
from chagama.application.photo import PhotoServices, photo
from chagama.infrastructure.comfyui.refinement_graph import DEPTH_NODE

LAYERS = {"layer-figure": b"figure-png", "layer-outline": b"outline-png",
          "layer-backdrop": b"backdrop-png"}


class ManagementFake:
    def __init__(self, *, request_kind="deliver", parameters=None, assets=None,
                 photo_assets=None):
        self.calls = []
        self.assets = {"dlv": dict(LAYERS if assets is None else assets),
                       "pic": dict(photo_assets or {})}
        self.context = {"request": {"id": "deliver-request", "recipe": "yukari",
                                    "kind": request_kind,
                                    "parameters": ({"kind": "deliver",
                                                    "base_generation": "src"}
                                                   if parameters is None else parameters)},
                        "generations": []}

    def request(self, method, path, payload=None, multipart=None):
        self.calls.append((method, path, payload, multipart))
        if path.endswith("/context"):
            return self.context
        if method == "POST" and path == "/api/v1/requests":
            return {"id": "request-id", "short_id": "req"}
        if method == "POST" and path.endswith("/jobs"):
            return {"id": "job-id"}
        if path.endswith("/generations"):
            return {"id": "staged-1", "short_id": "gen",
                    "canonical_url": "https://example/g/1"}
        return {}

    def fetch_generation_image(self, generation_id):
        return f"image:{generation_id}".encode()

    def list_assets(self, generation_id):
        return [{"role": role} for role in self.assets.get(generation_id, {})]

    def fetch_asset(self, generation_id, role):
        return self.assets.get(generation_id, {}).get(role)

    def asset_uploads(self):
        return [call for call in self.calls
                if call[0] == "POST" and call[1].endswith("/assets")]


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
        return [{"filename": f"{node['inputs']['filename_prefix']}_00001_.png"}
                for node in self.submitted[-1].values()
                if node["class_type"] == "SaveImage"]

    def fetch(self, image):
        return b"bytes:" + image["filename"].encode()


class Notifier:
    def send(self, *args):
        pass


def run(management=None, **kwargs):
    management = management or ManagementFake()
    with tempfile.TemporaryDirectory() as directory:
        services = PhotoServices(
            management=management, comfyui=ComfyFake(),
            git_metadata=lambda: {"commit": "c", "dirty": False},
            notifier=Notifier(), output_root=Path(directory), emit=lambda m: None,
            image_size=lambda data: (1448, 1808))
        options = {"photo_id": "pic", "place": (0.3, 0.9), "height": 0.6,
                   "stand_at": None, "nearer": 0.0, "focus": "subject",
                   "f_number": 2.0, **kwargs}
        result = photo("dlv", services, **options)
    return services, result


def stage_node(graph):
    return next(node for node in graph.values() if node["class_type"] == "YukariPhotoStage")


class PhotoSourceTest(unittest.TestCase):
    def test_a_picture_that_was_not_delivered_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "納品の絵だけ"):
            run(ManagementFake(request_kind="generate", parameters={}))

    def test_a_delivery_without_layers_and_a_dof_output_are_refused(self):
        for management in (ManagementFake(assets={"layer-figure": b"f"}),
                           ManagementFake(request_kind="dof",
                                          parameters={"kind": "dof", "base_generation": "d"})):
            with self.subTest(), self.assertRaisesRegex(SystemExit, "層が無い"):
                run(management)


class PhotoSaturatedSourceTest(unittest.TestCase):
    def test_a_saturated_delivery_can_be_stood_in_a_photo(self):
        management = ManagementFake(request_kind="saturation", parameters={
            "kind": "saturation", "base_generation": "src", "strength": 1.0})
        _, result = run(management)
        self.assertEqual(result["generation_ids"], ["staged-1"])


class PhotoOutputTest(unittest.TestCase):
    def test_photo_outputs_count_as_delivered(self):
        from chagama.application.picture_source import is_delivered
        self.assertTrue(is_delivered({"kind": "photo"}))
        self.assertTrue(is_delivered({"kind": "saturation"}))
        self.assertTrue(is_delivered({"kind": "import",
                                      "parameters": {"kind": "saturation"}}))


class PhotoUseCaseTest(unittest.TestCase):
    def test_the_photo_depth_is_built_and_kept_on_the_photo(self):
        services, result = run()
        graph = services.comfyui.submitted[0]
        self.assertIn(DEPTH_NODE, [node["class_type"] for node in graph.values()])
        self.assertIn(b"image:pic", [data for _, data in services.comfyui.uploaded])
        uploads = services.management.asset_uploads()
        self.assertEqual([(call[1], call[3][0]["role"]) for call in uploads], [
            ("/api/v1/generations/pic/assets", "depth"),
            ("/api/v1/generations/pic/assets", "cut")])
        self.assertEqual(result["generation_ids"], ["staged-1"])

    def test_a_current_photo_depth_is_reused(self):
        management = ManagementFake(photo_assets={
            "depth": b"depth-png", "cut": json.dumps(current_cut()).encode()})
        services, _ = run(management)
        graph = services.comfyui.submitted[0]
        self.assertNotIn(DEPTH_NODE, [node["class_type"] for node in graph.values()])
        self.assertEqual(management.asset_uploads(), [])

    def test_only_the_figure_and_outline_layers_go_into_the_graph(self):
        services, _ = run()
        uploaded = [data for _, data in services.comfyui.uploaded]
        self.assertIn(b"figure-png", uploaded)
        self.assertIn(b"outline-png", uploaded)
        self.assertNotIn(b"backdrop-png", uploaded)

    def test_the_stand_and_focus_reach_the_node(self):
        services, _ = run(stand_at=(0.8, 0.4), focus=(0.8, 0.35), nearer=0.2)
        inputs = stage_node(services.comfyui.submitted[0])["inputs"]
        self.assertEqual((inputs["stand"], inputs["stand_x"], inputs["stand_y"]),
                         ("point", 0.8, 0.4))
        self.assertEqual((inputs["focus"], inputs["focus_x"], inputs["focus_y"]),
                         ("point", 0.8, 0.35))
        self.assertEqual((inputs["nearer"], inputs["tilt"]), (0.2, 0.0))
        self.assertEqual(inputs["focal_mm"], 85.0)
        services, _ = run()
        inputs = stage_node(services.comfyui.submitted[0])["inputs"]
        self.assertEqual((inputs["stand"], inputs["focus"]), ("base", "subject"))

    def test_the_request_records_its_kind_base_photo_and_options(self):
        services, _ = run(stand_at=(0.8, 0.4))
        resolution = next(call for call in services.management.calls
                          if call[0] == "POST" and call[1] == "/api/v1/requests")[2]
        self.assertEqual(resolution["parameters"], {
            "kind": "photo", "base_generation": "dlv", "photo": "pic",
            "place": [0.3, 0.9], "height": 0.6, "stand_at": [0.8, 0.4],
            "nearer": 0.0, "focus": "subject", "f_number": 2.0, "tilt": 0.0, "bokeh": 0.0,
            "focal_length": "85mm"})
        self.assertEqual([ref["source_generation_id"] for ref in resolution["references"]],
                         ["dlv", "pic"])


if __name__ == "__main__":
    unittest.main()
