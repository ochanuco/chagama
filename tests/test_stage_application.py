"""Tests for the stage and lighting use cases and the dof of what they make.
All collaborators are fakes; the fake ComfyUI answers `wait_for` with the
SaveImage prefixes of the graph it was handed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from comfyui_recipes.application.dof import DofServices, dof
from comfyui_recipes.application.stage import StageServices, lighting, stage

LAYERS = {"layer-figure": b"figure-png", "layer-outline": b"outline-png",
          "layer-backdrop": b"backdrop-png"}
DELIVERY = {"kind": "deliver", "base_generation": "src", "transparent": True,
            "stroke_light": "nw", "outlines": [{"color": "#ffffff", "width": 1.0}]}


class ManagementFake:
    """Generations by id, each with its request parameters and assets."""

    def __init__(self, records):
        self.records = records
        self.calls = []

    def request(self, method, path, payload=None, multipart=None):
        self.calls.append((method, path, payload, multipart))
        if path.endswith("/context"):
            generation_id = path.split("/")[-2]
            return {"request": {"id": f"req-{generation_id}", "recipe": "yukari",
                                "kind": self.records[generation_id][0]["kind"],
                                "parameters": self.records[generation_id][0]},
                    "generations": []}
        if method == "POST" and path == "/api/v1/requests":
            return {"id": "request-id", "short_id": "req"}
        if method == "POST" and path.endswith("/jobs"):
            return {"id": "job-id"}
        if path.endswith("/generations"):
            return {"id": "out-1", "short_id": "gen", "canonical_url": "https://example/g/1"}
        return {}

    def fetch_generation_image(self, generation_id):
        return f"image:{generation_id}".encode()

    def list_assets(self, generation_id):
        return [{"role": role} for role in self.records.get(generation_id, ({}, {}))[1]]

    def fetch_asset(self, generation_id, role):
        return self.records[generation_id][1].get(role)

    def resolution(self):
        return next(call[2] for call in self.calls if call[1] == "/api/v1/requests")

    def asset_roles(self, generation_id="out-1"):
        return [call[3][0]["role"] for call in self.calls
                if call[0] == "POST" and call[1] == f"/api/v1/generations/{generation_id}/assets"]


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


def services_for(cls, management, directory):
    return cls(management=management, comfyui=ComfyFake(),
               git_metadata=lambda: {"commit": "c", "dirty": False},
               notifier=Notifier(), output_root=Path(directory), emit=lambda m: None)


def node(services, class_type):
    return next(n for n in services.comfyui.submitted[-1].values()
                if n["class_type"] == class_type)


class StageTest(unittest.TestCase):
    def test_a_pattern_inherits_the_deliverys_shadow_and_records_all_layers(self):
        management = ManagementFake({"dlv": (DELIVERY, LAYERS)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            result = stage("dlv", services, backdrop="stripes", inherit_shadow=True)
        inputs = node(services, "YukariStage")["inputs"]
        self.assertEqual((inputs["mode"], inputs["backdrop"], inputs["shadow_from"]),
                         ("pattern", "stripes", "nw"))
        self.assertEqual(result, {"generation_ids": ["out-1"]})
        self.assertEqual(management.resolution()["parameters"], {
            "kind": "stage", "base_generation": "dlv", "backdrop": "stripes",
            "shadow_from": "nw", "light": None})
        self.assertEqual(management.asset_roles(),
                         ["layer-figure", "layer-outline", "layer-backdrop"])

    def test_a_pattern_inherits_the_light_redraw_of_its_lineage(self):
        management = ManagementFake({"dlv": (DELIVERY, LAYERS)})
        light_redraw = {"request": {"parameters": {"kind": "redraw", "method": "light",
                                                   "scene": "moon", "from": "ne"}}}
        plain_request = management.request

        def request(method, path, payload=None, multipart=None):
            if method == "GET" and path == "/api/v1/generations/src":
                return light_redraw
            return plain_request(method, path, payload, multipart)

        management.request = request
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            stage("dlv", services, backdrop="dots", inherit_shadow=True, inherit_light=True)
            inputs = node(services, "YukariStage")["inputs"]
            self.assertEqual((inputs["light_scene"], inputs["light_from"]), ("moon", "ne"))
            # The delivery's own stroke light still throws the shadow.
            self.assertEqual(inputs["shadow_from"], "nw")
            stage("dlv", services, backdrop="dots", light=None, inherit_light=False,
                  key_prefix="no-scene")
            self.assertNotIn("light_scene", node(services, "YukariStage")["inputs"])

    def test_a_photo_places_the_layers_and_references_the_photo(self):
        management = ManagementFake({"dlv": (DELIVERY, LAYERS)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            stage("dlv", services, photo_id="pic", place=(0.6, 1.1), height=0.9)
        inputs = node(services, "YukariStage")["inputs"]
        self.assertEqual((inputs["mode"], inputs["place_x"], inputs["height"]),
                         ("photo", 0.6, 0.9))
        resolution = management.resolution()
        self.assertEqual(resolution["parameters"]["photo"], "pic")
        self.assertIn("pic", [ref["source_generation_id"] for ref in resolution["references"]])

    def test_only_a_delivery_can_be_staged(self):
        management = ManagementFake({"stg": ({"kind": "stage", "base_generation": "dlv"},
                                             LAYERS)})
        with (tempfile.TemporaryDirectory() as directory,
              self.assertRaisesRegex(SystemExit, "deliver の出力")):
            stage("stg", services_for(StageServices, management, directory),
                  backdrop="dots")


class LightingTest(unittest.TestCase):
    def test_lighting_relights_and_carries_the_backdrop_over(self):
        management = ManagementFake({"stg": ({"kind": "stage", "base_generation": "dlv",
                                              "backdrop": "dots"}, LAYERS)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            lighting("stg", services, direction="ne", strength=0.5)
        inputs = node(services, "YukariRelight")["inputs"]
        self.assertEqual((inputs["direction"], inputs["strength"]), ("ne", 0.5))
        self.assertEqual(management.asset_roles(),
                         ["layer-figure", "layer-outline", "layer-backdrop"])
        backdrop = next(call for call in management.calls if call[3]
                        and call[3][0].get("role") == "layer-backdrop")
        self.assertEqual(backdrop[3][3], b"backdrop-png")

    def test_lighting_twice_and_a_transparent_delivery_are_refused(self):
        management = ManagementFake({
            "lit": ({"kind": "lighting", "base_generation": "stg"}, LAYERS),
            "dlv": (DELIVERY, LAYERS)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            with self.assertRaisesRegex(SystemExit, "もう光"):
                lighting("lit", services, direction=None, strength=0.5)
            with self.assertRaisesRegex(SystemExit, "stage"):
                lighting("dlv", services, direction=None, strength=0.5)


class StagedDofTest(unittest.TestCase):
    def test_a_photo_set_picture_blurs_by_the_photos_depth(self):
        management = ManagementFake({
            "lit": ({"kind": "lighting", "base_generation": "stg"}, LAYERS),
            "stg": ({"kind": "stage", "base_generation": "dlv", "photo": "pic",
                     "place": [0.6, 1.1], "height": 0.9}, LAYERS),
            "dlv": (DELIVERY, LAYERS), "pic": ({"kind": "photo"}, {})})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(DofServices, management, directory)
            dof("lit", services, focus=(0.6, 0.3), f_number=1.4, scope={},
                photo_blur={"stand_at": None, "nearer": 0.2, "tilt": 0.5, "bokeh": 1.0})
        inputs = node(services, "YukariPhotoDepthOfField")["inputs"]
        self.assertEqual((inputs["place_x"], inputs["place_y"], inputs["height"]),
                         (0.6, 1.1, 0.9))
        self.assertEqual((inputs["tilt"], inputs["bokeh"], inputs["focus"]),
                         (0.5, 1.0, "point"))
        self.assertIn(("dof-lit-source-", b"image:pic"),
                      [(name[:15], data) for name, data in services.comfyui.uploaded])
        parameters = management.resolution()["parameters"]
        self.assertEqual((parameters["tilt"], parameters["stand_at"]), (0.5, None))
        self.assertNotIn("scope", parameters)

    def test_a_pattern_set_picture_takes_the_deliverys_source_depth(self):
        management = ManagementFake({
            "stg": ({"kind": "stage", "base_generation": "dlv", "backdrop": "dots"}, LAYERS),
            "dlv": (DELIVERY, LAYERS), "src": ({"kind": "generate"}, {})})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(DofServices, management, directory)
            dof("stg", services, focus=(0.5, 0.5), f_number=2.8,
                scope={"figure": True, "outline": True, "backdrop": True})
            self.assertIn("backdrop", node(services, "YukariDepthOfField")["inputs"])
            self.assertIn(b"image:src", [data for _, data in services.comfyui.uploaded])
            with self.assertRaisesRegex(SystemExit, "写真に置いた絵"):
                dof("stg", services, focus=(0.5, 0.5), f_number=2.8, scope={},
                    photo_blur={"stand_at": None, "nearer": 0, "tilt": 0, "bokeh": 0})


if __name__ == "__main__":
    unittest.main()
