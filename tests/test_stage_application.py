"""Tests for the stage and lighting use cases and the dof of what they make.
All collaborators are fakes; the fake ComfyUI answers `wait_for` with the
SaveImage prefixes of the graph it was handed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chagama.application.dof import DofServices, dof
from chagama.application.stage import StageServices, lighting, saturation, stage
from chagama.infrastructure.imaging.stage import _png

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
    extra = {"image_size": lambda data: (1448, 1808)} if cls is DofServices else {}
    return cls(management=management, comfyui=ComfyFake(),
               git_metadata=lambda: {"commit": "c", "dirty": False},
               notifier=Notifier(), output_root=Path(directory), emit=lambda m: None,
               **extra)


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

    def test_a_frame_resolves_its_canvas_from_the_layers_and_records_it(self):
        figure = np.zeros((80, 60, 4), np.float32)
        figure[10:70, 15:45] = (200, 60, 120, 255)
        layers = {"layer-figure": _png(figure, "RGBA"),
                  "layer-outline": _png(np.zeros_like(figure), "RGBA")}
        frame = {"box": [0.25, 0.25, 1.0, 1.0],
                 "sides": {"top": "out", "right": "cut", "bottom": "cut", "left": "cut"},
                 "line": {"color": "#885b80", "width": 5.0}, "paper": "#ffffff",
                 "canvas": None, "expand": True, "strokes": []}
        management = ManagementFake({"dlv": (DELIVERY, layers)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            result = stage("dlv", services, backdrop="stripes", frame=frame)
        resolved = {**frame, "canvas": [9 / 60, 6 / 80, 66 / 60, 86 / 80]}
        self.assertEqual(json.loads(node(services, "YukariStage")["inputs"]["frame"]),
                         resolved)
        self.assertEqual(result["frame"], resolved)
        resolution = management.resolution()
        self.assertEqual(resolution["parameters"]["frame"], resolved)
        self.assertIn("枠", resolution["raw_instruction"])

    def test_the_canvas_sides_left_null_are_resolved(self):
        figure = np.zeros((80, 60, 4), np.float32)
        figure[10:70, 15:45] = (200, 60, 120, 255)
        layers = {"layer-figure": _png(figure, "RGBA"),
                  "layer-outline": _png(np.zeros_like(figure), "RGBA")}
        frame = {"box": [0.25, 0.25, 1.0, 1.0],
                 "sides": {"top": "out", "right": "cut", "bottom": "cut", "left": "cut"},
                 "line": {"color": "#885b80", "width": 5.0}, "paper": "#ffffff",
                 "canvas": [None, -0.25, None, None],
                 "expand": {"top": True, "right": True, "bottom": False, "left": True},
                 "strokes": []}
        management = ManagementFake({"dlv": (DELIVERY, layers)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            result = stage("dlv", services, backdrop="stripes", frame=frame)
        self.assertEqual(result["frame"]["canvas"], [9 / 60, -0.25, 66 / 60, 1.0])

    def test_a_frame_with_its_own_canvas_is_sent_as_given(self):
        frame = {"box": [0, 0, 1, 1], "sides": {}, "line": None, "paper": "#ffffff",
                 "canvas": [0, 0, 1, 1], "strokes": []}
        management = ManagementFake({"dlv": (DELIVERY, LAYERS)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            stage("dlv", services, backdrop="dots", frame=frame)
        self.assertEqual(json.loads(node(services, "YukariStage")["inputs"]["frame"]), frame)

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
              self.assertRaisesRegex(SystemExit, "saturation の出力")):
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
        self.assertEqual((inputs["scene"], inputs["tint_backdrop"]), ("", False))

    def test_a_scene_tints_a_pattern_backdrop_but_never_a_photo(self):
        management = ManagementFake({
            "stg": ({"kind": "stage", "base_generation": "dlv", "backdrop": "dots"}, LAYERS),
            "pic": ({"kind": "stage", "base_generation": "dlv", "photo": "p",
                     "place": [0.5, 1.0], "height": 0.8}, LAYERS)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            lighting("stg", services, direction=None, strength=0.5, scene="moon")
            inputs = node(services, "YukariRelight")["inputs"]
            self.assertEqual((inputs["scene"], inputs["tint_backdrop"]), ("moon", True))
            self.assertEqual(management.resolution()["parameters"]["scene"], "moon")
            lighting("pic", services, direction=None, strength=0.5, scene="sunset",
                     key_prefix="photo")
            self.assertFalse(node(services, "YukariRelight")["inputs"]["tint_backdrop"])

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


SATURATED = {"kind": "saturation", "base_generation": "dlv", "strength": 1.0}
OPAQUE_DELIVERY = {"kind": "deliver", "base_generation": "src", "transparent": False,
                   "stroke_light": "n", "outlines": [{"color": "#ffffff", "width": 1.0}]}


class SaturationTest(unittest.TestCase):
    def test_a_delivery_with_a_backdrop_is_saturated_and_keeps_all_layers(self):
        management = ManagementFake({"dlv": (OPAQUE_DELIVERY, LAYERS)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            result = saturation("dlv", services, strength=0.6)
        graph = services.comfyui.submitted[0]
        inputs = node(services, "YukariSaturation")["inputs"]
        self.assertEqual(inputs["strength"], 0.6)
        self.assertIn("backdrop", inputs)
        self.assertEqual(sorted(n["inputs"]["filename_prefix"] for n in graph.values()
                                if n["class_type"] == "SaveImage"),
                         ["sat-dlv-delivered", "sat-dlv-layer-backdrop",
                          "sat-dlv-layer-figure", "sat-dlv-layer-outline"])
        self.assertEqual(result, {"generation_ids": ["out-1"]})
        self.assertEqual(management.resolution()["parameters"], {
            "kind": "saturation", "base_generation": "dlv", "strength": 0.6})
        self.assertEqual(management.asset_roles(),
                         ["layer-figure", "layer-outline", "layer-backdrop"])

    def test_a_transparent_delivery_has_no_backdrop_to_pass_through(self):
        management = ManagementFake({"dlv": (DELIVERY, {
            "layer-figure": b"figure-png", "layer-outline": b"outline-png"})})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            saturation("dlv", services, strength=1.0)
        self.assertNotIn("backdrop", node(services, "YukariSaturation")["inputs"])
        self.assertEqual(management.asset_roles(), ["layer-figure", "layer-outline"])

    def test_only_a_delivery_with_layers_can_be_saturated(self):
        management = ManagementFake({
            "stg": ({"kind": "stage", "base_generation": "dlv"}, LAYERS),
            "lit": ({"kind": "lighting", "base_generation": "stg"}, LAYERS),
            "sat": (SATURATED, LAYERS),
            "pic": ({"kind": "photo", "base_generation": "dlv"}, LAYERS),
            "gen": ({"kind": "generate"}, {}),
            "bare": (DELIVERY, {"layer-figure": b"f"})})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            for source in ("stg", "lit", "sat", "pic", "gen"):
                with self.subTest(source=source), self.assertRaisesRegex(
                        SystemExit, "deliver の出力"):
                    saturation(source, services, strength=1.0)
            with self.assertRaisesRegex(SystemExit, "層が無い"):
                saturation("bare", services, strength=1.0)
        self.assertEqual(services.comfyui.submitted, [])


class SaturatedSourceTest(unittest.TestCase):
    def records(self):
        return ManagementFake({"sat": (SATURATED, LAYERS), "dlv": (DELIVERY, LAYERS),
                               "src": ({"kind": "generate"}, {})})

    def test_a_saturated_delivery_can_be_staged_with_its_delivery_shadow(self):
        management = self.records()
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            stage("sat", services, backdrop="stripes", inherit_shadow=True)
        inputs = node(services, "YukariStage")["inputs"]
        self.assertEqual((inputs["backdrop"], inputs["shadow_from"]), ("stripes", "nw"))
        self.assertEqual(json.loads(inputs["outlines"]), DELIVERY["outlines"])
        self.assertEqual(management.resolution()["parameters"]["base_generation"], "sat")

    def test_lighting_follows_the_saturated_deliverys_transparency(self):
        management = ManagementFake({
            "sat-opaque": ({**SATURATED, "base_generation": "opaque"}, LAYERS),
            "sat-clear": (SATURATED, LAYERS),
            "opaque": (OPAQUE_DELIVERY, LAYERS), "dlv": (DELIVERY, LAYERS)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            lighting("sat-opaque", services, direction=None, strength=0.5)
            self.assertIn("YukariRelight",
                          [n["class_type"] for n in services.comfyui.submitted[-1].values()])
            with self.assertRaisesRegex(SystemExit, "stage"):
                lighting("sat-clear", services, direction=None, strength=0.5)

    def test_dof_passes_through_the_saturation_to_the_delivery(self):
        management = self.records()
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(DofServices, management, directory)
            dof("sat", services, focus=(0.5, 0.5), f_number=2.8,
                scope={"figure": True, "outline": True, "backdrop": False})
        self.assertIn(b"image:src", [data for _, data in services.comfyui.uploaded])
        self.assertEqual(management.resolution()["parameters"]["base_generation"], "sat")

    def test_dof_asks_a_saturated_opaque_delivery_for_its_backdrop_layer(self):
        management = ManagementFake({
            "sat": ({**SATURATED, "base_generation": "opaque"}, {
                "layer-figure": b"f", "layer-outline": b"o"}),
            "opaque": (OPAQUE_DELIVERY, LAYERS), "src": ({"kind": "generate"}, {})})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(DofServices, management, directory)
            with self.assertRaisesRegex(SystemExit, "層が無い"):
                dof("sat", services, focus=(0.5, 0.5), f_number=2.8, scope={})


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
        self.assertEqual(inputs["focal_mm"], 85.0)
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
            self.assertNotIn("depth_place", node(services, "YukariDepthOfField")["inputs"])

    def test_a_framed_picture_lays_the_depth_where_the_delivery_stands(self):
        frame = {"box": [0, 0, 1, 1], "canvas": [-0.25, 0.0, 1.25, 2.0]}
        management = ManagementFake({
            "lit": ({"kind": "lighting", "base_generation": "stg"}, LAYERS),
            "stg": ({"kind": "stage", "base_generation": "dlv", "backdrop": "dots",
                     "frame": frame}, LAYERS),
            "dlv": (DELIVERY, LAYERS), "src": ({"kind": "generate"}, {})})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(DofServices, management, directory)
            dof("lit", services, focus=(0.5, 0.5), f_number=2.8,
                scope={"figure": True, "outline": True, "backdrop": True})
        place = json.loads(node(services, "YukariDepthOfField")["inputs"]["depth_place"])
        self.assertEqual(place, [0.25 / 1.5, 0.0, 1.25 / 1.5, 0.5])


if __name__ == "__main__":
    unittest.main()
