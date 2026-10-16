"""Tests for the stage and lighting use cases and the dof of what they make.
All collaborators are fakes; the fake ComfyUI answers `wait_for` with the
SaveImage prefixes of the graph it was handed."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from chagama.application.cut_assets import current_cut
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
            with self.assertRaisesRegex(SystemExit, "どちらかを指定"):
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
            with self.assertRaisesRegex(SystemExit, "どちらかを指定"):
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


def png_bytes(size=(40, 30), colour=(10, 20, 30, 255)):
    output = io.BytesIO()
    Image.new("RGBA", size, colour).save(output, "PNG")
    return output.getvalue()


CUTOUT = png_bytes()
CUT_REDRAW = {"kind": "redraw", "method": "canvas", "base_generation": "src", "cut": True}
CUT_ASSETS = {"cutout": CUTOUT, "alpha": b"alpha-png", "depth": b"depth-png",
              "cut": json.dumps(current_cut()).encode()}
UNCUT_REDRAW = {"kind": "redraw", "method": "canvas", "base_generation": "src"}


def uploaded_data(services):
    return [data for _, data in services.comfyui.uploaded]


class CutRedrawSourceTest(unittest.TestCase):
    def records(self):
        return ManagementFake({
            "cr": (CUT_REDRAW, CUT_ASSETS), "plain": (UNCUT_REDRAW, {"depth": b"d"}),
            "nocut": (CUT_REDRAW, {"alpha": b"a"}), "src": ({"kind": "generate"}, {})})

    def test_a_cut_redraw_is_a_transparent_delivery_with_no_outlines(self):
        management = self.records()
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            saturation("cr", services, strength=0.5)
        figure, outline = uploaded_data(services)
        self.assertEqual(figure, CUTOUT)
        blank = Image.open(io.BytesIO(outline))
        self.assertEqual(blank.size, (40, 30))
        self.assertEqual(blank.getextrema()[3], (0, 0))
        self.assertNotIn("backdrop", node(services, "YukariSaturation")["inputs"])
        self.assertEqual(management.asset_roles(), ["layer-figure", "layer-outline"])
        self.assertEqual(management.resolution()["parameters"], {
            "kind": "saturation", "base_generation": "cr", "strength": 0.5})

    def test_a_redraw_that_was_not_cut_is_refused(self):
        management = self.records()
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            with self.assertRaisesRegex(SystemExit, "切り抜いた描き直し"):
                saturation("plain", services, strength=0.5)
            with self.assertRaisesRegex(SystemExit, "切り抜いた描き直し"):
                stage("plain", services, backdrop="dots")
            with self.assertRaisesRegex(SystemExit, "切り抜いた描き直し"):
                lighting("plain", services, direction="nw", strength=0.5)
            with self.assertRaisesRegex(SystemExit, "層が無い"):
                stage("nocut", services, backdrop="dots")
        self.assertEqual(services.comfyui.submitted, [])

    def test_a_pattern_throws_no_shadow_unless_asked_to(self):
        management = self.records()
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            stage("cr", services, backdrop="dots", inherit_shadow=True)
            inputs = node(services, "YukariStage")["inputs"]
            self.assertEqual((inputs["shadow_from"], inputs["outlines"]), ("", "[]"))
            self.assertEqual(management.resolution()["parameters"]["shadow_from"], None)
            stage("cr", services, backdrop="dots", shadow_from="nw", key_prefix="shadowed")
            self.assertEqual(node(services, "YukariStage")["inputs"]["shadow_from"], "nw")

    def test_a_frame_and_a_photo_take_a_cut_redraw(self):
        management = self.records()
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            stage("cr", services, backdrop="dots", frame={
                "box": [0, 0, 1, 1], "canvas": [0.0, 0.0, 1.0, 1.0]}, key_prefix="framed")
            self.assertEqual(node(services, "YukariStage")["inputs"]["mode"], "pattern")
            stage("cr", services, photo_id="pic", key_prefix="photoed")
            self.assertEqual(node(services, "YukariStage")["inputs"]["mode"], "photo")

    def test_a_saturation_of_a_cut_redraw_stages_with_no_outlines(self):
        management = ManagementFake({
            "sat": ({"kind": "saturation", "base_generation": "cr", "strength": 1.0}, LAYERS),
            "cr": (CUT_REDRAW, CUT_ASSETS)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            stage("sat", services, backdrop="dots", inherit_shadow=True)
        inputs = node(services, "YukariStage")["inputs"]
        self.assertEqual((inputs["shadow_from"], inputs["outlines"]), ("", "[]"))


class TransparentLightingTest(unittest.TestCase):
    def test_a_transparent_delivery_is_lit_by_direction_without_a_backdrop(self):
        management = ManagementFake({"dlv": (DELIVERY, {
            "layer-figure": b"figure-png", "layer-outline": b"outline-png"})})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            lighting("dlv", services, direction="ne", strength=0.5)
        inputs = node(services, "YukariRelight")["inputs"]
        self.assertNotIn("backdrop", inputs)
        self.assertEqual((inputs["direction"], inputs["strength"]), ("ne", 0.5))
        self.assertEqual(management.asset_roles(), ["layer-figure", "layer-outline"])
        self.assertEqual(management.resolution()["parameters"], {
            "kind": "lighting", "base_generation": "dlv", "direction": "ne",
            "strength": 0.5, "scene": None})
        saved = [n["inputs"]["filename_prefix"] for n in
                 services.comfyui.submitted[0].values() if n["class_type"] == "SaveImage"]
        self.assertEqual(sorted(saved), ["light-dlv-layer-figure", "light-dlv-layer-outline",
                                         "light-dlv-lit"])

    def test_a_scene_alone_lights_a_cut_redraw(self):
        management = ManagementFake({"cr": (CUT_REDRAW, CUT_ASSETS)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            lighting("cr", services, direction=None, strength=0.5, scene="moon")
        self.assertNotIn("backdrop", node(services, "YukariRelight")["inputs"])
        self.assertEqual(management.asset_roles(), ["layer-figure", "layer-outline"])

    def test_nothing_to_shade_with_is_refused(self):
        management = ManagementFake({"cr": (CUT_REDRAW, CUT_ASSETS)})
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(StageServices, management, directory)
            with self.assertRaisesRegex(SystemExit, "direction か scene"):
                lighting("cr", services, direction=None, strength=0.5)
        self.assertEqual(services.comfyui.submitted, [])


class CutRedrawDofTest(unittest.TestCase):
    def blur(self, generation_id, management):
        with tempfile.TemporaryDirectory() as directory:
            services = services_for(DofServices, management, directory)
            dof(generation_id, services, focus=(0.5, 0.5), f_number=2.8,
                scope={"figure": True, "outline": True, "backdrop": False})
        return services

    def test_a_cut_redraw_blurs_by_its_own_stored_depth(self):
        management = ManagementFake({"cr": (CUT_REDRAW, CUT_ASSETS)})
        services = self.blur("cr", management)
        self.assertIn(b"depth-png", uploaded_data(services))
        self.assertEqual(management.asset_roles("cr"), [])
        self.assertNotIn("backdrop", node(services, "YukariDepthOfField")["inputs"])

    def test_a_stale_depth_is_recomputed_from_the_redraw_and_attached_to_it(self):
        stale = {**current_cut(), "depth": {"ckpt": "old"}}
        management = ManagementFake({"cr": (CUT_REDRAW, {
            **CUT_ASSETS, "cut": json.dumps(stale).encode()})})
        services = self.blur("cr", management)
        self.assertNotIn(b"depth-png", uploaded_data(services))
        self.assertIn(b"image:cr", uploaded_data(services))
        self.assertEqual(management.asset_roles("cr"), ["depth", "cut"])

    def test_pictures_built_on_a_cut_redraw_take_the_redraws_depth(self):
        management = ManagementFake({
            "cr": (CUT_REDRAW, CUT_ASSETS),
            "sat": ({"kind": "saturation", "base_generation": "cr", "strength": 1.0},
                    {"layer-figure": b"f", "layer-outline": b"o"}),
            "stg": ({"kind": "stage", "base_generation": "sat", "backdrop": "dots"}, LAYERS),
            "lit": ({"kind": "lighting", "base_generation": "sat"},
                    {"layer-figure": b"f", "layer-outline": b"o"})})
        for source, backdrop in (("sat", False), ("stg", True), ("lit", False)):
            with self.subTest(source=source):
                services = self.blur(source, management)
                self.assertIn(b"depth-png", uploaded_data(services))
                self.assertEqual("backdrop" in node(services, "YukariDepthOfField")["inputs"],
                                 backdrop)

    def test_a_redraw_that_was_not_cut_is_refused(self):
        management = ManagementFake({"plain": (UNCUT_REDRAW, {})})
        with self.assertRaisesRegex(SystemExit, "切り抜いた描き直し"):
            self.blur("plain", management)

    def test_a_lit_transparent_delivery_needs_no_backdrop_layer(self):
        management = ManagementFake({
            "lit": ({"kind": "lighting", "base_generation": "dlv"},
                    {"layer-figure": b"f", "layer-outline": b"o"}),
            "dlv": (DELIVERY, {}), "src": ({"kind": "generate"}, {})})
        services = self.blur("lit", management)
        self.assertNotIn("backdrop", node(services, "YukariDepthOfField")["inputs"])


if __name__ == "__main__":
    unittest.main()
