"""Tests for the dof use case. All collaborators are fakes; the fake ComfyUI
answers `wait_for` with the SaveImage prefixes of the graph it was handed."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from chagama.application.cut_assets import current_cut
from chagama.application.dof import DofServices, dof
from chagama.application.picture_source import is_delivered
from chagama.infrastructure.comfyui.refinement_graph import DEPTH_NODE

ALL = {"figure": True, "outline": True, "backdrop": True}
LAYERS = {"layer-figure": b"figure-png", "layer-outline": b"outline-png",
          "layer-backdrop": b"backdrop-png"}


def _cutout():
    output = io.BytesIO()
    Image.new("RGBA", (40, 30), (1, 2, 3, 255)).save(output, "PNG")
    return output.getvalue()


CUTOUT = _cutout()


class ManagementFake:
    def __init__(self, *, parameters=None, request_kind="deliver", assets=None,
                 source_assets=None):
        self.calls = []
        self.assets = {"dlv": dict(LAYERS if assets is None else assets),
                       "src": dict(source_assets or {})}
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
        if (method == "PUT" and path.endswith("/resolution")) or (
                method == "POST" and path == "/api/v1/requests"):
            return {"id": "request-id", "short_id": "req"}
        if method == "POST" and path.endswith("/jobs"):
            return {"id": "job-id"}
        if path.endswith("/generations"):
            index = sum(1 for call in self.calls if call[1].endswith("/generations"))
            return {"id": f"dof-{index}", "short_id": "gen",
                    "canonical_url": f"https://example/g/{index}"}
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
        services = DofServices(
            management=management, comfyui=ComfyFake(),
            git_metadata=lambda: {"commit": "c", "dirty": False},
            notifier=Notifier(), output_root=Path(directory), emit=lambda m: None,
            image_size=lambda data: (1448, 1808))
        options = {"focus": (0.5, 0.4), "f_number": 2.8, "scope": ALL,
                   "viewfinder": "off", **kwargs}
        result = dof("dlv", services, **options)
    return services, result


def classes(graph):
    return [node["class_type"] for node in graph.values()]


class DofSourceTest(unittest.TestCase):
    def refused(self, management, message):
        with self.assertRaisesRegex(SystemExit, message):
            run(management)

    def test_a_dof_output_is_refused(self):
        self.refused(ManagementFake(request_kind="dof",
                                    parameters={"kind": "dof", "base_generation": "dlv0"}),
                     "dof の出力")

    def test_a_picture_that_was_not_delivered_is_refused(self):
        self.refused(ManagementFake(request_kind="generate", parameters={}),
                     "切り抜いた描き直し")
        self.refused(ManagementFake(request_kind="redraw",
                                    parameters={"kind": "redraw", "method": "canvas"}),
                     "切り抜いた描き直し")

    def test_a_delivery_without_layers_is_refused(self):
        self.refused(ManagementFake(assets={}), "層が無い")
        self.refused(ManagementFake(assets={"layer-figure": b"f", "layer-outline": b"o"}),
                     "層が無い")
        self.refused(ManagementFake(request_kind="finalize", parameters={}), "層が無い")

    def test_a_transparent_delivery_needs_no_backdrop_layer(self):
        management = ManagementFake(
            parameters={"kind": "deliver", "base_generation": "src", "transparent": True},
            assets={"layer-figure": b"f", "layer-outline": b"o"})
        services, _ = run(management)
        [(_, blur)] = [(k, n) for k, n in services.comfyui.submitted[0].items()
                       if n["class_type"] == "YukariDepthOfField"]
        self.assertNotIn("backdrop", blur["inputs"])

    def test_a_cut_redraw_is_blurred_by_its_own_depth(self):
        cut = {"kind": "redraw", "method": "canvas", "base_generation": "src", "cut": True}
        stored = {"cutout": CUTOUT, "alpha": b"a", "depth": b"depth-png",
                  "cut": json.dumps(current_cut()).encode()}
        services, _ = run(ManagementFake(request_kind="redraw", parameters=cut,
                                         assets=stored))
        graph = services.comfyui.submitted[0]
        self.assertNotIn(DEPTH_NODE, classes(graph))
        self.assertIn(b"depth-png", [data for _, data in services.comfyui.uploaded])
        blur = next(n for n in graph.values() if n["class_type"] == "YukariDepthOfField")
        self.assertNotIn("backdrop", blur["inputs"])

    def test_a_stale_depth_of_a_cut_redraw_is_recomputed_and_attached_to_it(self):
        cut = {"kind": "redraw", "method": "canvas", "base_generation": "src", "cut": True}
        stale = {"depth": {"ckpt": "old"}}
        management = ManagementFake(request_kind="redraw", parameters=cut, assets={
            "cutout": CUTOUT, "alpha": b"a", "depth": b"old-depth",
            "cut": json.dumps(stale).encode()})
        services, _ = run(management)
        self.assertIn(DEPTH_NODE, classes(services.comfyui.submitted[0]))
        self.assertIn(b"image:dlv", [data for _, data in services.comfyui.uploaded])
        self.assertEqual([call[1] for call in management.asset_uploads()],
                         ["/api/v1/generations/dlv/assets"] * 2)

    def test_dof_outputs_count_as_delivered(self):
        self.assertTrue(is_delivered({"kind": "dof"}))
        self.assertTrue(is_delivered({"kind": "import", "parameters": {"kind": "dof"}}))


class DofUseCaseTest(unittest.TestCase):
    def test_a_missing_depth_is_built_from_the_source_and_attached_to_it(self):
        services, result = run()
        graph = services.comfyui.submitted[0]
        self.assertIn(DEPTH_NODE, classes(graph))
        [depth] = [node for node in graph.values() if node["class_type"] == DEPTH_NODE]
        self.assertEqual(depth["inputs"]["resolution"], 1448)
        self.assertTrue(any(name.startswith("dof-dlv-source-") and data == b"image:src"
                            for name, data in services.comfyui.uploaded))
        uploads = services.management.asset_uploads()
        self.assertEqual([(call[1], call[3][0]["role"]) for call in uploads], [
            ("/api/v1/generations/src/assets", "depth"),
            ("/api/v1/generations/src/assets", "cut")])
        self.assertEqual(json.loads(uploads[1][3][3]), {"depth": current_cut()["depth"]})
        self.assertEqual(result["generation_ids"], ["dof-1"])

    def test_a_current_depth_on_the_source_is_reused(self):
        management = ManagementFake(source_assets={
            "depth": b"depth-png", "cut": json.dumps(current_cut()).encode()})
        services, _ = run(management)
        self.assertNotIn(DEPTH_NODE, classes(services.comfyui.submitted[0]))
        self.assertIn(b"depth-png", [data for _, data in services.comfyui.uploaded])
        self.assertEqual(management.asset_uploads(), [])

    def test_a_depth_cut_at_a_fixed_resolution_is_recomputed(self):
        stale = {**current_cut(), "depth": {
            "ckpt": current_cut()["depth"]["ckpt"], "resolution": 1024}}
        management = ManagementFake(source_assets={
            "depth": b"old-depth", "cut": json.dumps(stale).encode()})
        services, _ = run(management)
        self.assertIn(DEPTH_NODE, classes(services.comfyui.submitted[0]))
        self.assertNotIn(b"old-depth", [data for _, data in services.comfyui.uploaded])

    def test_the_layers_are_uploaded_into_the_graph(self):
        services, _ = run()
        uploaded = [data for _, data in services.comfyui.uploaded]
        for data in LAYERS.values():
            self.assertIn(data, uploaded)

    def test_viewfinder_both_uploads_two_generations(self):
        services, result = run(viewfinder="both")
        self.assertEqual(result["generation_ids"], ["dof-1", "dof-2"])
        names = [call[3][0]["original_filename"] for call in services.management.calls
                 if call[1].endswith("/generations")]
        self.assertEqual(names, ["dof-dlv-dof_00001_.png", "dof-dlv-viewfinder_00001_.png"])

    def test_the_request_records_its_kind_base_and_options(self):
        services, _ = run(scope={"figure": True, "outline": False, "backdrop": False},
                          f_number=4.0)
        resolution = next(call for call in services.management.calls
                          if call[0] == "POST" and call[1] == "/api/v1/requests")[2]
        self.assertEqual(resolution["parameters"], {
            "kind": "dof", "base_generation": "dlv", "focus": [0.5, 0.4],
            "f_number": 4.0, "viewfinder": "off",
            "scope": {"figure": True, "outline": False, "backdrop": False}})
        self.assertEqual(resolution["references"][0]["source_generation_id"], "dlv")

    def test_a_missing_dof_output_fails_loudly(self):
        class Silent(ComfyFake):
            def wait_for(self, prompt_id):
                return []

        with tempfile.TemporaryDirectory() as directory:
            services = DofServices(
                management=ManagementFake(), comfyui=Silent(),
                git_metadata=lambda: {"commit": "c", "dirty": False},
                notifier=Notifier(), output_root=Path(directory), emit=lambda m: None,
            image_size=lambda data: (1448, 1808))
            with self.assertRaisesRegex(SystemExit, "dof"):
                dof("dlv", services, focus=(0.5, 0.5), f_number=2.8, scope=ALL)


if __name__ == "__main__":
    unittest.main()
