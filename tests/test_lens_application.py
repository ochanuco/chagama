"""Tests for the lens use case. All collaborators are fakes; the fake ComfyUI
answers `wait_for` with the SaveImage prefixes of the graph it was handed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chagama.application.lens import LensServices, lens
from chagama.application.picture_source import is_delivered


class ManagementFake:
    def __init__(self, *, request_kind="deliver", parameters=None, generations=None):
        self.calls = []
        self.context = {"id": "dlv-uuid",
                        "request": {"id": "source-request", "recipe": "yukari",
                                    "kind": request_kind,
                                    "parameters": ({"kind": "deliver",
                                                    "base_generation": "src"}
                                                   if parameters is None else parameters)},
                        "generations": generations or []}

    def request(self, method, path, payload=None, multipart=None):
        self.calls.append((method, path, payload, multipart))
        if path.endswith("/context"):
            return self.context
        if method == "POST" and path == "/api/v1/requests":
            return {"id": "request-id", "short_id": "req"}
        if method == "POST" and path.endswith("/jobs"):
            return {"id": "job-id"}
        if path.endswith("/generations"):
            return {"id": "lens-1", "short_id": "gen",
                    "canonical_url": "https://example/g/1"}
        return {}

    def fetch_generation_image(self, generation_id):
        return f"image:{generation_id}".encode()


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


def run(management=None, comfyui=None, **kwargs):
    management = management or ManagementFake()
    with tempfile.TemporaryDirectory() as directory:
        services = LensServices(
            management=management, comfyui=comfyui or ComfyFake(),
            git_metadata=lambda: {"commit": "c", "dirty": False},
            notifier=Notifier(), output_root=Path(directory), emit=lambda m: None)
        options = {"center": (0.5, 0.8), "strength": 0.7, **kwargs}
        result = lens("dlv", services, **options)
    return services, result


class LensSourceTest(unittest.TestCase):
    def test_a_picture_that_was_not_delivered_is_refused(self):
        for kind, parameters in (("generate", {}),
                                 ("redraw", {"kind": "redraw", "method": "canvas"})):
            with self.subTest(kind=kind), self.assertRaisesRegex(SystemExit, "納品の絵だけ"):
                run(ManagementFake(request_kind=kind, parameters=parameters))

    def test_dof_and_lens_pictures_are_accepted(self):
        for kind, parameters in (
                ("dof", {"kind": "dof", "base_generation": "d", "viewfinder": "off"}),
                ("dof", {"kind": "dof", "base_generation": "d", "viewfinder": "both"}),
                ("lens", {"kind": "lens", "base_generation": "d"})):
            with self.subTest(parameters=parameters):
                management = ManagementFake(
                    request_kind=kind, parameters=parameters,
                    generations=[{"id": "dlv-uuid", "comfy_output_index": 0},
                                 {"id": "other", "comfy_output_index": 1}])
                _, result = run(management)
                self.assertEqual(result["generation_ids"], ["lens-1"])

    def test_a_viewfinder_picture_is_refused(self):
        on = ManagementFake(request_kind="dof", parameters={
            "kind": "dof", "base_generation": "d", "viewfinder": "on"})
        both = ManagementFake(
            request_kind="dof",
            parameters={"kind": "dof", "base_generation": "d", "viewfinder": "both"},
            generations=[{"id": "other", "comfy_output_index": 0},
                         {"id": "dlv-uuid", "comfy_output_index": 1}])
        for management in (on, both):
            with self.assertRaisesRegex(SystemExit, "ビューファインダー"):
                run(management)

    def test_lens_outputs_count_as_delivered(self):
        self.assertTrue(is_delivered({"kind": "lens"}))
        self.assertTrue(is_delivered({"kind": "import", "parameters": {"kind": "lens"}}))


class LensUseCaseTest(unittest.TestCase):
    def test_the_picture_goes_through_the_lens_node(self):
        services, _ = run(strength=-0.4)
        [(_, picture)] = services.comfyui.uploaded
        self.assertEqual(picture, b"image:dlv")
        graph = services.comfyui.submitted[0]
        node = next(n for n in graph.values() if n["class_type"] == "YukariLens")
        self.assertEqual((node["inputs"]["center_x"], node["inputs"]["center_y"],
                          node["inputs"]["strength"]), (0.5, 0.8, -0.4))

    def test_the_request_records_its_kind_base_and_options(self):
        services, _ = run()
        resolution = next(call for call in services.management.calls
                          if call[0] == "POST" and call[1] == "/api/v1/requests")[2]
        self.assertEqual(resolution["parameters"], {
            "kind": "lens", "base_generation": "dlv", "center": [0.5, 0.8],
            "strength": 0.7, "shape": "fisheye"})
        self.assertEqual(resolution["references"][0]["source_generation_id"], "dlv")
        names = [call[3][0]["original_filename"] for call in services.management.calls
                 if call[1].endswith("/generations")]
        self.assertEqual(names, ["lens-dlv-lens_00001_.png"])

    def test_a_sphere_carries_its_radius_into_the_node_and_the_record(self):
        services, _ = run(shape="sphere", radius=0.35)
        node = next(n for n in services.comfyui.submitted[0].values()
                    if n["class_type"] == "YukariLens")
        self.assertEqual((node["inputs"]["shape"], node["inputs"]["radius"]),
                         ("sphere", 0.35))
        resolution = next(call for call in services.management.calls
                          if call[0] == "POST" and call[1] == "/api/v1/requests")[2]
        self.assertEqual((resolution["parameters"]["shape"],
                          resolution["parameters"]["radius"]), ("sphere", 0.35))

    def test_a_missing_lens_output_fails_loudly(self):
        class Silent(ComfyFake):
            def wait_for(self, prompt_id):
                return []

        with self.assertRaisesRegex(SystemExit, "lens output"):
            run(comfyui=Silent())


class DofAfterLensTest(unittest.TestCase):
    def test_dof_refuses_a_lens_picture_and_names_the_order(self):
        from chagama.application.dof import DofServices, dof
        management = ManagementFake(request_kind="lens",
                                    parameters={"kind": "lens", "base_generation": "d"})
        with tempfile.TemporaryDirectory() as directory:
            services = DofServices(
                management=management, comfyui=ComfyFake(),
                git_metadata=lambda: {"commit": "c", "dirty": False},
                notifier=Notifier(), output_root=Path(directory), emit=lambda m: None)
            with self.assertRaisesRegex(SystemExit, "dof を先に"):
                dof("dlv", services, focus=(0.5, 0.5), f_number=2.8,
                    scope={"figure": True, "outline": True, "backdrop": True})


if __name__ == "__main__":
    unittest.main()
