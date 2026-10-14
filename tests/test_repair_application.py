"""Adapter-level tests for the repair use case, all collaborators faked."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chagama.application.repair import RepairServices, repair

SOURCE_GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {
        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
        "latent_image": ["5", 0], "seed": 1, "steps": 20, "cfg": 5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 1}},
    "4": {"class_type": "DiffusersLoader", "inputs": {"model_path": "m"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {
        "clip": ["4", 1], "text": "masterpiece, (pantyhose feet:1.2)"}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {
        "clip": ["4", 1], "text": "worst quality"}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "x"}},
}


def _pose_outputs() -> dict:
    """A minimal DWPreprocessor outputs dict with one detected right foot."""
    body = [0.0, 0.0, 0.0] * 18
    body[9 * 3:9 * 3 + 3] = [100.0, 300.0, 1.0]  # Rknee
    body[10 * 3:10 * 3 + 3] = [100.0, 500.0, 1.0]  # Rank
    frame = {
        "people": [{
            "pose_keypoints_2d": body, "face_keypoints_2d": None,
            "hand_left_keypoints_2d": None, "hand_right_keypoints_2d": None,
        }],
        "canvas_width": 800, "canvas_height": 1000,
    }
    return {"2": {"openpose_json": [json.dumps([frame])]}}


class ManagementFake:
    def __init__(self, *, request_parameters=None, generations=None,
                request_recipe="yukari", request_kind=None):
        self.calls = []
        self.context = {
            "request": {"id": "source-request", "short_id": "srcreq",
                        "recipe": request_recipe, "kind": request_kind,
                        "parameters": request_parameters or {}},
            "generations": generations or [],
        }
        self._job_counter = 0
        self.job_graph = None

    def request(self, method, path, payload=None, multipart=None):
        self.calls.append((method, path, payload, multipart))
        if path.endswith("/context"):
            return self.context
        if method == "GET" and path.startswith("/api/v1/generations/") \
                and path.count("/") == 4:
            return {"id": path.rsplit("/", 1)[1], "comfy_job": self.job_graph}
        is_resolution = method == "PUT" and path.endswith("/resolution")
        if is_resolution or (method == "POST" and path == "/api/v1/requests"):
            return {"id": "repair-request-id", "short_id": "reprequest"}
        if method == "POST" and path.endswith("/jobs"):
            self._job_counter += 1
            return {"id": f"job-{self._job_counter}"}
        if path.endswith("/generations"):
            seed = multipart[0]["seed"]
            index = multipart[0]["comfy_output_index"]
            gen_id = f"gen-{seed}-{index}"
            return {"id": gen_id, "short_id": gen_id,
                    "canonical_url": f"https://example/{gen_id}"}
        if path.endswith("/assets"):
            return {}
        return {}

    def fetch_generation_image(self, generation_id):
        return b"picked"


class ComfyFake:
    def __init__(self, *, outputs_per_job=None, pose_outputs=None):
        self.uploaded = []
        self.submitted = []
        self.outputs_per_job = outputs_per_job
        self.pose_outputs = pose_outputs if pose_outputs is not None else _pose_outputs()

    def upload_image(self, name, data):
        self.uploaded.append((name, data))
        return f"uploaded-{name}"

    def submit(self, graph, purpose="render"):
        self.submitted.append(graph)
        return f"prompt-{len(self.submitted)}"

    def wait_for_outputs(self, prompt_id):
        return self.pose_outputs

    def wait_for(self, prompt_id):
        if self.outputs_per_job is not None:
            return self.outputs_per_job
        return [{"filename": f"{prompt_id}-out.png"}]

    def fetch(self, image):
        return b"raw-bytes"


class RecordingNotifier:
    def __init__(self):
        self.calls = []

    def send(self, *args):
        self.calls.append(args)


def base_services(directory, **overrides):
    kwargs = dict(
        management=ManagementFake(),
        comfyui=ComfyFake(),
        graph_from_png=lambda data: SOURCE_GRAPH,
        image_size=lambda data: (800, 1000),
        git_metadata=lambda: {"commit": "commit", "dirty": False},
        notifier=RecordingNotifier(),
        output_root=Path(directory),
        emit=lambda message: None,
        pose_graph=lambda image_name, prefix="pose": {"pose-graph-for": image_name},
        repair_graph=lambda source, **kw: {"repair-graph": kw.get("seed")},
    )
    kwargs.update(overrides)
    return RepairServices(**kwargs)


def resolution_call(services):
    return next(
        call for call in services.management.calls
        if call[1] == "/api/v1/requests"
        or call[1].endswith("/resolution"))


class RepairApplicationTest(unittest.TestCase):
    def test_delivered_sources_are_refused(self):
        cases = [
            dict(request_kind="deliver"),
            dict(request_kind="finalize"),
            dict(request_parameters={"kind": "deliver"}),
            dict(request_parameters={"kind": "hires-chain"}),
            dict(request_parameters={"kind": "repair", "deliver_only": True}),
        ]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                services = base_services(
                    directory, management=ManagementFake(**case))
                with self.assertRaisesRegex(SystemExit, "納品済み"):
                    repair("gen-1", services,
                           parts=[], regions=[[0.0, 0.0, 0.1, 0.1]])
                self.assertEqual(services.comfyui.submitted, [])

    def test_a_picture_request_uses_the_given_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            fetched = []
            management = ManagementFake(request_parameters={"kind": "generate"})
            management.fetch_generation_image = lambda gid: (
                fetched.append(gid) or b"picked")
            services = base_services(directory, management=management)
            repair("gen-1", services, parts=[], regions=[[0.0, 0.0, 0.1, 0.1]])
            self.assertEqual(fetched, ["gen-1"])

    def test_a_given_context_skips_its_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=[], regions=[[0.0, 0.0, 0.1, 0.1]],
                  context=services.management.context)
            fetch_calls = [
                call for call in services.management.calls
                if call[0] == "GET" and call[1].endswith("/context")]
            self.assertEqual(fetch_calls, [])

    def test_no_regions_found_raises_system_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            with self.assertRaises(SystemExit):
                repair("gen-1", services, parts=[], regions=[])

    def test_parts_empty_skips_the_pose_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            comfy = ComfyFake()
            services = base_services(directory, comfyui=comfy)
            repair("gen-1", services, parts=[], regions=[[0.0, 0.0, 0.1, 0.1]])
            pose_submissions = [g for g in comfy.submitted if "pose-graph-for" in g]
            self.assertEqual(pose_submissions, [])

    def test_parts_present_submits_the_pose_graph_first(self):
        with tempfile.TemporaryDirectory() as directory:
            comfy = ComfyFake()
            services = base_services(directory, comfyui=comfy)
            repair("gen-1", services, parts=["feet"])
            self.assertIn("pose-graph-for", comfy.submitted[0])

    def test_one_job_per_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            comfy = ComfyFake()
            services = base_services(directory, comfyui=comfy)
            result = repair("gen-1", services, parts=["feet"], seeds=[10, 20, 30])
            job_posts = [call for call in services.management.calls
                        if call[0] == "POST" and call[1].endswith("/jobs")]
            self.assertEqual(len(job_posts), 3)
            self.assertEqual(len(result["generation_ids"]), 3)

    def test_raw_output_uploaded_as_a_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            comfy = ComfyFake(outputs_per_job=[{"filename": "rep-out.png"}])
            services = base_services(directory, comfyui=comfy)
            repair("gen-1", services, parts=["feet"], seeds=[7])
            generation_posts = [call for call in services.management.calls
                                if call[0] == "POST" and call[1].endswith("/generations")]
            self.assertEqual(len(generation_posts), 1)
            metadata, field, filename, data, content_type = generation_posts[0][3]
            self.assertEqual(filename, "rep-out.png")
            self.assertEqual(data, b"raw-bytes")

    def test_only_the_picture_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            result = repair("gen-1", services, parts=["feet"], seeds=[7])
            generation_posts = [call for call in services.management.calls
                                if call[0] == "POST" and call[1].endswith("/generations")]
            self.assertEqual(len(generation_posts), 1)
            self.assertEqual(len(result["generation_ids"]), 1)
            roles = [call[3][0].get("role") for call in services.management.calls
                     if call[0] == "POST" and call[1].endswith("/assets")]
            self.assertEqual(roles, ["repair-mask"])

    def test_region_mask_uploaded_with_repair_mask_role(self):
        with tempfile.TemporaryDirectory() as directory:
            comfy = ComfyFake()
            services = base_services(directory, comfyui=comfy)
            repair("gen-1", services, parts=["feet"], seeds=[7])
            asset_posts = [call for call in services.management.calls
                          if call[0] == "POST" and call[1].endswith("/assets")]
            repair_mask_posts = [
                call for call in asset_posts
                if call[3][0].get("role") == "repair-mask"]
            self.assertEqual(len(repair_mask_posts), 1)

    def test_no_raw_output_raises_system_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            comfy = ComfyFake(outputs_per_job=[])
            services = base_services(directory, comfyui=comfy)
            with self.assertRaises(SystemExit):
                repair("gen-1", services, parts=["feet"], seeds=[7])

    def test_request_parameters_record_the_request(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], regions=[[0.1, 0.2, 0.3, 0.4]],
                  denoise=0.7, seeds=[5], size=768, pad=1.5)
            parameters = resolution_call(services)[2]["parameters"]
            self.assertEqual(parameters["kind"], "repair")
            self.assertEqual(parameters["base_generation"], "gen-1")
            self.assertEqual(parameters["requested_generation"], "gen-1")
            self.assertEqual(parameters["parts"], ["feet"])
            self.assertEqual(parameters["regions"], [[0.1, 0.2, 0.3, 0.4]])
            self.assertEqual(parameters["denoise"], 0.7)
            self.assertEqual(parameters["size"], 768)
            self.assertEqual(parameters["pad"], 1.5)
            self.assertEqual(parameters["seeds"], [5])
            self.assertIn("mask_bbox", parameters)

    def test_key_prefix_derives_request_and_job_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], seeds=[1], key_prefix="request:r1")
            posts = {call[1]: call[2] for call in services.management.calls
                     if call[0] == "POST" and call[2]}
            self.assertEqual(posts["/api/v1/requests"]["idempotency_key"], "request:r1")
            self.assertEqual(
                posts["/api/v1/requests/repair-request-id/jobs"]["idempotency_key"],
                "request:r1:job:0")
            generation_call = next(
                call for call in services.management.calls
                if call[0] == "POST" and call[1].endswith("/generations"))
            self.assertEqual(
                generation_call[3][0]["idempotency_key"],
                "request:r1:job:0:gen:0")
            asset_call = next(
                call for call in services.management.calls
                if call[0] == "POST" and call[1].endswith("/assets"))
            self.assertEqual(
                asset_call[3][0]["idempotency_key"],
                "request:r1:job:0:asset:repair-mask")

    def test_returns_generation_ids_only(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            result = repair("gen-1", services, parts=["feet"], seeds=[1, 2])
            self.assertNotIn("batch_id", result)
            self.assertEqual(len(result["generation_ids"]), 2)

    def test_worker_request_reports_resolution_and_jobs_source_the_redraw(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(
                request_parameters={"kind": "redraw"},
                generations=[{"id": "raw-gen", "short_id": "rawshort",
                              "image_width": 1280, "image_height": 2560}])
            services = base_services(directory, management=management)
            repair("raw-gen", services, parts=["feet"], seeds=[1, 2],
                   key_prefix="request:r1", request_id="r1")
            calls = management.calls
            self.assertFalse(any(call[1] == "/api/v1/requests" for call in calls))
            resolution = next(
                call for call in calls if call[1] == "/api/v1/requests/r1/resolution")
            self.assertEqual(resolution[0], "PUT")
            self.assertEqual(
                resolution[2]["references"][0]["source_generation_id"], "raw-gen")
            jobs = [call for call in calls if call[1].endswith("/jobs")]
            self.assertEqual(len(jobs), 2)
            for job in jobs:
                self.assertEqual(job[1], "/api/v1/requests/repair-request-id/jobs")
                self.assertEqual(job[2]["source_generation_id"], "raw-gen")

    def test_standalone_run_imports_and_jobs_carry_no_source(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], seeds=[1])
            calls = services.management.calls
            create = next(call for call in calls if call[1] == "/api/v1/requests")
            self.assertEqual(create[2]["kind"], "import")
            self.assertEqual(create[2]["status"], "done")
            job_call = next(call for call in calls if call[1].endswith("/jobs"))
            self.assertNotIn("source_generation_id", job_call[2])

    def test_lora_reaches_the_graph_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            seen = {}

            def fake_repair_graph(source, **kwargs):
                seen["loras"] = kwargs.get("loras")
                return {"repair-graph": kwargs.get("seed")}

            services = base_services(directory, repair_graph=fake_repair_graph)
            repair("gen-1", services, parts=["feet"], seeds=[1], lora=0.8)
            self.assertEqual(seen["loras"], (("feet-xl-ill.safetensors", 0.8),))

    def test_no_lora_reaches_the_graph_builder_as_an_empty_tuple(self):
        with tempfile.TemporaryDirectory() as directory:
            seen = {}

            def fake_repair_graph(source, **kwargs):
                seen["loras"] = kwargs.get("loras")
                return {"repair-graph": kwargs.get("seed")}

            services = base_services(directory, repair_graph=fake_repair_graph)
            repair("gen-1", services, parts=["feet"], seeds=[1])
            self.assertEqual(seen["loras"], ())

    def test_lora_is_recorded_in_request_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], seeds=[1], lora=0.8)
            parameters = resolution_call(services)[2]["parameters"]
            self.assertEqual(parameters["lora"], 0.8)

    def test_lora_defaults_to_none_in_request_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], seeds=[1])
            parameters = resolution_call(services)[2]["parameters"]
            self.assertIsNone(parameters["lora"])

    def test_model_reaches_the_graph_builder_as_a_model_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            seen = {}

            def fake_repair_graph(source, **kwargs):
                seen["model_hooks"] = kwargs.get("model_hooks")
                return {"repair-graph": kwargs.get("seed")}

            services = base_services(directory, repair_graph=fake_repair_graph)
            repair("gen-1", services, parts=["feet"], seeds=[1], model="anima")
            self.assertEqual(len(seen["model_hooks"]), 1)

    def test_no_model_reaches_the_graph_builder_as_an_empty_tuple(self):
        with tempfile.TemporaryDirectory() as directory:
            seen = {}

            def fake_repair_graph(source, **kwargs):
                seen["model_hooks"] = kwargs.get("model_hooks")
                return {"repair-graph": kwargs.get("seed")}

            services = base_services(directory, repair_graph=fake_repair_graph)
            repair("gen-1", services, parts=["feet"], seeds=[1])
            self.assertEqual(seen["model_hooks"], ())

    def test_model_skips_the_part_lora_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            seen = {}

            def fake_repair_graph(source, **kwargs):
                seen["loras"] = kwargs.get("loras")
                return {"repair-graph": kwargs.get("seed")}

            services = base_services(directory, repair_graph=fake_repair_graph)
            repair("gen-1", services, parts=["feet"], seeds=[1], lora=0.8, model="anima")
            self.assertEqual(seen["loras"], ())

    def test_model_is_recorded_in_request_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], seeds=[1], model="anima")
            parameters = resolution_call(services)[2]["parameters"]
            self.assertEqual(parameters["model"], "anima")

    def test_model_defaults_to_none_in_request_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], seeds=[1])
            parameters = resolution_call(services)[2]["parameters"]
            self.assertIsNone(parameters["model"])

    def test_control_reaches_the_graph_builder_as_a_conditioning_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            seen = {}

            def fake_repair_graph(source, **kwargs):
                seen["conditioning_hooks"] = kwargs.get("conditioning_hooks")
                return {"repair-graph": kwargs.get("seed")}

            services = base_services(directory, repair_graph=fake_repair_graph)
            repair("gen-1", services, parts=["feet"], seeds=[1],
                  control="lineart", control_strength=0.5)
            self.assertEqual(len(seen["conditioning_hooks"]), 1)

    def test_no_control_reaches_the_graph_builder_as_an_empty_list(self):
        with tempfile.TemporaryDirectory() as directory:
            seen = {}

            def fake_repair_graph(source, **kwargs):
                seen["conditioning_hooks"] = kwargs.get("conditioning_hooks")
                return {"repair-graph": kwargs.get("seed")}

            services = base_services(directory, repair_graph=fake_repair_graph)
            repair("gen-1", services, parts=["feet"], seeds=[1])
            self.assertEqual(seen["conditioning_hooks"], [])

    def test_control_stages_the_reference_hint_image(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], seeds=[1], control="lineart")
            names = [name for name, _data in services.comfyui.uploaded]
            self.assertTrue(any(name.endswith("-control.png") for name in names))

    def test_no_control_stages_no_reference_hint_image(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], seeds=[1])
            names = [name for name, _data in services.comfyui.uploaded]
            self.assertFalse(any(name.endswith("-control.png") for name in names))

    def test_control_is_recorded_in_request_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], seeds=[1],
                  control="lineart", control_strength=0.5)
            parameters = resolution_call(services)[2]["parameters"]
            self.assertEqual(parameters["control"], "lineart")
            self.assertEqual(parameters["control_strength"], 0.5)

    def test_control_defaults_to_none_in_request_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], seeds=[1])
            parameters = resolution_call(services)[2]["parameters"]
            self.assertIsNone(parameters["control"])
            self.assertIsNone(parameters["control_strength"])

    def test_notifier_is_sent_once_after_the_request_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], seeds=[1, 2])
            self.assertEqual(len(services.notifier.calls), 1)

    def test_the_request_is_never_patched(self):
        with tempfile.TemporaryDirectory() as directory:
            services = base_services(directory)
            repair("gen-1", services, parts=["feet"], seeds=[1, 2])
            self.assertFalse(any(
                call[0] == "PATCH" and call[1].startswith("/api/v1/requests/")
                for call in services.management.calls))


if __name__ == "__main__":
    unittest.main()


class SourceGraphFromJobTest(unittest.TestCase):
    def test_the_job_graph_wins_over_the_png_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake()
            job_graph = dict(SOURCE_GRAPH)
            management.job_graph = {"graph": job_graph}
            seen = {}

            def fake_repair_graph(source, **kwargs):
                seen["source"] = source
                return {"repair-graph": kwargs.get("seed")}

            services = base_services(
                directory, management=management,
                graph_from_png=lambda data: {"png": "stale"},
                repair_graph=fake_repair_graph)
            repair("gen-1", services, parts=(),
                   regions=((0.1, 0.1, 0.5, 0.5),), seeds=(1,))
            self.assertIs(seen["source"], job_graph)

    def test_a_record_without_a_job_graph_falls_back_to_the_png(self):
        with tempfile.TemporaryDirectory() as directory:
            seen = {}

            def fake_repair_graph(source, **kwargs):
                seen["source"] = source
                return {"repair-graph": kwargs.get("seed")}

            services = base_services(directory, repair_graph=fake_repair_graph)
            repair("gen-1", services, parts=(),
                   regions=((0.1, 0.1, 0.5, 0.5),), seeds=(1,))
            self.assertIs(seen["source"], SOURCE_GRAPH)
