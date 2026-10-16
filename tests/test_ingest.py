"""Unit tests for the shared classify/record/upload/attach ingest helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chagama.application.ingest import (
    attach_asset,
    classify_cut_outputs,
    classify_deliver_outputs,
    classify_dof_outputs,
    classify_redraw_outputs,
    import_images,
    open_request,
    record_job,
)


class ManagementFake:
    def __init__(self):
        self.calls = []
        self._job_counter = 0

    def request(self, method, path, payload=None, multipart=None):
        self.calls.append((method, path, payload, multipart))
        if method == "POST" and path.endswith("/jobs"):
            self._job_counter += 1
            return {"id": f"job-{self._job_counter}"}
        if method == "POST" and path == "/api/v1/requests":
            return {"id": "import-1", "short_id": "imp"}
        if path.endswith("/generations"):
            return {"id": f"gen-{len(self.calls)}",
                    "canonical_url": "https://example/g"}
        return {}


class ClassifyOutputsTests(unittest.TestCase):
    def test_deliver_outputs_split_by_filename(self):
        outputs = [
            {"filename": "a-alpha.png"},
            {"filename": "a-delivered.png"},
            {"filename": "a-layer-figure.png"},
            {"filename": "a-layer-outline.png"},
            {"filename": "a-layer-backdrop.png"},
        ]
        found = classify_deliver_outputs(outputs)
        self.assertEqual(found["alpha"], [outputs[0]])
        self.assertEqual(found["delivered"], [outputs[1]])
        self.assertEqual(found["layer-figure"], [outputs[2]])
        self.assertEqual(found["layer-outline"], [outputs[3]])
        self.assertEqual(found["layer-backdrop"], [outputs[4]])

    def test_dof_outputs_split_by_filename(self):
        outputs = [{"filename": "dof-g-depth_00001_.png"},
                   {"filename": "dof-g-dof_00001_.png"},
                   {"filename": "dof-g-viewfinder_00001_.png"}]
        found = classify_dof_outputs(outputs)
        self.assertEqual(found["depth"], [outputs[0]])
        self.assertEqual(found["dof"], [outputs[1]])
        self.assertEqual(found["viewfinder"], [outputs[2]])

    def test_redraw_picture_is_what_the_cuts_are_not(self):
        outputs = [{"filename": "a.png"}, {"filename": "a-alpha.png"},
                   {"filename": "a-depth.png"}]
        found = classify_redraw_outputs(outputs)
        self.assertEqual(found["picture"], [outputs[0]])
        self.assertEqual(found["alpha"], [outputs[1]])
        self.assertEqual(found["depth"], [outputs[2]])

    def test_a_cut_is_its_alpha_depth_and_foreground(self):
        outputs = [{"filename": "a-out-alpha_00001_.png"}, {"filename": "a-out-depth_00001_.png"},
                   {"filename": "a-out-foreground_00001_.png"}]
        found = classify_cut_outputs(outputs)
        self.assertEqual([found[role] for role in ("alpha", "depth", "foreground")],
                         [[outputs[0]], [outputs[1]], [outputs[2]]])


class RecordJobTests(unittest.TestCase):
    def test_call_order_and_payloads(self):
        management = ManagementFake()
        job = record_job(management, "request-1", key_prefix=None, index=0,
                         seed=7, prompt_id="prompt-1", graph={"n": 1})
        self.assertEqual(job, {"id": "job-1"})
        self.assertEqual(management.calls, [
            ("POST", "/api/v1/requests/request-1/jobs",
             {"idempotency_key": management.calls[0][2]["idempotency_key"],
              "seed": 7, "index": 0}, None),
            ("PATCH", "/api/v1/jobs/job-1",
             {"status": "queued", "comfy_prompt_id": "prompt-1",
              "graph": {"n": 1}}, None),
            ("PATCH", "/api/v1/jobs/job-1", {"status": "completed"}, None),
        ])

    def test_idempotency_key_with_prefix(self):
        management = ManagementFake()
        record_job(management, "request-1", key_prefix="run-x", index=2,
                  seed=1, prompt_id="p", graph={})
        self.assertEqual(management.calls[0][2]["idempotency_key"], "run-x:job:2")

    def test_idempotency_key_without_prefix_is_a_uuid(self):
        management = ManagementFake()
        record_job(management, "request-1", key_prefix=None, index=0,
                  seed=1, prompt_id="p", graph={})
        key = management.calls[0][2]["idempotency_key"]
        self.assertNotIn(":job:", key)
        self.assertEqual(len(key), 36)


    def test_source_generation_id_is_sent_only_when_given(self):
        management = ManagementFake()
        record_job(management, "request-1", key_prefix="run-x", index=0,
                   seed=1, prompt_id="p", graph={}, source_generation_id="src-1")
        self.assertEqual(management.calls[0][2]["source_generation_id"], "src-1")


class OpenRequestTests(unittest.TestCase):
    RESOLUTION = {"recipe": "yukari", "parameters": {}, "git_commit": "c",
                  "git_dirty": False}

    def test_a_claimed_request_gets_its_resolution_put(self):
        management = ManagementFake()
        open_request(management, request_id="req-9", idempotency_key="k",
                     resolution=self.RESOLUTION)
        self.assertEqual(management.calls, [
            ("PUT", "/api/v1/requests/req-9/resolution", self.RESOLUTION, None)])

    def test_without_a_claimed_request_an_import_request_is_created(self):
        management = ManagementFake()
        created = open_request(management, request_id=None, idempotency_key="k",
                               resolution=self.RESOLUTION)
        self.assertEqual(created["id"], "import-1")
        method, path, payload, _ = management.calls[0]
        self.assertEqual((method, path), ("POST", "/api/v1/requests"))
        self.assertEqual(payload["kind"], "import")
        self.assertEqual(payload["status"], "done")
        self.assertEqual(payload["idempotency_key"], "k")
        self.assertEqual(payload["created_by"], "system")
        self.assertEqual(payload["recipe"], "yukari")


    def test_an_import_request_carries_its_run_id(self):
        management = ManagementFake()
        open_request(management, request_id=None, idempotency_key="k",
                     resolution=self.RESOLUTION, run_id="run-1")
        self.assertEqual(management.calls[0][2]["run_id"], "run-1")

    def test_a_claimed_request_never_sends_run_id(self):
        management = ManagementFake()
        open_request(management, request_id="req-9", idempotency_key="k",
                     resolution=self.RESOLUTION, run_id="run-1")
        self.assertNotIn("run_id", management.calls[0][2])


class ImportImagesTests(unittest.TestCase):
    def test_registers_each_image_under_one_import_request_and_job(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.png"
            second = Path(directory) / "b.png"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            management = ManagementFake()
            result = import_images(
                management, lambda message: None, images=[first, second],
                resolution={"recipe": None, "parameters": {}},
                idempotency_key="import:k", seed=3)
        self.assertEqual(management.calls[0][1], "/api/v1/requests")
        job_call = management.calls[1]
        self.assertEqual(job_call[1], "/api/v1/requests/import-1/jobs")
        self.assertEqual(job_call[2], {
            "idempotency_key": "import:k:job:0", "seed": 3, "index": 0})
        uploads = management.calls[2:]
        self.assertEqual([call[1] for call in uploads],
                         ["/api/v1/jobs/job-1/generations"] * 2)
        self.assertEqual(
            [call[3][0]["idempotency_key"] for call in uploads],
            ["import:k:job:0:gen:0", "import:k:job:0:gen:1"])
        self.assertEqual([call[3][2] for call in uploads], ["a.png", "b.png"])
        self.assertEqual(result["request_id"], "import-1")
        self.assertEqual(len(result["generation_ids"]), 2)


class AttachAssetTests(unittest.TestCase):
    def test_posts_multipart_asset(self):
        management = ManagementFake()
        attach_asset(management, "gen-1", role="mask", name="a.png", data=b"x")
        self.assertEqual(len(management.calls), 1)
        method, path, payload, multipart = management.calls[0]
        self.assertEqual((method, path, payload),
                         ("POST", "/api/v1/generations/gen-1/assets", None))
        self.assertEqual(multipart[0]["role"], "mask")
        self.assertTrue(multipart[0]["idempotency_key"])
        self.assertEqual(multipart[1:], ("file", "a.png", b"x", "image/png"))


if __name__ == "__main__":
    unittest.main()
