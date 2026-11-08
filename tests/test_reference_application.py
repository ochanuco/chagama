from __future__ import annotations

import base64
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from chagama.application.generate import validate_request
from chagama.application.reference import (
    FREE_SETTLE_SECONDS,
    ReferenceImageMissing,
    ReferenceServices,
    generate_request,
    pose_canvas,
    pose_patch,
    reference,
    reference_render,
    with_headroom,
)
from chagama.application.work import work_once
from chagama.infrastructure.imaging.pose_skeleton import HEAD_POINTS, head_overflow
from chagama.domain.yukari.reference import Plan
from chagama.infrastructure.reference.analyze import analyze
from chagama.infrastructure.reference.anima_tagger import TagGroup
from test_work import ManagementFake, make_services

FIXTURES = Path(__file__).parent / "fixtures" / "reference"
GROUPS = json.loads((FIXTURES / "groups.json").read_text(encoding="utf-8"))
REF02 = json.loads((FIXTURES / "ref02.json").read_text(encoding="utf-8"))


def png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (8, 12), (200, 200, 200)).save(out, "PNG")
    return out.getvalue()


@dataclasses.dataclass
class FakeTagging:
    scores: dict
    kept: dict
    groups: tuple


class FakeTagger:
    def __init__(self, log):
        self.log = log

    def predict(self, image):
        self.log.append("tag")
        return FakeTagging(
            REF02["scores"], dict.fromkeys(REF02["kept"], 0.9),
            tuple(TagGroup(name, "multilabel", tuple(tags))
                  for name, tags in GROUPS.items()))

    def close(self):
        self.log.append("tagger closed")


class FakeVerifier:
    def __init__(self, log):
        self.log = log

    def verify(self, image, tags):
        self.log.append(("verify", tuple(tags)))
        return {tag: REF02["verdicts"].get(tag, 0.0) for tag in tags}

    def close(self):
        self.log.append("verifier closed")


def fake_estimator(log):
    def estimate(bgr):
        log.append(("pose", bgr.shape))
        height, width = bgr.shape[:2]
        keypoints = [[[width / 2, height / 4]] * 133, [[1.0, 1.0]] * 133]
        scores = [[0.9] * 133, [0.1] * 133]
        return keypoints, scores
    return estimate


def fake_segmenters(log):
    def anime(rgb):
        log.append(("segment", "anime"))
        mask = np.zeros(rgb.shape[:2], np.float32)
        mask[2:6, 3:9] = 1.0
        return mask
    return [("anime", anime)]


def fake_depth(image):
    width, height = image.size
    return np.ones((height, width), np.float32)


class Chimera:
    def __init__(self, image=b""):
        self.image = image or png()
        self.calls = []
        self.deleted = []

    source = None

    def request(self, method, path, payload=None, multipart=None):
        self.calls.append((method, path, payload))
        if method == "GET":
            return self.source
        return {"id": "gen-req-1", "short_id": "abc123"}


def make(chimera, log, image=...):
    return ReferenceServices(
        management=chimera,
        fetch_image=lambda request_id: chimera.image if image is ... else image,
        delete_image=chimera.deleted.append,
        free_comfyui=lambda: log.append("free"),
        analyze=lambda data: analyze(Image.open(io.BytesIO(data)),
                                     lambda: FakeTagger(log), lambda: FakeVerifier(log),
                                     lambda: fake_estimator(log),
                                     lambda: fake_segmenters(log),
                                     lambda: fake_depth),
        emit=lambda message: None,
        sleep=lambda seconds: log.append(("sleep", seconds)),
        random_seed=lambda: 42,
    )


def row(**payload):
    return {"id": "ref-1", "short_id": "r00001", "kind": "reference",
            "recipe_ref": "production", "attempt": 1, "max_attempts": 3,
            "payload": {"recipe": "yukari", **payload}}


class ReferenceTest(unittest.TestCase):
    def test_analyses_proposes_and_deletes_the_image_without_drawing(self):
        chimera, log = Chimera(), []
        result = reference(row(seed=7), make(chimera, log))
        self.assertEqual(log, [
            "free", ("sleep", FREE_SETTLE_SECONDS), "tag", "tagger closed",
            ("verify", ("looking up", "light smile")), "verifier closed",
            ("pose", (12, 8, 3)), ("segment", "anime")])
        self.assertEqual(chimera.calls, [])
        self.assertEqual(chimera.deleted, ["ref-1"])
        self.assertEqual(set(result), {"generation_ids", "pose", "candidates",
                                       "applied", "dropped", "figure"})
        self.assertEqual(result["figure"]["aspect"], 8 / 12)
        self.assertEqual(result["figure"]["cut"], "anime")
        self.assertEqual(result["figure"]["points"][0], [0.5, 0.25, 0.9])
        self.assertEqual(result["generation_ids"], [])
        self.assertEqual(result["pose"], "yukari_persona")
        self.assertIn("hand between legs", result["applied"]["gesture"])
        listed = {(row["part"], row["tag"]) for row in result["candidates"]}
        for part, tags in result["applied"].items():
            self.assertLessEqual({(part, tag) for tag in tags}, listed)
        json.dumps(result)

    def test_a_rated_image_is_left_to_chimera(self):
        chimera, log = Chimera(), []
        rating = {"model": "wd", "rating": {}, "tags": {}}
        services = dataclasses.replace(make(chimera, log), rate_image=lambda data: rating)
        reference(row(), services)
        self.assertEqual(chimera.calls,
                         [("PUT", "/api/v1/requests/ref-1/reference-safety", rating)])
        self.assertEqual(chimera.deleted, [])

    def test_a_failed_rating_deletes_the_image(self):
        chimera, log = Chimera(), []

        def broken(data):
            raise RuntimeError("onnxruntime is not installed")

        services = dataclasses.replace(make(chimera, log), rate_image=broken)
        reference(row(), services)
        self.assertEqual(chimera.calls, [])
        self.assertEqual(chimera.deleted, ["ref-1"])

    def test_a_missing_image_raises_before_any_model_loads(self):
        chimera, log = Chimera(), []
        with self.assertRaises(ReferenceImageMissing):
            reference(row(), make(chimera, log, image=None))
        self.assertEqual(log, [])
        self.assertEqual(chimera.calls, [])

    def test_an_analysis_failure_keeps_the_image(self):
        chimera, log = Chimera(), []

        def broken(data):
            raise RuntimeError("reference analysis exited 1: no timm")

        services = dataclasses.replace(make(chimera, log), analyze=broken)
        with self.assertRaisesRegex(RuntimeError, "no timm"):
            reference(row(), services)
        self.assertEqual(chimera.deleted, [])
        self.assertEqual(chimera.calls, [])

    def test_the_analysis_carries_only_candidate_scores(self):
        found = analyze(Image.new("RGB", (4, 4)), lambda: FakeTagger([]),
                        lambda: FakeVerifier([]), lambda: fake_estimator([]),
                        lambda: fake_segmenters([]), lambda: fake_depth)
        self.assertEqual(set(found), {"scores", "kept", "groups", "verdicts", "figure"})
        self.assertEqual(set(found["groups"]), set(GROUPS))
        self.assertIn("1girl", found["kept"])
        self.assertNotIn("1girl", found["scores"])
        json.dumps(found)

    def test_no_figure_is_none(self):
        found = analyze(Image.new("RGB", (4, 4)), lambda: FakeTagger([]),
                        lambda: FakeVerifier([]), lambda: lambda bgr: ([], []),
                        lambda: fake_segmenters([]), lambda: fake_depth)
        self.assertIsNone(found["figure"])

    def test_another_recipe_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "recipe"):
            reference({**row(), "payload": {"recipe": "other"}}, make(Chimera(), []))


def render_row(**payload):
    return {"id": "rr-1", "kind": "reference_render", "recipe_ref": "production",
            "payload": {"reference_request_id": "ref-1", "pose": "yukari_persona",
                        "applied": {"gesture": ["sitting"],
                                    "framing_tags": ["from above"]},
                        **payload}}


def analysed_source():
    chimera = Chimera()
    result = reference(row(), make(chimera, []))
    return {"id": "ref-1", "short_id": "r00001", "kind": "reference",
            "status": "done", "result": result}


class ReferenceRenderTest(unittest.TestCase):
    def test_queues_one_generate_from_the_picked_tags(self):
        chimera = Chimera()
        chimera.source = analysed_source()
        result = reference_render(render_row(seed=9), make(chimera, []))
        (get, post) = chimera.calls
        self.assertEqual(get[:2], ("GET", "/api/v1/requests/ref-1"))
        method, path, body = post
        self.assertEqual((method, path), ("POST", "/api/v1/requests"))
        self.assertEqual(body["created_by"], "system")
        self.assertEqual(body["idempotency_key"], "reference_render:rr-1")
        self.assertEqual(body["recipe_ref"], "production")
        payload = body["payload"]
        validate_request(payload)
        self.assertEqual(payload["request"]["seeds"], [9])
        self.assertEqual(payload["generation"]["parameters"], {"pose": "yukari_persona"})
        self.assertEqual(payload["semantic"]["attributes"]["reference"], {
            "source_request": "r00001", "pose": "yukari_persona",
            "applied": {"framing_tags": ["from above"], "gesture": ["sitting"]},
            "pose_follow": "loose"})
        patches = {patch["target"]: patch for patch in payload["generation"]["patches"]
                   if patch["op"] != "remove"}
        self.assertEqual(patches["prompt.positive.framing_tags"]["value"],
                         "(from above:1.3), ")
        figure = chimera.source["result"]["figure"]
        self.assertEqual(patches["render.pose"]["value"], {
            "points": figure["points"], "aspect": figure["aspect"], "strength": 1.0})
        self.assertEqual((patches["render.width"]["value"], patches["render.height"]["value"]),
                         pose_canvas("yukari_persona", 8 / 12))
        self.assertEqual(result, {"generation_ids": [],
                                  "generate_request_id": "gen-req-1",
                                  "generate_request_short_id": "abc123"})

    def pose_of(self, chimera):
        payload = chimera.calls[-1][2]["payload"]
        validate_request(payload)
        (pose,) = [patch["value"] for patch in payload["generation"]["patches"]
                   if patch["target"] == "render.pose"]
        return pose

    def test_firm_follows_the_depth_with_the_head_free(self):
        chimera = Chimera()
        chimera.source = analysed_source()
        reference_render(render_row(seed=9, pose_follow="firm"), make(chimera, []))
        figure = chimera.source["result"]["figure"]
        self.assertEqual(self.pose_of(chimera), {
            "control": "depth", "picture": figure["depth"], "free_head": True,
            "free_chest": True, "blur": 0.015, "points": figure["points"], "aspect": figure["aspect"], "strength": 1.0})

    def test_a_back_view_holds_the_head_and_chest_and_draws_the_back(self):
        figure = analysed_source()["result"]["figure"]
        plan = Plan(pose="yukari_persona", patches=[], dropped=[], applied={
            "framing_tags": ["from behind", "from below"], "gesture": ["standing"]})
        payload = generate_request(plan, seed=1, source="r1", figure=figure,
                                   pose_follow="firm")
        validate_request(payload)
        self.assertEqual(payload["generation"]["parameters"],
                         {"pose": "yukari_persona", "angle": "from_behind"})
        (pose,) = [patch["value"] for patch in payload["generation"]["patches"]
                   if patch["target"] == "render.pose"]
        self.assertEqual((pose["free_head"], pose["free_chest"]), (False, False))

    def test_a_back_view_looking_back_leaves_the_head_free(self):
        figure = analysed_source()["result"]["figure"]
        plan = Plan(pose="yukari_persona", patches=[], dropped=[], applied={
            "framing_tags": ["facing away"], "gesture": ["looking back"]})
        pose = pose_patch(figure, "firm", applied=plan.applied)["value"]
        self.assertEqual((pose["free_head"], pose["free_chest"]), (True, False))

    def test_firm_without_a_depth_follows_the_skeleton(self):
        chimera = Chimera()
        chimera.source = analysed_source()
        del chimera.source["result"]["figure"]["depth"]
        reference_render(render_row(pose_follow="firm"), make(chimera, []))
        self.assertNotIn("control", self.pose_of(chimera))

    def test_an_unknown_follow_is_refused(self):
        chimera = Chimera()
        chimera.source = analysed_source()
        with self.assertRaisesRegex(SystemExit, "pose_follow"):
            reference_render(render_row(pose_follow="hard"), make(chimera, []))
        self.assertEqual([call[0] for call in chimera.calls], ["GET"])

    def test_the_canvas_keeps_the_pose_area_at_the_reference_aspect(self):
        self.assertEqual(pose_canvas("yukari_persona", 1.39), (1520, 1104))
        self.assertEqual(pose_canvas("yukari_persona", 1024 / 1640), (1024, 1632))
        self.assertEqual(pose_canvas("yukari_persona", 5.0), pose_canvas("yukari_persona", 2.0))

    def test_off_follows_only_the_tags(self):
        chimera = Chimera()
        chimera.source = analysed_source()
        reference_render(render_row(pose_follow="off"), make(chimera, []))
        payload = chimera.calls[-1][2]["payload"]
        validate_request(payload)
        targets = [patch["target"] for patch in payload["generation"]["patches"]]
        self.assertNotIn("render.pose", targets)
        self.assertNotIn("render.width", targets)

    def test_a_reference_without_a_figure_follows_only_the_tags(self):
        chimera = Chimera()
        chimera.source = analysed_source()
        chimera.source["result"]["figure"] = None
        reference_render(render_row(), make(chimera, []))
        payload = chimera.calls[-1][2]["payload"]
        self.assertNotIn("render.pose", [patch["target"]
                                         for patch in payload["generation"]["patches"]])
        self.assertIsNone(payload["semantic"]["attributes"]["reference"]["pose_follow"])

    def test_a_missing_seed_is_drawn(self):
        chimera = Chimera()
        chimera.source = analysed_source()
        reference_render(render_row(), make(chimera, []))
        self.assertEqual(chimera.calls[-1][2]["payload"]["request"]["seeds"], [42])

    def test_a_tag_outside_the_candidates_fails_before_queuing(self):
        chimera = Chimera()
        chimera.source = analysed_source()
        with self.assertRaisesRegex(SystemExit, "mouth:sitting"):
            reference_render(render_row(applied={"mouth": ["sitting"]}),
                             make(chimera, []))
        self.assertEqual([call[0] for call in chimera.calls], ["GET"])

    def test_a_prepared_pose_fails(self):
        chimera = Chimera()
        chimera.source = analysed_source()
        with self.assertRaisesRegex(SystemExit, "pose"):
            reference_render(render_row(pose="stand"), make(chimera, []))

    def test_the_source_must_be_a_done_reference(self):
        chimera = Chimera()
        chimera.source = {**analysed_source(), "status": "running"}
        with self.assertRaisesRegex(SystemExit, "not a done reference"):
            reference_render(render_row(), make(chimera, []))

    def test_a_malformed_payload_fails(self):
        with self.assertRaisesRegex(SystemExit, "applied"):
            reference_render(render_row(applied={"gesture": "sitting"}),
                             make(Chimera(), []))


class WorkerReferenceTest(unittest.TestCase):
    def run_row(self, claimed, reference_result):
        management = ManagementFake(claim_responses=[claimed])
        with tempfile.TemporaryDirectory() as directory:
            services = dataclasses.replace(
                make_services(directory, management, branch="production"),
                reference_services="reference-services",
                reference=reference_result)
            work_once(services)
        return [call[2] for call in management.calls
                if call[0] == "PATCH" and call[1] == "/api/v1/requests/ref-1"
                and call[2]["status"] != "running"]

    def test_done_reports_the_result(self):
        result = {"generation_ids": [], "generate_request_id": "g"}
        reports = self.run_row(row(), lambda claimed, services: result)
        self.assertEqual(reports, [{"status": "done", "result": result,
                                    "worker_id": "test-worker"}])

    def test_a_render_row_reaches_reference_render(self):
        result = {"generation_ids": [], "generate_request_id": "g",
                  "generate_request_short_id": "s"}
        management = ManagementFake(claim_responses=[{**render_row(), "id": "ref-1"}])
        with tempfile.TemporaryDirectory() as directory:
            services = dataclasses.replace(
                make_services(directory, management, branch="production"),
                reference_services="reference-services",
                reference_render=lambda claimed, services: result)
            work_once(services)
        done = [call[2] for call in management.calls
                if call[0] == "PATCH" and call[2]["status"] == "done"]
        self.assertEqual(done, [{"status": "done", "result": result,
                                 "worker_id": "test-worker"}])

    def test_a_missing_image_with_attempts_left_goes_back_to_the_queue(self):
        def missing(claimed, services):
            raise ReferenceImageMissing("no reference image")
        reports = self.run_row(row(), missing)
        self.assertEqual(reports, [{"status": "queued", "worker_id": "test-worker"}])

    def test_a_missing_image_on_the_last_attempt_fails(self):
        def missing(claimed, services):
            raise ReferenceImageMissing("no reference image")
        reports = self.run_row({**row(), "attempt": 3}, missing)
        self.assertEqual(reports, [{"status": "failed", "error": "no reference image",
                                    "worker_id": "test-worker"}])


if __name__ == "__main__":
    unittest.main()


def cropped_head_figure():
    points = [[0.5, 0.6, 0.9]] * 133
    for index in HEAD_POINTS:
        points[index] = [0.5, 0.02, 0.9]
    points[0], points[3], points[4] = [0.5, 0.06, 0.9], [0.45, 0.02, 0.9], [0.55, 0.02, 0.9]
    depth = io.BytesIO()
    Image.new("L", (20, 10), 200).save(depth, format="PNG")
    return {"points": points, "aspect": 2.0,
            "depth": base64.b64encode(depth.getvalue()).decode("ascii")}


class HeadroomTest(unittest.TestCase):
    def test_a_head_the_reference_crops_gets_room_above_it(self):
        figure = cropped_head_figure()
        grown = with_headroom(figure)
        self.assertLess(grown["aspect"], figure["aspect"])
        self.assertEqual(head_overflow(grown["points"], grown["aspect"]), (0.0, 0.0, 0.0, 0.0))
        depth = np.asarray(Image.open(io.BytesIO(base64.b64decode(grown["depth"]))))
        self.assertEqual(depth.shape[1], 20)
        self.assertGreater(depth.shape[0], 10)
        self.assertEqual(depth[0].max(), 0)
        self.assertEqual(depth[-1].min(), 200)

    def test_a_head_inside_the_frame_is_left_alone(self):
        figure = cropped_head_figure()
        figure["points"] = [[x, y + 0.4, score] for x, y, score in figure["points"]]
        self.assertIs(with_headroom(figure), figure)

    def test_a_cropped_head_is_drawn_whole_without_the_cropping_tags(self):
        chimera = Chimera()
        chimera.source = analysed_source()
        chimera.source["result"]["figure"] = cropped_head_figure()
        chimera.source["result"]["candidates"].append(
            {"tag": "head out of frame", "part": "framing_tags", "score": 0.6,
             "kept": True, "d1": None, "applied": True})
        reference_render(render_row(pose_follow="firm", applied={
            "gesture": ["sitting"], "framing_tags": ["from above", "head out of frame"]}),
            make(chimera, []))
        payload = chimera.calls[-1][2]["payload"]
        framing = [patch["value"] for patch in payload["generation"]["patches"]
                   if patch["target"] == "prompt.positive.framing_tags"]
        self.assertTrue(framing)
        self.assertNotIn("head out of frame", "".join(framing))
        pose = next(patch["value"] for patch in payload["generation"]["patches"]
                    if patch["target"] == "render.pose")
        self.assertLess(pose["aspect"], 2.0)
