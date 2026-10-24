"""Adapter-level tests for the generate use case.

All collaborators below are fakes: this suite never opens a network socket.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from chagama.application import ingest
from chagama.application.generate import (
    GenerateServices,
    _image_output_path,
    _output_directory,
    apply_presets,
    generate,
    request_generation,
    request_graph,
    resolution_payload,
    validate_request,
)


class ManagementFake:
    def __init__(self):
        self.calls = []
        self.resolution_jobs = None

    def request(self, method, path, payload=None, multipart=None):
        self.calls.append((method, path, payload, multipart))
        is_resolution = method == "PUT" and path.endswith("/resolution")
        if is_resolution or (method == "POST" and path == "/api/v1/requests"):
            request = {"id": "request-id", "short_id": "req"}
            if is_resolution and self.resolution_jobs is not None:
                request["jobs"] = self.resolution_jobs
            return request
        if path.endswith("/generations"):
            return {"id": "generation", "short_id": "gen", "canonical_url": "https://example/g"}
        if method == "POST" and path.endswith("/jobs"):
            return {"id": "job-id"}
        return {}

    def resolve_character(self, name):
        return "character-id"

    def put_semantic(self, generation_id, semantic):
        self.calls.append(("semantic", generation_id, semantic))
        return {}


class ComfyFake:
    def __init__(self):
        self.submits = []
        self.waits = []
        self.fetches = []

    def submit(self, graph):
        self.submits.append(graph)
        return "prompt-id"

    def knows(self, prompt_id):
        return True

    def wait_for(self, prompt_id):
        self.waits.append(prompt_id)
        return []

    def fetch(self, image):
        self.fetches.append(image)
        return b"image"


class StateFake:
    def __init__(self, state):
        self.state = state
        self.saved = []

    def load(self, path):
        return self.state

    def save(self, path, state):
        self.saved.append(json.loads(json.dumps(state)))


class NullNotifier:
    def send(self, *args):
        raise AssertionError("no notification expected in this test")


class RecordingNotifier:
    def __init__(self):
        self.calls = []

    def send(self, *args):
        self.calls.append(args)


def base_request(**generation):
    return {
        "schema_version": 1,
        "request": {"count": 1, "instruction": "test", "seeds": [42]},
        "generation": {"recipe": "yukari", "parameters": {"pose": "lounge"}, **generation},
        "semantic": {"summary": "test arm"},
    }


class GenerateApplicationTest(unittest.TestCase):
    def test_validate_request_rejects_schema_and_semantic(self):
        with self.assertRaises(SystemExit):
            validate_request({**base_request(), "schema_version": 2})
        missing = base_request()
        del missing["semantic"]
        with self.assertRaises(SystemExit):
            validate_request(missing)

    def test_validate_request_rejects_malformed_nested_values(self):
        invalid = [
            [],
            {**base_request(), "request": None},
            {**base_request(), "request": []},
            {**base_request(), "generation": []},
            {**base_request(), "semantic": []},
            {**base_request(), "generation": {
                "recipe": "yukari", "parameters": None}},
            {**base_request(), "request": {
                "count": 1, "instruction": "test", "seeds": "42"}},
            {**base_request(), "request": {
                "count": 1, "instruction": "test", "seeds": [True]}},
        ]
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(SystemExit):
                validate_request(request)

    def test_validate_request_accepts_a_pose_pinned_only_by_a_preset(self):
        validate_request(base_request(
            parameters={},
            presets=[{"kind": "pose", "name": "date", "version": 1}]))

    def test_validate_request_still_requires_a_pose_without_a_pose_preset(self):
        with self.assertRaises(SystemExit):
            validate_request(base_request(
                parameters={},
                presets=[{"kind": "costume", "name": "outing", "version": 1}]))

    def test_validate_request_rejects_unknown_parameters(self):
        request = base_request()
        request["generation"]["parameters"]["bogus_param"] = "smile"
        with self.assertRaises(SystemExit):
            validate_request(request)

    def test_validate_request_allows_unknown_parameters_in_graph_mode(self):
        request = base_request(
            graph={"a": {"class_type": "KSampler", "inputs": {}}})
        request["generation"]["parameters"]["bogus_param"] = "smile"
        validate_request(request)

    def test_validate_request_rejects_patches_with_graph(self):
        request = base_request(
            graph={"a": {"class_type": "KSampler", "inputs": {}}},
            patches=[{"target": "render.cfg", "op": "set", "value": 4.5,
                     "reason": "test"}])
        with self.assertRaises(SystemExit):
            validate_request(request)

    def test_validate_request_rejects_patches_with_prompt_override(self):
        request = base_request(
            prompt="override",
            patches=[{"target": "render.cfg", "op": "set", "value": 4.5,
                     "reason": "test"}])
        with self.assertRaises(SystemExit):
            validate_request(request)

    def test_validate_request_rejects_malformed_patch_shape(self):
        request = base_request(patches=[
            {"target": "render.cfg", "op": "set", "value": 4.5}])
        with self.assertRaises(SystemExit):
            validate_request(request)

    def test_validate_request_accepts_well_formed_patches(self):
        request = base_request(patches=[
            {"target": "render.cfg", "op": "set", "value": 4.5,
             "reason": "test"}])
        validate_request(request)

    def test_validate_request_accepts_a_patch_word_for_a_recipe_with_dials(self):
        request = base_request(
            patches=[{"target": "render.width", "op": "set",
                     "value": "draft", "reason": "test"}])
        validate_request(request)

    def test_validate_request_rejects_an_unknown_patch_word(self):
        request = base_request(patches=[
            {"target": "render.width", "op": "set", "value": "bogus",
             "reason": "test"}])
        with self.assertRaisesRegex(SystemExit, "bogus"):
            validate_request(request)

    def test_validate_request_accepts_well_formed_presets(self):
        request = base_request(presets=[
            {"kind": "pose", "name": "lounge", "version": 1}])
        validate_request(request)

    def test_validate_request_accepts_presets_with_patches(self):
        request = base_request(
            presets=[{"kind": "pose", "name": "lounge", "version": 1}],
            patches=[{"target": "render.cfg", "op": "set", "value": 4.5,
                     "reason": "test"}])
        validate_request(request)

    def test_validate_request_rejects_malformed_presets_entry(self):
        invalid = [
            [{"kind": "pose", "name": "lounge"}],
            [{"kind": "pose", "name": "lounge", "version": 0}],
            [{"kind": "pose", "name": "lounge", "version": 1, "extra": "x"}],
            [{"kind": "", "name": "lounge", "version": 1}],
            [{"kind": "pose", "name": "", "version": 1}],
            [{"kind": "pose", "name": "lounge", "version": True}],
            [{"kind": "pose", "name": "lounge", "version": "1"}],
            "not-a-list",
        ]
        for presets in invalid:
            with self.subTest(presets=presets), self.assertRaises(SystemExit):
                validate_request(base_request(presets=presets))

    def test_validate_request_rejects_presets_with_graph(self):
        request = base_request(
            graph={"a": {"class_type": "KSampler", "inputs": {}}},
            presets=[{"kind": "pose", "name": "lounge", "version": 1}])
        with self.assertRaises(SystemExit):
            validate_request(request)

    def test_validate_request_rejects_presets_with_prompt_override(self):
        request = base_request(
            prompt="override",
            presets=[{"kind": "pose", "name": "lounge", "version": 1}])
        with self.assertRaises(SystemExit):
            validate_request(request)

    def test_validate_request_rejects_empty_lint_waiver(self):
        with self.assertRaises(SystemExit):
            validate_request(base_request(lint_waiver=""))
        with self.assertRaises(SystemExit):
            validate_request(base_request(lint_waiver=123))

    def test_validate_request_accepts_lint_waiver(self):
        validate_request(base_request(lint_waiver="deliberate for this arm"))

    def test_validate_request_rejects_empty_identity_override(self):
        with self.assertRaises(SystemExit):
            validate_request(base_request(identity_override=""))
        with self.assertRaises(SystemExit):
            validate_request(base_request(identity_override=123))

    def test_validate_request_accepts_identity_override(self):
        validate_request(base_request(identity_override="deliberate crop"))

    def test_validate_request_accepts_experiment_block(self):
        request = base_request()
        request["experiment"] = {"experiment_id": "exp-1", "run_id": "run-1"}
        validate_request(request)
        request["experiment"]["overrides"] = {"patches": [
            {"target": "render.cfg", "op": "set", "value": 4.5,
             "reason": "test"}]}
        validate_request(request)

    def test_validate_request_accepts_explicit_null_experiment(self):
        request = base_request()
        request["experiment"] = None
        validate_request(request)

    def test_validate_request_rejects_malformed_experiment(self):
        invalid = [
            {**base_request(), "experiment": []},
            {**base_request(), "experiment": {"run_id": "run-1"}},
            {**base_request(), "experiment": {
                "experiment_id": "", "run_id": "run-1"}},
            {**base_request(), "experiment": {
                "experiment_id": 1, "run_id": "run-1"}},
            {**base_request(), "experiment": {"experiment_id": "exp-1"}},
            {**base_request(), "experiment": {
                "experiment_id": "exp-1", "run_id": ""}},
            {**base_request(), "experiment": {
                "experiment_id": "exp-1", "run_id": 1}},
            {**base_request(), "experiment": {
                "experiment_id": "exp-1", "run_id": "run-1",
                "overrides": []}},
            {**base_request(), "experiment": {
                "experiment_id": "exp-1", "run_id": "run-1",
                "overrides": {"patches": [
                    {"target": "nope", "op": "set", "value": 1,
                     "reason": "test"}]}}},
        ]
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(SystemExit):
                validate_request(request)

    def test_validate_request_rejects_experiment_patches_with_generation_patches(self):
        request = base_request(patches=[
            {"target": "render.cfg", "op": "set", "value": 4.5,
             "reason": "test"}])
        request["experiment"] = {
            "experiment_id": "exp-1", "run_id": "run-1",
            "overrides": {"patches": [
                {"target": "render.steps", "op": "set", "value": 20,
                 "reason": "test"}]}}
        with self.assertRaises(SystemExit):
            validate_request(request)

    def test_validate_request_rejects_experiment_patches_with_graph(self):
        request = base_request(
            graph={"a": {"class_type": "KSampler", "inputs": {}}})
        request["experiment"] = {
            "experiment_id": "exp-1", "run_id": "run-1",
            "overrides": {"patches": [
                {"target": "render.cfg", "op": "set", "value": 4.5,
                 "reason": "test"}]}}
        with self.assertRaises(SystemExit):
            validate_request(request)

    def test_validate_request_rejects_experiment_patches_with_prompt_override(self):
        request = base_request(prompt="override")
        request["experiment"] = {
            "experiment_id": "exp-1", "run_id": "run-1",
            "overrides": {"patches": [
                {"target": "render.cfg", "op": "set", "value": 4.5,
                 "reason": "test"}]}}
        with self.assertRaises(SystemExit):
            validate_request(request)

    def test_request_generation_lifts_experiment_patches(self):
        patches = [{"target": "render.cfg", "op": "set", "value": 4.5,
                   "reason": "test"}]
        request = base_request()
        request["experiment"] = {
            "experiment_id": "exp-1", "run_id": "run-1",
            "overrides": {"patches": patches}}
        generation = request_generation(request)
        self.assertEqual(generation["patches"], patches)

    def test_request_generation_leaves_existing_generation_patches_alone(self):
        own_patches = [{"target": "render.steps", "op": "set", "value": 20,
                        "reason": "test"}]
        experiment_patches = [{"target": "render.cfg", "op": "set",
                               "value": 4.5, "reason": "test"}]
        request = base_request(patches=own_patches)
        request["experiment"] = {
            "experiment_id": "exp-1", "run_id": "run-1",
            "overrides": {"patches": experiment_patches}}
        generation = request_generation(request)
        self.assertEqual(generation["patches"], own_patches)

    def test_request_graph_applies_experiment_override_patches(self):
        from chagama.domain.generation.models import PromptPair, RenderSpec

        def builder(*args, **kwargs):
            return RenderSpec(
                model_path="m", prompts=PromptPair("base positive", "base neg"),
                width=8, height=8, seed=42, steps=30, cfg=5.0,
                sampler_name="s", scheduler="k", denoise=1.0,
                filename_prefix="p")

        def encode(spec):
            return {"6": {"inputs": {"text": spec.prompts.positive}},
                    "7": {"inputs": {"text": spec.prompts.negative}}}

        request = base_request()
        request["experiment"] = {
            "experiment_id": "exp-1", "run_id": "run-1",
            "overrides": {"patches": [
                {"target": "prompt.positive", "op": "append",
                 "value": ", extra tag", "reason": "test"}]}}
        generation = request_generation(request)
        graph = request_graph(generation, 42, "prefix", builder, encode)
        self.assertEqual(graph["6"]["inputs"]["text"], "base positive, extra tag")

    def test_resolution_payload_does_not_forward_experiment_refinement_or_story(self):
        request = base_request()
        request["experiment"] = {"experiment_id": "exp-1", "run_id": "run-1"}
        request["refinement"] = {"actor": "human"}
        request["story"] = {"title": "t"}
        payload = resolution_payload(request, {"commit": "c", "dirty": False})
        for key in ("experiment", "refinement", "story", "idempotency_key"):
            self.assertNotIn(key, payload)

    def test_resolution_payload_carries_patches_and_pose_fingerprint(self):
        request = base_request()
        resolved = dict(request["generation"])
        resolved["patches"] = [{"target": "render.cfg", "op": "set",
                                "value": 4.5, "reason": "test"}]
        resolved["parameters"] = {**resolved["parameters"], "pose": "resolved-pose"}
        payload = resolution_payload(
            request, {"commit": "c", "dirty": False},
            generation=resolved, pose_fingerprint="sha256:abc")
        self.assertEqual(payload["patches"], resolved["patches"])
        self.assertEqual(payload["pose_fingerprint"], "sha256:abc")
        self.assertEqual(payload["parameters"]["pose"], "resolved-pose")

    def test_resolution_payload_records_only_the_requests_own_patches(self):
        request = base_request()
        own = [{"target": "render.cfg", "op": "set", "value": 4.5,
                "reason": "the alpha"}]
        resolved = dict(request["generation"])
        resolved["patches"] = [{"target": "render.steps", "op": "set",
                                "value": 30, "reason": "from the preset"}] + own
        payload = resolution_payload(
            request, {"commit": "c", "dirty": False},
            generation=resolved, patches=own)
        self.assertEqual(payload["patches"], own)

    def test_resolution_payload_omits_patches_when_the_request_added_none(self):
        request = base_request()
        resolved = dict(request["generation"])
        resolved["patches"] = [{"target": "render.steps", "op": "set",
                                "value": 30, "reason": "from the preset"}]
        payload = resolution_payload(
            request, {"commit": "c", "dirty": False},
            generation=resolved, patches=[])
        self.assertNotIn("patches", payload)

    def test_resolution_payload_omits_patches_and_fingerprint_when_absent(self):
        request = base_request()
        payload = resolution_payload(request, {"commit": "c", "dirty": False})
        self.assertNotIn("patches", payload)
        self.assertNotIn("pose_fingerprint", payload)

    def test_apply_presets_leaves_generation_without_presets_untouched(self):
        generation = {"recipe": "yukari", "parameters": {"pose": "lounge"}}
        frozen = json.loads(json.dumps(generation))
        result = apply_presets(generation, lambda *_: {})
        self.assertEqual(result, generation)
        self.assertEqual(generation, frozen)

    def test_apply_presets_sets_pose_and_orders_patches_before_own(self):
        generation = {
            "recipe": "yukari",
            "parameters": {"pose": "placeholder"},
            "patches": [{"target": "render.cfg", "op": "set", "value": 4.5,
                        "reason": "own"}],
            "presets": [{"kind": "pose", "name": "lounge", "version": 3}],
        }
        frozen = json.loads(json.dumps(generation))

        def fetch(recipe, kind, name, version):
            self.assertEqual((recipe, kind, name, version), ("yukari", "pose", "lounge", 3))
            return {
                "record": {"recipe_pose": "lounge-v3"},
                "patches": [{"target": "render.steps", "op": "set", "value": 20,
                            "reason": "preset"}],
            }

        result = apply_presets(generation, fetch)
        self.assertEqual(result["parameters"]["pose"], "lounge-v3")
        self.assertEqual(
            result["patches"],
            [{"target": "render.steps", "op": "set", "value": 20, "reason": "preset"},
             {"target": "render.cfg", "op": "set", "value": 4.5, "reason": "own"}])
        self.assertEqual(generation, frozen)

    def test_apply_presets_raises_on_unsupported_kind(self):
        generation = {
            "recipe": "yukari", "parameters": {},
            "presets": [{"kind": "costume", "name": "school", "version": 1}],
        }
        with self.assertRaises(SystemExit):
            apply_presets(generation, lambda *_: {"record": {}})

    def test_apply_presets_raises_on_missing_recipe_pose(self):
        generation = {
            "recipe": "yukari", "parameters": {},
            "presets": [{"kind": "pose", "name": "lounge", "version": 1}],
        }
        with self.assertRaises(SystemExit):
            apply_presets(generation, lambda *_: {"record": {}})

    def test_output_paths_must_remain_inside_configured_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(_output_directory(root, "req"), root / "req")
            for identifier in ("../escape", "/tmp/escape", r"..\escape"):
                with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                    _output_directory(root, identifier)
            output = root / "req"
            output.mkdir()
            self.assertEqual(
                _image_output_path(output, "render.png"), output / "render.png")
            for filename in ("../escape.png", "/tmp/escape.png"):
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    _image_output_path(output, filename)

    def test_request_graph_explicit_graph_rewrites_job_fields_only(self):
        original = {"a": {"class_type": "KSampler", "inputs": {"seed": 1}},
                    "b": {"class_type": "SaveImage", "inputs": {"filename_prefix": "old"}}}
        graph = request_graph({"recipe": "probe", "graph": original}, 99, "new",
                              lambda *a, **k: None, lambda s: None)
        self.assertEqual(graph["a"]["inputs"]["seed"], 99)
        self.assertEqual(graph["b"]["inputs"]["filename_prefix"], "new")
        self.assertEqual(original["a"]["inputs"]["seed"], 1)

    def test_request_graph_builds_yukari_and_applies_overrides(self):
        from chagama.domain.generation.models import PromptPair, RenderSpec

        seen = {}

        def builder(*args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return RenderSpec(
                model_path="m", prompts=PromptPair("built", "built-neg"),
                width=8, height=8, seed=42, steps=30, cfg=5.0,
                sampler_name="s", scheduler="k", denoise=1.0,
                filename_prefix="p")

        def encode(spec):
            seen["spec"] = spec
            return {"6": {"inputs": {"text": spec.prompts.positive}},
                    "7": {"inputs": {"text": spec.prompts.negative}}}

        generation = base_request(prompt="override", negative_prompt="override-neg")["generation"]
        graph = request_graph(generation, 42, "prefix", builder, encode)
        self.assertEqual(seen["args"], ("lounge", 42, "prefix"))
        # No costume/hires/denoise/expression in generation.parameters, so
        # none is forwarded -- the recipe's own default applies instead of
        # this layer inventing one.
        self.assertEqual(seen["kwargs"], {})
        self.assertEqual(seen["spec"].prompts.positive, "override")
        self.assertEqual(seen["spec"].prompts.negative, "override-neg")
        self.assertEqual(graph["6"]["inputs"]["text"], "override")
        self.assertEqual(graph["7"]["inputs"]["text"], "override-neg")

    def test_request_graph_forwards_only_present_optional_parameters(self):
        from chagama.domain.generation.models import PromptPair, RenderSpec

        seen = {}

        def builder(*args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return RenderSpec(
                model_path="m", prompts=PromptPair("built", "built-neg"),
                width=8, height=8, seed=42, steps=30, cfg=5.0,
                sampler_name="s", scheduler="k", denoise=1.0,
                filename_prefix="p")

        def encode(spec):
            return {"6": {"inputs": {"text": spec.prompts.positive}},
                    "7": {"inputs": {"text": spec.prompts.negative}}}

        generation = base_request()["generation"]
        generation["parameters"] = {
            "pose": "coffee", "costume": "outing", "expression": "doya"}
        request_graph(generation, 7, "prefix", builder, encode)
        self.assertEqual(seen["args"], ("coffee", 7, "prefix"))
        self.assertEqual(
            seen["kwargs"], {"costume": "outing", "expression": "doya"})

    def test_request_graph_passes_angle_and_framing_to_builder(self):
        from chagama.domain.generation.models import PromptPair, RenderSpec

        seen = {}

        def builder(*args, **kwargs):
            seen["kwargs"] = kwargs
            return RenderSpec(
                model_path="m", prompts=PromptPair("built", "built-neg"),
                width=8, height=8, seed=42, steps=30, cfg=5.0,
                sampler_name="s", scheduler="k", denoise=1.0,
                filename_prefix="p")

        generation = base_request()["generation"]
        generation["parameters"] = {
            "pose": "gao", "angle": "from_below,dutch", "framing": "full"}
        request_graph(generation, 7, "prefix", builder, lambda spec: {})
        self.assertEqual(seen["kwargs"],
                         {"angle": "from_below,dutch", "framing": "full"})

    def test_generate_resume_ingested_job_does_not_submit_or_create_job(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(base_request()), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            state = StateFake({
                "idempotency_key": "fixed-key", "request_id": "old-request",
                "seeds": [42], "jobs": [{"idempotency_key": "job-key",
                                           "job_id": "job-id",
                                           "comfy_prompt_id": "old-prompt",
                                           "status": "ingested"}],
            })
            services = GenerateServices(
                management, comfy, state, NullNotifier(),
                lambda generation, seed, prefix: {"6": {"inputs": {"text": "x"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None)
            generate(path, services)
            self.assertEqual(comfy.submits, [])
            self.assertEqual(comfy.waits, [])
            self.assertEqual(comfy.fetches, [])
            self.assertFalse(any(call[1].endswith("/jobs") for call in management.calls))
            self.assertFalse(any(
                call[0] == "PATCH" and call[1].startswith("/api/v1/requests/")
                for call in management.calls))

    def test_generate_resume_skips_registered_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(base_request()), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            comfy.wait_for = lambda prompt_id: [{"filename": "render.png"}]
            state = StateFake({
                "idempotency_key": "fixed-key", "seeds": [42],
                "jobs": [{"idempotency_key": "job-key", "job_id": "job-id",
                          "comfy_prompt_id": "old-prompt", "status": "failed",
                          "generations": [{"id": "generation",
                                           "short_id": "gen",
                                           "canonical_url": "https://example/g",
                                           "status": "registered"}]}],
            })
            services = GenerateServices(
                management, comfy, state, NullNotifier(),
                lambda generation, seed, prefix: {},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None)
            generate(path, services)
            self.assertEqual(comfy.fetches, [])
            self.assertFalse(any(call[0] == "POST" and call[1].endswith("/generations")
                                 for call in management.calls))
            self.assertEqual(state.state["jobs"][0]["status"], "ingested")

    def test_generate_resume_unknown_prompt_resubmits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(base_request()), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            comfy.knows = lambda prompt_id: False
            state = StateFake({
                "idempotency_key": "fixed-key", "seeds": [42],
                "jobs": [{"idempotency_key": "job-key", "job_id": "job-id",
                          "comfy_prompt_id": "old-prompt", "status": "failed"}],
            })
            services = GenerateServices(
                management, comfy, state, RecordingNotifier(),
                lambda generation, seed, prefix: {"6": {"inputs": {"text": "x"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None)
            generate(path, services)
            self.assertEqual(len(comfy.submits), 1)
            self.assertEqual(comfy.waits, ["prompt-id"])
            self.assertEqual(state.state["jobs"][0]["comfy_prompt_id"], "prompt-id")

    def test_generate_resume_known_prompt_does_not_resubmit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(base_request()), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            state = StateFake({
                "idempotency_key": "fixed-key", "seeds": [42],
                "jobs": [{"idempotency_key": "job-key", "job_id": "job-id",
                          "comfy_prompt_id": "old-prompt", "status": "failed"}],
            })
            services = GenerateServices(
                management, comfy, state, RecordingNotifier(),
                lambda generation, seed, prefix: {"6": {"inputs": {"text": "x"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None)
            generate(path, services)
            self.assertEqual(comfy.submits, [])
            self.assertEqual(comfy.waits, ["old-prompt"])
            self.assertEqual(state.state["jobs"][0]["comfy_prompt_id"], "old-prompt")

    def test_generate_reuses_output_idempotency_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(base_request()), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            comfy.wait_for = lambda prompt_id: [{"filename": "render.png"}]
            state = StateFake({
                "idempotency_key": "fixed-key", "seeds": [42],
                "jobs": [{"idempotency_key": "job-key", "job_id": "job-id",
                          "comfy_prompt_id": "old-prompt", "status": "failed",
                          "generations": [{"idempotency_key": "output-key",
                                           "comfy_output_index": 0}]}],
            })
            services = GenerateServices(
                management, comfy, state, RecordingNotifier(),
                lambda generation, seed, prefix: {},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None)
            generate(path, services)
            call = next(call for call in management.calls
                        if call[0] == "POST" and call[1].endswith("/generations"))
            self.assertEqual(call[3][0]["idempotency_key"], "output-key")
            self.assertEqual(
                state.state["jobs"][0]["generations"][0]["status"],
                "registered")

    def test_generate_records_patches_in_semantic_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            patches = [{"target": "render.cfg", "op": "set", "value": 4.5,
                       "reason": "test"}]
            path.write_text(json.dumps(base_request(patches=patches)), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            comfy.wait_for = lambda prompt_id: [{"filename": "render.png"}]
            state = StateFake({
                "idempotency_key": "fixed-key", "seeds": [42],
                "jobs": [{"idempotency_key": "job-key", "job_id": "job-id",
                          "comfy_prompt_id": "old-prompt", "status": "failed"}],
            })
            services = GenerateServices(
                management, comfy, state, RecordingNotifier(),
                lambda generation, seed, prefix: {
                    "6": {"inputs": {"text": "x"}}, "7": {"inputs": {"text": "y"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None,
                pose_fingerprint=lambda *_: "sha256:test")
            generate(path, services)
            semantic_call = next(
                call for call in management.calls if call[0] == "semantic")
            self.assertEqual(semantic_call[2]["attributes"]["patches"], patches)

    def test_generate_rates_each_registered_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(base_request()), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            comfy.wait_for = lambda prompt_id: [{"filename": "render.png"}]
            services = GenerateServices(
                management, comfy,
                StateFake({"idempotency_key": "fixed-key", "seeds": [42], "jobs": []}),
                RecordingNotifier(),
                lambda generation, seed, prefix: {
                    "6": {"inputs": {"text": "x"}}, "7": {"inputs": {"text": "y"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None,
                pose_fingerprint=lambda *_: "sha256:test")
            with mock.patch.object(ingest, "rate_image",
                                   return_value={"rating": {"general": 0.9}}):
                generate(path, services)
            safety_call = next(
                call for call in management.calls
                if call[0] == "PUT" and call[1].endswith("/safety"))
            self.assertEqual(safety_call[1], "/api/v1/generations/generation/safety")
            self.assertEqual(safety_call[2], {"rating": {"general": 0.9}})

    def test_generate_records_palette_from_measure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(base_request()), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            comfy.wait_for = lambda prompt_id: [{"filename": "render.png"}]
            state = StateFake({
                "idempotency_key": "fixed-key", "seeds": [42],
                "jobs": [{"idempotency_key": "job-key", "job_id": "job-id",
                          "comfy_prompt_id": "old-prompt", "status": "failed"}],
            })
            services = GenerateServices(
                management, comfy, state, RecordingNotifier(),
                lambda generation, seed, prefix: {
                    "6": {"inputs": {"text": "x"}}, "7": {"inputs": {"text": "y"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None,
                measure=lambda data: {"fig_sat_mean": 40.0, "light_sat": 20.0,
                                       "fails": ["mean saturation too high"]})
            generate(path, services)
            semantic_call = next(
                call for call in management.calls if call[0] == "semantic")
            palette = semantic_call[2]["attributes"]["palette"]
            self.assertIn("verdict", palette)
            self.assertTrue(palette["verdict"].startswith("FAIL"))

    def test_generate_survives_measure_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(base_request()), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            comfy.wait_for = lambda prompt_id: [{"filename": "render.png"}]
            state = StateFake({
                "idempotency_key": "fixed-key", "seeds": [42],
                "jobs": [{"idempotency_key": "job-key", "job_id": "job-id",
                          "comfy_prompt_id": "old-prompt", "status": "failed"}],
            })

            def broken_measure(data):
                raise ValueError("bad image")

            services = GenerateServices(
                management, comfy, state, RecordingNotifier(),
                lambda generation, seed, prefix: {
                    "6": {"inputs": {"text": "x"}}, "7": {"inputs": {"text": "y"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None,
                measure=broken_measure)
            generate(path, services)
            self.assertEqual(state.state["jobs"][0]["status"], "ingested")

    def test_generate_never_patches_experiment_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            request = base_request()
            request["experiment"] = {"experiment_id": "exp-1", "run_id": "run-1"}
            path.write_text(json.dumps(request), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            state = StateFake({
                "idempotency_key": "fixed-key", "seeds": [42],
                "jobs": [{"idempotency_key": "job-key", "job_id": "job-id",
                          "comfy_prompt_id": "old-prompt", "status": "ingested"}],
            })
            services = GenerateServices(
                management, comfy, state, NullNotifier(),
                lambda generation, seed, prefix: {"6": {"inputs": {"text": "x"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None)
            generate(path, services)
            self.assertFalse(any(
                call[1].startswith("/api/v1/experiment-runs/")
                for call in management.calls))

    def test_generate_probe_stops_before_request_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            patches = [{"target": "render.cfg", "op": "set", "value": 4.5,
                       "reason": "test"}]
            path.write_text(json.dumps(base_request(patches=patches)), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            state = StateFake(
                {"idempotency_key": "fixed-key", "seeds": [42], "jobs": []})

            def failing_builder(generation, seed, prefix):
                raise ValueError("needle absent")

            services = GenerateServices(
                management, comfy, state, NullNotifier(), failing_builder,
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None)
            with self.assertRaises(SystemExit):
                generate(path, services)
            self.assertFalse(
                any(call[1].startswith("/api/v1/requests") for call in management.calls))

    def test_generate_preset_probe_stops_before_request_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            request = base_request(
                presets=[{"kind": "pose", "name": "lounge", "version": 1}])
            path.write_text(json.dumps(request), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            state = StateFake(
                {"idempotency_key": "fixed-key", "seeds": [42], "jobs": []})

            def failing_builder(generation, seed, prefix):
                raise ValueError("needle absent")

            def fetch_preset(recipe, kind, name, version):
                return {"record": {"recipe_pose": "lounge-v1"},
                        "patches": [{"target": "render.cfg", "op": "set",
                                    "value": 4.5, "reason": "preset"}]}

            services = GenerateServices(
                management, comfy, state, NullNotifier(), failing_builder,
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None,
                presets=fetch_preset)
            with self.assertRaises(SystemExit):
                generate(path, services)
            self.assertFalse(
                any(call[1].startswith("/api/v1/requests") for call in management.calls))

    def test_generate_lint_waiver_suppresses_conflict_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            request = base_request(
                prompt="solid legwear", negative_prompt="sheer legwear",
                lint_waiver="deliberate for this arm")
            path.write_text(json.dumps(request), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            state = StateFake(
                {"idempotency_key": "fixed-key", "seeds": [42], "jobs": []})
            emits = []
            services = GenerateServices(
                management, comfy, state, NullNotifier(),
                lambda generation, seed, prefix: {
                    "6": {"inputs": {"text": "x"}}, "7": {"inputs": {"text": "y"}}},
                lambda: {"commit": "commit", "dirty": False},
                lambda positive, negative: [("solid", "sheer", "color")],
                Path(directory), emits.append)
            generate(path, services)
            self.assertTrue(
                any("deliberate for this arm" in message for message in emits))

    def test_generate_refuses_patches_without_a_fingerprint(self):
        request = base_request(patches=[
            {"target": "render.cfg", "op": "set", "value": 4.5,
             "reason": "test"}])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            management = ManagementFake()
            services = GenerateServices(
                management=management,
                comfyui=ComfyFake(),
                state=StateFake({}),
                notifier=NullNotifier(),
                graph_builder=lambda *_: {"3": {"inputs": {"seed": 0}}},
                git_metadata=lambda: {"commit": "c", "dirty": False},
                conflicts=lambda *_: [],
                output_root=Path(root) / "out",
                emit=lambda _: None,
                pose_fingerprint=lambda *_: None,
            )
            with self.assertRaises(SystemExit):
                generate(path, services)
        self.assertEqual(
            [call for call in management.calls if call[1].startswith("/api/v1/requests")],
            [])

    def test_generate_fails_before_rendering_when_identity_tags_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(base_request()), encoding="utf-8")
            management = ManagementFake()
            services = GenerateServices(
                management=management,
                comfyui=ComfyFake(),
                state=StateFake({}),
                notifier=NullNotifier(),
                graph_builder=lambda *_: {
                    "3": {"class_type": "KSampler",
                          "inputs": {"positive": ["6", 0], "negative": ["7", 0]}},
                    "6": {"inputs": {"text": "1girl, no identity here"}},
                    "7": {"inputs": {"text": "n"}}},
                git_metadata=lambda: {"commit": "c", "dirty": False},
                conflicts=lambda *_: [],
                output_root=Path(directory),
                emit=lambda _: None,
                identity_tags=lambda *_: frozenset({"purple eyes", "tareme"}),
            )
            with self.assertRaises(SystemExit) as ctx:
                generate(path, services)
            self.assertIn("purple eyes", str(ctx.exception))
            self.assertIn("identity_override", str(ctx.exception))
            self.assertFalse(
                any(call[1].startswith("/api/v1/requests") for call in management.calls))

    def test_generate_renders_and_records_identity_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(
                json.dumps(base_request(identity_override="deliberate crop")),
                encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            comfy.wait_for = lambda prompt_id: [{"filename": "render.png"}]
            state = StateFake({
                "idempotency_key": "fixed-key", "seeds": [42],
                "jobs": [{"idempotency_key": "job-key", "job_id": "job-id",
                          "comfy_prompt_id": "old-prompt", "status": "failed"}],
            })
            emits = []
            services = GenerateServices(
                management, comfy, state, RecordingNotifier(),
                lambda generation, seed, prefix: {
                    "3": {"class_type": "KSampler",
                          "inputs": {"positive": ["6", 0], "negative": ["7", 0]}},
                    "6": {"inputs": {"text": "1girl, no identity here"}},
                    "7": {"inputs": {"text": "y"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), emits.append,
                identity_tags=lambda *_: frozenset({"purple eyes", "tareme"}))
            generate(path, services)
            self.assertTrue(any("purple eyes" in message for message in emits))
            semantic_call = next(
                call for call in management.calls if call[0] == "semantic")
            self.assertEqual(
                semantic_call[2]["attributes"]["identity_override"],
                "deliberate crop")
            self.assertEqual(
                sorted(semantic_call[2]["attributes"]["identity_removed"]),
                ["purple eyes", "tareme"])

    def test_generate_does_not_record_an_unused_identity_override(self):
        # identity_override set defensively, but nothing was actually
        # removed -- it must not be stamped onto the semantic
        # attributes as if it had excused a real drop.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(
                json.dumps(base_request(identity_override="just in case")),
                encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            comfy.wait_for = lambda prompt_id: [{"filename": "render.png"}]
            state = StateFake({
                "idempotency_key": "fixed-key", "seeds": [42],
                "jobs": [{"idempotency_key": "job-key", "job_id": "job-id",
                          "comfy_prompt_id": "old-prompt", "status": "failed"}],
            })
            services = GenerateServices(
                management, comfy, state, RecordingNotifier(),
                lambda generation, seed, prefix: {
                    "3": {"class_type": "KSampler",
                          "inputs": {"positive": ["6", 0], "negative": ["7", 0]}},
                    "6": {"inputs": {"text": "1girl, purple eyes, tareme"}},
                    "7": {"inputs": {"text": "y"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None,
                identity_tags=lambda *_: frozenset({"purple eyes", "tareme"}))
            generate(path, services)
            semantic_call = next(
                call for call in management.calls if call[0] == "semantic")
            self.assertNotIn(
                "identity_override", semantic_call[2]["attributes"])
            self.assertNotIn(
                "identity_removed", semantic_call[2]["attributes"])

    def test_generate_skips_the_identity_guard_in_graph_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            request = {
                "schema_version": 1,
                "request": {"count": 1, "instruction": "test", "seeds": [42]},
                "generation": {"recipe": "yukari",
                              "graph": {"6": {"inputs": {"text": "x"}}}},
                "semantic": {"summary": "test arm"},
            }
            path.write_text(json.dumps(request), encoding="utf-8")

            def unexpected_identity_tags(*args):
                raise AssertionError("identity_tags must not be called in graph mode")

            management = ManagementFake()
            comfy = ComfyFake()
            state = StateFake({"idempotency_key": "fixed-key", "seeds": [42], "jobs": []})
            services = GenerateServices(
                management, comfy, state, NullNotifier(),
                lambda generation, seed, prefix: dict(generation["graph"]),
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None,
                identity_tags=unexpected_identity_tags)
            generate(path, services)

    def test_generate_reproduces_the_repair_crop_incident(self):
        """A patch that overwrites the identity block with a repair-crop
        prompt (the ChatGPT-via-MCP incident this guard exists for) must fail
        before rendering, and must render -- with the loss recorded -- once
        `identity_override` states why.
        """
        from chagama.interfaces.agent import (
            _build_generation_graph, _identity_tags,
        )

        strip_patch = [
            {"target": "prompt.positive.identity", "op": "remove",
             "old": ("light purple hair, short hair with long locks, very "
                     "long sidelocks, (dark greyish purple eyes:1.4), "
                     "(dark irises:1.2), hair ornament, "),
             "reason": "repair crop"},
            {"target": "prompt.positive.eyes", "op": "remove",
             "old": "(jitome:2.0), (tareme:1.2), (long eyes:1.25), ",
             "reason": "repair crop"},
        ]

        def _request(**extra):
            return {
                "schema_version": 1,
                "request": {"count": 1, "instruction": "test", "seeds": [7]},
                "generation": {"recipe": "yukari",
                              "parameters": {"pose": "bust"},
                              "patches": strip_patch, **extra},
                "semantic": {"summary": "test arm"},
            }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(_request()), encoding="utf-8")
            services = GenerateServices(
                management=ManagementFake(), comfyui=ComfyFake(),
                state=StateFake({}), notifier=NullNotifier(),
                graph_builder=_build_generation_graph,
                git_metadata=lambda: {"commit": "c", "dirty": False},
                conflicts=lambda *_: [], output_root=Path(directory),
                emit=lambda _: None,
                pose_fingerprint=lambda *_: "sha256:test",
                identity_tags=_identity_tags)
            with self.assertRaises(SystemExit) as ctx:
                generate(path, services)
            message = str(ctx.exception)
            for tag in ("dark greyish purple eyes", "light purple hair",
                       "short hair with long locks", "very long sidelocks",
                       "hair ornament", "jitome"):
                self.assertIn(tag, message)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(
                json.dumps(_request(identity_override="deliberate repair crop")),
                encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            comfy.wait_for = lambda prompt_id: [{"filename": "render.png"}]
            state = StateFake({
                "idempotency_key": "fixed-key", "seeds": [7],
                "jobs": [{"idempotency_key": "job-key", "job_id": "job-id",
                          "comfy_prompt_id": "old-prompt", "status": "failed"}],
            })
            services = GenerateServices(
                management=management, comfyui=comfy, state=state,
                notifier=RecordingNotifier(),
                graph_builder=_build_generation_graph,
                git_metadata=lambda: {"commit": "c", "dirty": False},
                conflicts=lambda *_: [], output_root=Path(directory),
                emit=lambda _: None,
                pose_fingerprint=lambda *_: "sha256:test",
                identity_tags=_identity_tags)
            generate(path, services)
            semantic_call = next(
                call for call in management.calls if call[0] == "semantic")
            attributes = semantic_call[2]["attributes"]
            self.assertTrue(
                {"dark greyish purple eyes", "jitome"}
                <= set(attributes["identity_removed"]))
            self.assertEqual(
                attributes["identity_override"], "deliberate repair crop")

    def test_generate_returns_generation_ids_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(base_request()), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            comfy.wait_for = lambda prompt_id: [{"filename": "render.png"}]
            state = StateFake({
                "idempotency_key": "fixed-key", "seeds": [42],
                "jobs": [{"idempotency_key": "job-key", "job_id": "job-id",
                          "comfy_prompt_id": "old-prompt", "status": "failed"}],
            })
            services = GenerateServices(
                management, comfy, state, RecordingNotifier(),
                lambda generation, seed, prefix: {
                    "6": {"inputs": {"text": "x"}}, "7": {"inputs": {"text": "y"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None)
            result = generate(path, services)
            self.assertEqual(result, {"generation_ids": ["generation"]})

    def test_generate_returns_none_on_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(base_request()), encoding="utf-8")
            management = ManagementFake()
            comfy = ComfyFake()
            state = StateFake({"idempotency_key": "fixed-key", "seeds": [42], "jobs": []})
            services = GenerateServices(
                management, comfy, state, NullNotifier(),
                lambda generation, seed, prefix: {"6": {"inputs": {"text": "x"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None)
            self.assertIsNone(generate(path, services, dry_run=True))

    def _generate_once(self, directory, *, state=None, request=None, **kwargs):
        path = Path(directory) / "request.json"
        path.write_text(json.dumps(request or base_request()), encoding="utf-8")
        management = ManagementFake()
        comfy = ComfyFake()
        comfy.wait_for = lambda prompt_id: [{"filename": "render.png"}]
        state = state or StateFake({"idempotency_key": "own-key", "jobs": []})
        services = GenerateServices(
            management, comfy, state, RecordingNotifier(),
            lambda generation, seed, prefix: {
                "6": {"inputs": {"text": "x"}}, "7": {"inputs": {"text": "y"}}},
            lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
            Path(directory), lambda message: None)
        generate(path, services, **kwargs)
        return management, comfy

    def test_worker_request_reports_resolution_then_creates_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            management, comfy = self._generate_once(
                directory, key_prefix="request:req-1", request_id="req-1")
            paths = [(call[0], call[1]) for call in management.calls
                     if call[0] in ("PUT", "POST") and "/requests" in call[1]]
            self.assertEqual(paths, [
                ("PUT", "/api/v1/requests/req-1/resolution"),
                ("POST", "/api/v1/requests/request-id/jobs")])
            resolution = management.calls[
                [call[1] for call in management.calls].index(
                    "/api/v1/requests/req-1/resolution")][2]
            self.assertEqual(resolution["recipe"], "yukari")
            self.assertEqual(resolution["git_commit"], "commit")
            self.assertNotIn("idempotency_key", resolution)
            self.assertNotIn("kind", resolution)
            self.assertEqual(len(comfy.submits), 1)

    def test_worker_request_keys_follow_the_request_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            management, _ = self._generate_once(
                directory, key_prefix="request:req-1", request_id="req-1")
            job_call = next(
                call for call in management.calls
                if call[0] == "POST" and call[1].endswith("/jobs"))
            self.assertEqual(job_call[2]["idempotency_key"], "request:req-1:job:0")
            self.assertNotIn("source_generation_id", job_call[2])
            generation_call = next(
                call for call in management.calls
                if call[0] == "POST" and call[1].endswith("/generations"))
            self.assertEqual(
                generation_call[3][0]["idempotency_key"], "request:req-1:job:0:gen:0")

    def test_standalone_run_creates_an_import_request(self):
        with tempfile.TemporaryDirectory() as directory:
            management, _ = self._generate_once(directory)
            create = next(
                call for call in management.calls
                if call[0] == "POST" and call[1] == "/api/v1/requests")
            self.assertEqual(create[2]["kind"], "import")
            self.assertEqual(create[2]["status"], "done")
            self.assertEqual(create[2]["idempotency_key"], "own-key")
            self.assertEqual(create[2]["recipe"], "yukari")
            self.assertFalse(any(call[0] == "PUT" for call in management.calls))
            job_call = next(
                call for call in management.calls
                if call[0] == "POST" and call[1].endswith("/jobs"))
            self.assertEqual(job_call[1], "/api/v1/requests/request-id/jobs")
            self.assertNotIn("source_generation_id", job_call[2])

    def test_standalone_experiment_run_links_its_run_on_the_import_request(self):
        request = {**base_request(),
                   "experiment": {"experiment_id": "exp-1", "run_id": "run-1"}}
        with tempfile.TemporaryDirectory() as directory:
            management, _ = self._generate_once(directory, request=request)
            create = next(
                call for call in management.calls
                if call[0] == "POST" and call[1] == "/api/v1/requests")
            self.assertEqual(create[2]["run_id"], "run-1")
            self.assertNotIn("experiment", create[2])
            self.assertFalse(any(
                call[1].startswith("/api/v1/experiment-runs")
                for call in management.calls))

    def test_worker_request_does_not_send_run_id(self):
        request = {**base_request(),
                   "experiment": {"experiment_id": "exp-1", "run_id": "run-1"}}
        with tempfile.TemporaryDirectory() as directory:
            management, _ = self._generate_once(
                directory, request=request,
                key_prefix="request:req-1", request_id="req-1")
            put = next(call for call in management.calls if call[0] == "PUT")
            self.assertNotIn("run_id", put[2])

    def test_output_directory_and_comfy_prefix_use_the_request_short_id(self):
        with tempfile.TemporaryDirectory() as directory:
            _, comfy = self._generate_once(directory)
            self.assertTrue((Path(directory) / "req").is_dir())

    def test_resend_jobs_adopts_seeds_when_state_has_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            request = base_request()
            del request["request"]["seeds"]
            request["request"]["count"] = 2
            path.write_text(json.dumps(request), encoding="utf-8")

            management = ManagementFake()
            management.resolution_jobs = [
                {"id": "job-0", "index": 0, "seed": 111, "status": "ingested",
                 "comfy_prompt_id": "old-prompt",
                 "generations": [{"id": "gen-0", "comfy_output_index": 0}]},
                {"id": "job-1", "index": 1, "seed": 222, "status": "queued"},
            ]
            comfy = ComfyFake()
            state = StateFake({"idempotency_key": "fixed-key", "jobs": []})
            services = GenerateServices(
                management, comfy, state, NullNotifier(),
                lambda generation, seed, prefix: {"6": {"inputs": {"text": "x"}}},
                lambda: {"commit": "commit", "dirty": False}, lambda *_: [],
                Path(directory), lambda message: None)
            generate(path, services, key_prefix="request:req-2",
                     request_id="req-2")

            self.assertEqual(state.state["seeds"], [111, 222])
            self.assertEqual(state.state["jobs"][0]["status"], "ingested")
            self.assertEqual(state.state["jobs"][0]["job_id"], "job-0")
            self.assertEqual(
                state.state["jobs"][0]["generations"],
                [{"id": "gen-0", "comfy_output_index": 0, "status": "registered",
                  "idempotency_key": "request:req-2:job:0:gen:0"}])
            # job 0 is already ingested, so only job 1 (the resent "queued"
            # one) is actually submitted to ComfyUI this run.
            self.assertEqual(len(comfy.submits), 1)


if __name__ == "__main__":
    unittest.main()
