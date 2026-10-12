"""Tests for the resident requests-queue worker.

All collaborators are fakes: this suite never opens a network socket and
never calls the real use cases (those are covered by test_generate_application.py,
test_redraw_application.py, test_deliver_application.py and the repair tests).
"""

from __future__ import annotations

import contextlib
import dataclasses
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from comfyui_recipes.application.generate import GenerateServices
from comfyui_recipes.application.work import (
    WorkServices,
    execute,
    release_claims,
    work,
    work_once,
)
from comfyui_recipes.application.worker_channels import Heartbeat, HubListener, ProgressRelay
from comfyui_recipes.interfaces.agent import DEFAULT_KINDS


class ManagementFake:
    def __init__(self, claim_responses=None, dry_run_items=None,
                context=None):
        self.calls = []
        self.claim_responses = list(claim_responses or [])
        self.claim_error = None
        self.dry_run_items = dry_run_items or []
        self.running_items = []
        # A redraw/deliver/repair/masked_redraw row's dial resolution fetches the
        # source generation's context for `request.recipe`.
        self.context = context if context is not None else {
            "request": {"id": "request-1", "recipe": "yukari"}, "generations": []}

    def request(self, method, path, payload=None, multipart=None):
        self.calls.append((method, path, payload, multipart))
        if method == "POST" and path == "/api/v1/requests/claim":
            if self.claim_error is not None:
                raise self.claim_error
            if self.claim_responses:
                next_item = self.claim_responses.pop(0)
                if isinstance(next_item, BaseException):
                    raise next_item
                return next_item
            return None
        if method == "GET" and path.startswith("/api/v1/requests?status=queued"):
            return {"items": self.dry_run_items}
        if method == "GET" and path.startswith("/api/v1/requests?status=running"):
            return {"items": self.running_items}
        if method == "PATCH" and path.startswith("/api/v1/requests/"):
            return {}
        if method == "GET" and path.endswith("/context"):
            return self.context
        raise AssertionError(f"unexpected management call: {method} {path}")


class FakeHubConnection:
    """A scripted Connection: pops dicts/exceptions from `script`, else idles.

    An empty script makes recv() behave like a real timeout -- it sleeps a
    short, bounded time and returns None -- so a listener with nothing left
    to read does not busy-loop.
    """

    def __init__(self, script=None, idle_timeout_cap=0.05):
        self.script = list(script or [])
        self.sent = []
        self.closed = False
        self.idle_timeout_cap = idle_timeout_cap

    def send(self, message):
        self.sent.append(message)

    def recv(self, timeout):
        if self.closed:
            raise ConnectionError("closed")
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        time.sleep(min(timeout, self.idle_timeout_cap) if timeout else self.idle_timeout_cap)
        return None

    def close(self):
        self.closed = True


class GatedHubConnection:
    """Like FakeHubConnection, but the message after `pre_script` is held back
    until the test calls `release.set()` -- lets a test synchronize with an
    in-progress `wake.wait()` before delivering the message that ends it.
    """

    def __init__(self, pre_script, gated_message):
        self.pre_script = list(pre_script)
        self.gated_message = gated_message
        self.release = threading.Event()
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def recv(self, timeout):
        if self.pre_script:
            item = self.pre_script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        if self.release.wait(timeout=min(timeout, 2) if timeout else 2):
            self.release.clear()
            return self.gated_message
        return None

    def close(self):
        self.closed = True


class HubFactory:
    """A `services.hub`/`services.progress_feed` factory: one connection (or
    exception) per call, consumed in order.
    """

    def __init__(self, connections):
        self.connections = list(connections)

    def __call__(self):
        next_item = self.connections.pop(0)
        if isinstance(next_item, BaseException):
            raise next_item
        return next_item


class RecordingHeartbeat:
    def __init__(self, management, row_id, worker_id, *, interval=30, emit=print):
        self.management = management
        self.row_id = row_id
        self.worker_id = worker_id
        self.interval = interval
        self.emit = emit
        self.events = []

    def __enter__(self):
        self.events.append("start")
        return self

    def __exit__(self, *exc_info):
        self.events.append("stop")
        return False


def make_generate_services(directory: Path, **overrides) -> GenerateServices:
    defaults = dict(
        management=None, comfyui=None, state=None, notifier=None,
        graph_builder=lambda generation, seed, prefix: {},
        git_metadata=lambda: {"commit": "c", "dirty": False},
        conflicts=lambda *_: [], output_root=directory, measure=None,
        emit=lambda message: None,
    )
    defaults.update(overrides)
    return GenerateServices(**defaults)


def make_services(directory: Path, management, *, heartbeats=None,
                  generate=None,
                  deliver=None, deliver_services=None,
                  redraw=None, redraw_services=None,
                  repair=None, repair_services=None,
                  masked_redraw=None, masked_redraw_services=None,
                  branch="dev/requests-worker", emit=None, sleep=None,
                  kinds=("generate", "deliver")) -> WorkServices:
    kwargs = dict(
        management=management,
        generate_services=make_generate_services(Path(directory)),
        redraw_services=(redraw_services if redraw_services is not None
                         else "redraw-services"),
        deliver_services=(deliver_services if deliver_services is not None
                          else "deliver-services"),
        repair_services=(repair_services if repair_services is not None
                         else "repair-services"),
        masked_redraw_services=(masked_redraw_services
                                if masked_redraw_services is not None
                                else "masked-redraw-services"),
        dof_services="dof-services",
        lens_services="lens-services",
        photo_services="photo-services",
        git_metadata=lambda: {"branch": branch},
        worker_id="test-worker",
        emit=(emit or (lambda message: None)),
        kinds=kinds,
    )
    if heartbeats is not None:
        def factory(management, row_id, worker_id, *, interval=30, emit=print):
            instance = RecordingHeartbeat(
                management, row_id, worker_id, interval=interval, emit=emit)
            heartbeats.append(instance)
            return instance
        kwargs["heartbeat"] = factory
    if deliver is not None:
        kwargs["deliver"] = deliver
    if redraw is not None:
        kwargs["redraw"] = redraw
    if generate is not None:
        kwargs["generate"] = generate
    if repair is not None:
        kwargs["repair"] = repair
    if masked_redraw is not None:
        kwargs["masked_redraw"] = masked_redraw
    if sleep is not None:
        kwargs["sleep"] = sleep
    return WorkServices(**kwargs)


def make_hub_services(directory: Path, *, hub=None, progress_feed=None,
                      sleep=None, emit=None, ping_interval=0.05, backoff_max=1,
                      clock=None, kinds=("generate", "deliver")) -> WorkServices:
    kwargs = dict(
        management=ManagementFake(),
        generate_services=make_generate_services(Path(directory)),
        redraw_services="redraw-services",
        deliver_services="deliver-services",
        repair_services="repair-services",
        masked_redraw_services="masked-redraw-services",
        dof_services="dof-services",
        lens_services="lens-services",
        photo_services="photo-services",
        git_metadata=lambda: {"branch": "dev/requests-worker"},
        worker_id="test-worker",
        emit=(emit or (lambda message: None)),
        kinds=kinds,
        hub=hub,
        progress_feed=progress_feed,
        ping_interval=ping_interval,
        backoff_max=backoff_max,
    )
    if sleep is not None:
        kwargs["sleep"] = sleep
    if clock is not None:
        kwargs["clock"] = clock
    return WorkServices(**kwargs)


def repair_row(**overrides):
    row = {
        "id": "req-3", "kind": "repair", "status": "running",
        "recipe_ref": "dev/requests-worker", "run_id": None, "attempt": 1,
        "payload": {"generation_id": "gen-1", "options": {}},
    }
    row.update(overrides)
    return row


def masked_redraw_row(**overrides):
    row = {
        "id": "req-4", "kind": "masked_redraw", "status": "running",
        "recipe_ref": "dev/requests-worker", "run_id": None, "attempt": 1,
        "payload": {"generation_id": "gen-1", "options": {
            "regions": [[0.1, 0.1, 0.5, 0.5]], "prompt_patch": "a dress"}},
    }
    row.update(overrides)
    return row


def generate_row(**overrides):
    row = {
        "id": "req-1", "kind": "generate", "status": "running",
        "recipe_ref": "dev/requests-worker", "run_id": None, "attempt": 1,
        "payload": {"schema_version": 1, "request": {"count": 1},
                    "generation": {"recipe": "yukari"}, "semantic": {}},
    }
    row.update(overrides)
    return row


def redraw_row(**overrides):
    row = {
        "id": "req-6", "kind": "redraw", "status": "running",
        "recipe_ref": "dev/requests-worker", "run_id": None, "attempt": 1,
        "payload": {"generation_id": "gen-1", "options": {"method": "canvas"}},
    }
    row.update(overrides)
    return row


def deliver_row(**overrides):
    row = {
        "id": "req-5", "kind": "deliver", "status": "running",
        "recipe_ref": "dev/requests-worker", "run_id": None, "attempt": 1,
        "payload": {"generation_id": "gen-1", "options": {}},
    }
    row.update(overrides)
    return row


DEFAULT_OUTLINES = [{"color": "#ffffff", "width": 0.8}, {"color": "#885b80", "width": 1.04}]


def dof_row(**overrides):
    row = {
        "id": "req-7", "kind": "dof", "status": "running",
        "recipe_ref": "dev/requests-worker", "run_id": None, "attempt": 1,
        "payload": {"generation_id": "gen-1", "options": {"focus": [0.5, 0.4]}},
    }
    row.update(overrides)
    return row


def lens_row(**overrides):
    row = {
        "id": "req-8", "kind": "lens", "status": "running",
        "recipe_ref": "dev/requests-worker", "run_id": None, "attempt": 1,
        "payload": {"generation_id": "gen-1", "options": {"strength": -0.5}},
    }
    row.update(overrides)
    return row


def photo_row(**overrides):
    row = {
        "id": "req-9", "kind": "photo", "status": "running",
        "recipe_ref": "dev/requests-worker", "run_id": None, "attempt": 1,
        "payload": {"generation_id": "gen-1",
                    "options": {"photo": "pic-1", "stand_at": [0.8, 0.4]}},
    }
    row.update(overrides)
    return row


def finalize_row(**overrides):
    row = {
        "id": "req-2", "kind": "finalize", "status": "running",
        "recipe_ref": "dev/requests-worker", "run_id": None, "attempt": 1,
        "payload": {"generation_id": "gen-1", "options": {}},
    }
    row.update(overrides)
    return row


class ExecuteTest(unittest.TestCase):
    def test_recipe_ref_mismatch_fails_without_executing(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            services = make_services(
                directory, ManagementFake(),
                generate=lambda *a, **k: calls.append((a, k)))
            row = generate_row(recipe_ref="other-branch")
            with self.assertRaises(SystemExit) as ctx:
                execute(services, row)
            self.assertEqual(str(ctx.exception), "recipe_ref not served: other-branch")
            self.assertEqual(calls, [])

    def test_generate_kind_writes_the_request_file_and_derives_the_key_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def fake_generate(path, generate_services, *, key_prefix=None, **kwargs):
                calls.append((path, key_prefix, kwargs))
                return {"generation_ids": ["g1"]}

            services = make_services(directory, ManagementFake(), generate=fake_generate)
            result = execute(services, generate_row())
            self.assertEqual(result, {"generation_ids": ["g1"]})
            self.assertEqual(len(calls), 1)
            path, key_prefix, kwargs = calls[0]
            self.assertEqual(kwargs["request_id"], "req-1")
            self.assertEqual(path, Path(directory) / "requests" / "req-1.json")
            self.assertTrue(path.exists())
            self.assertEqual(key_prefix, "request:req-1")

    def test_redraw_kind_maps_options_and_uses_the_redraw_services(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def fake_redraw(generation_id, redraw_services, **kwargs):
                calls.append((generation_id, redraw_services, kwargs))
                return {"generation_ids": ["g2"]}

            services = make_services(
                directory, ManagementFake(), redraw=fake_redraw,
                redraw_services="redraw-services-sentinel")
            row = redraw_row(payload={
                "generation_id": "gen-1",
                "options": {"method": "canvas", "denoise": "keep", "size": 2048}})
            result = execute(services, row)
            self.assertEqual(result, {
                "generation_ids": ["g2"],
                "resolved_options": {"method": "canvas", "denoise": 0.4,
                                     "size": 2048}})
            generation_id, redraw_services, kwargs = calls[0]
            self.assertEqual(generation_id, "gen-1")
            self.assertEqual(redraw_services, "redraw-services-sentinel")
            self.assertEqual(kwargs["method"], "canvas")
            self.assertEqual(kwargs["request_id"], "req-6")
            self.assertEqual(kwargs["key_prefix"], "request:req-6")
            self.assertEqual(kwargs["denoise"], 0.4)
            self.assertIn("context", kwargs)

    def test_redraw_kind_rejects_bad_options_and_a_missing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            services = make_services(
                directory, ManagementFake(),
                redraw=lambda *a, **k: calls.append((a, k)))
            for payload in (
                    {"generation_id": "gen-1", "options": {}},
                    {"generation_id": "gen-1"},
                    {"generation_id": "gen-1",
                     "options": {"method": "light", "scene": "moon", "size": 1}},
                    {"options": {"method": "canvas"}}):
                with self.subTest(payload=payload), self.assertRaises(SystemExit):
                    execute(services, redraw_row(payload=payload))
            self.assertEqual(calls, [])

    def test_deliver_kind_maps_options_and_uses_the_deliver_services(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def fake_deliver(generation_id, deliver_services, **kwargs):
                calls.append((generation_id, deliver_services, kwargs))
                return {"generation_ids": ["g2"]}

            services = make_services(
                directory, ManagementFake(), deliver=fake_deliver,
                deliver_services="deliver-services-sentinel")
            row = deliver_row(payload={
                "generation_id": "gen-1",
                "options": {"keep_legwear": True, "transparent": True}})
            result = execute(services, row)
            self.assertEqual(result, {
                "generation_ids": ["g2"],
                "resolved_options": {"keep_legwear": 0.62, "transparent": True,
                                     "outlines": DEFAULT_OUTLINES}})
            generation_id, deliver_services, kwargs = calls[0]
            self.assertEqual(generation_id, "gen-1")
            self.assertEqual(deliver_services, "deliver-services-sentinel")
            self.assertEqual(kwargs["request_id"], "req-5")
            self.assertEqual(kwargs["key_prefix"], "request:req-5")
            self.assertIs(kwargs["repin"], True)
            self.assertIs(kwargs["transparent"], True)
            self.assertIsNone(kwargs["backdrop"])
            self.assertIn("context", kwargs)

    def test_deliver_kind_rejects_redraw_options_and_a_missing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            services = make_services(
                directory, ManagementFake(),
                deliver=lambda *a, **k: calls.append((a, k)))
            with self.assertRaisesRegex(SystemExit, "denoise"):
                execute(services, deliver_row(payload={
                    "generation_id": "gen-1", "options": {"denoise": 0.4}}))
            with self.assertRaises(SystemExit):
                execute(services, deliver_row(payload={"options": {}}))
            self.assertEqual(calls, [])

    def test_deliver_payload_ignores_extra_keys_such_as_chimeras_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            services = make_services(
                directory, ManagementFake(),
                deliver=lambda generation_id, deliver_services, **kwargs: {
                    "generation_ids": ["g2"]})
            row = deliver_row(payload={
                "generation_id": "gen-1", "options": {},
                "profile": {"name": "cinema-tidy", "version": 3},
            })
            result = execute(services, row)
            self.assertEqual(result["generation_ids"], ["g2"])

    def test_deliver_resolves_words_against_the_source_requests_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            services = make_services(
                directory, ManagementFake(),
                deliver=lambda generation_id, deliver_services, **kwargs: {
                    "generation_ids": ["g2"]})
            row = deliver_row(payload={
                "generation_id": "gen-1",
                "options": {"repin": True, "keep_legwear": True}})
            result = execute(services, row)
            self.assertEqual(result["resolved_options"],
                             {"repin": True, "keep_legwear": 0.62,
                              "outlines": DEFAULT_OUTLINES})

    def test_a_word_unknown_to_the_source_recipe_fails_the_request(self):
        with tempfile.TemporaryDirectory() as directory:
            services = make_services(
                directory, ManagementFake(),
                redraw=lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("redraw must not run")))
            row = redraw_row(payload={
                "generation_id": "gen-1",
                "options": {"method": "canvas", "denoise": "redraw"}})
            with self.assertRaisesRegex(SystemExit, "denoise"):
                execute(services, row)

    def test_dof_kind_maps_options_and_uses_the_dof_services(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def fake_dof(generation_id, dof_services, **kwargs):
                calls.append((generation_id, dof_services, kwargs))
                return {"generation_ids": ["g3"]}

            services = make_services(directory, ManagementFake())
            services = dataclasses.replace(services, dof=fake_dof)
            result = execute(services, dof_row())
            self.assertEqual(result["resolved_options"], {
                "focus": [0.5, 0.4], "f_number": 2.8, "viewfinder": "off",
                "scope": {"figure": True, "outline": True, "backdrop": True}})
            generation_id, dof_services, kwargs = calls[0]
            self.assertEqual((generation_id, dof_services), ("gen-1", "dof-services"))
            self.assertEqual(kwargs["request_id"], "req-7")
            self.assertEqual(kwargs["focus"], (0.5, 0.4))

    def test_dof_kind_rejects_bad_options_and_a_missing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            services = dataclasses.replace(
                make_services(directory, ManagementFake()),
                dof=lambda *a, **k: calls.append((a, k)))
            with self.assertRaisesRegex(SystemExit, "focus"):
                execute(services, dof_row(payload={"generation_id": "gen-1",
                                                   "options": {}}))
            with self.assertRaisesRegex(SystemExit, "generation_id"):
                execute(services, dof_row(payload={"options": {"focus": [0, 0]}}))
            self.assertEqual(calls, [])

    def test_lens_kind_maps_options_and_uses_the_lens_services(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def fake_lens(generation_id, lens_services, **kwargs):
                calls.append((generation_id, lens_services, kwargs))
                return {"generation_ids": ["g4"]}

            services = dataclasses.replace(
                make_services(directory, ManagementFake()), lens=fake_lens)
            result = execute(services, lens_row())
            self.assertEqual(result["resolved_options"],
                             {"center": [0.5, 0.5], "strength": -0.5,
                              "shape": "fisheye"})
            generation_id, lens_services, kwargs = calls[0]
            self.assertEqual((generation_id, lens_services), ("gen-1", "lens-services"))
            self.assertEqual(kwargs["request_id"], "req-8")
            self.assertEqual(kwargs["center"], (0.5, 0.5))

    def test_lens_kind_rejects_bad_options_and_a_missing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            services = dataclasses.replace(
                make_services(directory, ManagementFake()),
                lens=lambda *a, **k: calls.append((a, k)))
            with self.assertRaisesRegex(SystemExit, "strength"):
                execute(services, lens_row(payload={"generation_id": "gen-1",
                                                    "options": {"strength": 2}}))
            with self.assertRaisesRegex(SystemExit, "generation_id"):
                execute(services, lens_row(payload={"options": {}}))
            self.assertEqual(calls, [])

    def test_photo_kind_maps_options_and_uses_the_photo_services(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def fake_photo(generation_id, photo_services, **kwargs):
                calls.append((generation_id, photo_services, kwargs))
                return {"generation_ids": ["g5"]}

            services = dataclasses.replace(
                make_services(directory, ManagementFake()), photo=fake_photo)
            result = execute(services, photo_row())
            self.assertEqual(result["resolved_options"], {
                "photo": "pic-1", "place": [0.5, 0.95], "height": 0.7,
                "stand_at": [0.8, 0.4], "nearer": 0.0, "focus": "subject",
                "f_number": 2.8})
            generation_id, photo_services, kwargs = calls[0]
            self.assertEqual((generation_id, photo_services), ("gen-1", "photo-services"))
            self.assertEqual((kwargs["photo_id"], kwargs["request_id"]), ("pic-1", "req-9"))

    def test_photo_kind_rejects_bad_options_and_a_missing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            services = dataclasses.replace(
                make_services(directory, ManagementFake()),
                photo=lambda *a, **k: calls.append((a, k)))
            with self.assertRaisesRegex(SystemExit, "photo"):
                execute(services, photo_row(payload={"generation_id": "gen-1",
                                                     "options": {}}))
            with self.assertRaisesRegex(SystemExit, "generation_id"):
                execute(services, photo_row(payload={"options": {"photo": "p"}}))
            self.assertEqual(calls, [])

    def test_finalize_requests_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            services = make_services(directory, ManagementFake())
            with self.assertRaisesRegex(SystemExit, "no longer supported"):
                execute(services, finalize_row())

    def test_the_default_kinds_are_exactly_the_eight(self):
        self.assertEqual(
            WorkServices.__dataclass_fields__["kinds"].default,
            ("generate", "redraw", "repair", "masked_redraw", "deliver", "dof",
             "lens", "photo"))
        self.assertEqual(
            DEFAULT_KINDS,
            ("generate", "redraw", "repair", "masked_redraw", "deliver", "dof",
             "lens", "photo"))

    def test_unsupported_kind_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            services = make_services(directory, ManagementFake())
            with self.assertRaises(SystemExit):
                execute(services, generate_row(kind="probe"))

    def test_repair_kind_maps_options_and_uses_the_configured_services(self):
        with tempfile.TemporaryDirectory() as directory:
            repair_calls = []

            def fake_repair(generation_id, repair_services, **kwargs):
                repair_calls.append((generation_id, repair_services, kwargs))
                return {"generation_ids": ["g3"]}

            services = make_services(
                directory, ManagementFake(), repair=fake_repair,
                repair_services="repair-services-sentinel")
            row = repair_row(payload={
                "generation_id": "gen-1",
                "options": {"parts": ["feet"], "seeds": [7]},
            })
            result = execute(services, row)
            self.assertEqual(result, {
                "generation_ids": ["g3"],
                "resolved_options": {"parts": ["feet"], "seeds": [7]},
            })
            self.assertEqual(repair_calls[0][0], "gen-1")
            self.assertEqual(repair_calls[0][1], "repair-services-sentinel")
            self.assertEqual(repair_calls[0][2]["parts"], ["feet"])
            self.assertEqual(repair_calls[0][2]["seeds"], [7])
            self.assertIn("context", repair_calls[0][2])
            self.assertEqual(repair_calls[0][2]["request_id"], "req-3")

    def test_repair_kind_with_bad_options_fails_before_repairing(self):
        with tempfile.TemporaryDirectory() as directory:
            repair_calls = []
            services = make_services(
                directory, ManagementFake(),
                repair=lambda *a, **k: repair_calls.append((a, k)))
            row = repair_row(payload={
                "generation_id": "gen-1", "options": {"nope": True}})
            with self.assertRaises(SystemExit) as ctx:
                execute(services, row)
            self.assertIn("nope", str(ctx.exception))
            self.assertEqual(repair_calls, [])

    def test_repair_kind_without_generation_id_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            services = make_services(directory, ManagementFake())
            row = repair_row(payload={"options": {}})
            with self.assertRaises(SystemExit):
                execute(services, row)

    def test_masked_redraw_kind_maps_options_and_uses_the_configured_services(self):
        with tempfile.TemporaryDirectory() as directory:
            masked_redraw_calls = []

            def fake_masked_redraw(generation_id, masked_redraw_services, **kwargs):
                masked_redraw_calls.append((generation_id, masked_redraw_services, kwargs))
                return {"generation_ids": ["g4"]}

            services = make_services(
                directory, ManagementFake(), masked_redraw=fake_masked_redraw,
                masked_redraw_services="masked-redraw-services-sentinel")
            row = masked_redraw_row(payload={
                "generation_id": "gen-1",
                "options": {"regions": [[0.1, 0.1, 0.5, 0.5]],
                            "prompt_patch": "a dress", "seeds": [7]},
            })
            result = execute(services, row)
            self.assertEqual(result, {
                "generation_ids": ["g4"],
                "resolved_options": {"regions": [[0.1, 0.1, 0.5, 0.5]],
                                     "prompt_patch": "a dress", "seeds": [7]},
            })
            self.assertEqual(masked_redraw_calls[0][0], "gen-1")
            self.assertEqual(masked_redraw_calls[0][1], "masked-redraw-services-sentinel")
            self.assertEqual(masked_redraw_calls[0][2]["regions"], [[0.1, 0.1, 0.5, 0.5]])
            self.assertEqual(masked_redraw_calls[0][2]["prompt_patch"], "a dress")
            self.assertEqual(masked_redraw_calls[0][2]["seeds"], [7])
            self.assertIn("context", masked_redraw_calls[0][2])
            self.assertEqual(masked_redraw_calls[0][2]["request_id"], "req-4")

    def test_masked_redraw_kind_with_bad_options_fails_before_running(self):
        with tempfile.TemporaryDirectory() as directory:
            masked_redraw_calls = []
            services = make_services(
                directory, ManagementFake(),
                masked_redraw=lambda *a, **k: masked_redraw_calls.append((a, k)))
            row = masked_redraw_row(payload={
                "generation_id": "gen-1", "options": {"nope": True}})
            with self.assertRaises(SystemExit) as ctx:
                execute(services, row)
            self.assertIn("nope", str(ctx.exception))
            self.assertEqual(masked_redraw_calls, [])

    def test_masked_redraw_kind_without_generation_id_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            services = make_services(directory, ManagementFake())
            row = masked_redraw_row(payload={
                "options": {"regions": [[0.1, 0.1, 0.5, 0.5]], "prompt_patch": "a dress"}})
            with self.assertRaises(SystemExit):
                execute(services, row)


class WorkOnceTest(unittest.TestCase):
    def test_no_queued_row_returns_false_and_makes_no_further_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[None])
            services = make_services(directory, management)
            self.assertFalse(work_once(services))
            self.assertEqual(management.calls, [
                ("POST", "/api/v1/requests/claim",
                 {"worker_id": "test-worker", "kinds": ["generate", "deliver"]}, None)])

    def test_done_row_patches_status_done_with_the_result(self):
        with tempfile.TemporaryDirectory() as directory:
            row = generate_row()
            management = ManagementFake(claim_responses=[row])

            def fake_generate(path, generate_services, *, key_prefix=None, **kwargs):
                return {"generation_ids": ["g1"]}

            services = make_services(directory, management, generate=fake_generate)
            self.assertTrue(work_once(services))
            patch_call = next(
                call for call in management.calls
                if call[0] == "PATCH" and call[1] == "/api/v1/requests/req-1")
            self.assertEqual(
                patch_call[2],
                {"status": "done", "result": {"generation_ids": ["g1"]},
                 "worker_id": "test-worker"})

    def test_failed_row_patches_status_failed_with_the_error_message(self):
        with tempfile.TemporaryDirectory() as directory:
            row = generate_row()
            management = ManagementFake(claim_responses=[row])

            def fake_generate(path, generate_services, *, key_prefix=None, **kwargs):
                raise RuntimeError("boom")

            services = make_services(directory, management, generate=fake_generate)
            self.assertTrue(work_once(services))
            patch_call = next(
                call for call in management.calls
                if call[0] == "PATCH" and call[1] == "/api/v1/requests/req-1")
            self.assertEqual(patch_call[2], {"status": "failed", "error": "boom",
                                             "worker_id": "test-worker"})

    def test_recipe_ref_mismatch_reports_failed_with_the_exact_message(self):
        with tempfile.TemporaryDirectory() as directory:
            row = generate_row(recipe_ref="other-branch")
            management = ManagementFake(claim_responses=[row])
            calls = []
            services = make_services(
                directory, management, generate=lambda *a, **k: calls.append((a, k)))
            self.assertTrue(work_once(services))
            self.assertEqual(calls, [])
            patch_call = next(
                call for call in management.calls
                if call[0] == "PATCH" and call[1] == "/api/v1/requests/req-1")
            self.assertEqual(
                patch_call[2],
                {"status": "failed", "error": "recipe_ref not served: other-branch",
                 "worker_id": "test-worker"})

    def test_heartbeat_is_started_and_stopped_around_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            row = generate_row()
            management = ManagementFake(claim_responses=[row])
            heartbeats = []

            def fake_generate(path, generate_services, *, key_prefix=None, **kwargs):
                self.assertEqual(heartbeats[0].events, ["start"])
                return {"generation_ids": []}

            services = make_services(
                directory, management, heartbeats=heartbeats, generate=fake_generate)
            work_once(services)
            self.assertEqual(len(heartbeats), 1)
            self.assertEqual(heartbeats[0].row_id, "req-1")
            self.assertEqual(heartbeats[0].events, ["start", "stop"])

    def test_keep_awake_is_held_around_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[generate_row()])
            events = []

            @contextlib.contextmanager
            def keep_awake():
                events.append("enter")
                try:
                    yield
                finally:
                    events.append("exit")

            def fake_generate(path, generate_services, *, key_prefix=None, **kwargs):
                events.append("generate")
                return {"generation_ids": []}

            services = make_services(directory, management, generate=fake_generate)
            services = dataclasses.replace(services, keep_awake=keep_awake)
            work_once(services)
            self.assertEqual(events, ["enter", "generate", "exit"])

    def test_keyboard_interrupt_from_the_executor_propagates_and_marks_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            row = generate_row()
            management = ManagementFake(claim_responses=[row])
            heartbeats = []

            def fake_generate(path, generate_services, *, key_prefix=None, **kwargs):
                raise KeyboardInterrupt

            services = make_services(
                directory, management, heartbeats=heartbeats, generate=fake_generate)
            with self.assertRaises(KeyboardInterrupt):
                work_once(services)
            self.assertFalse(any(call[0] == "PATCH" for call in management.calls))
            self.assertEqual(heartbeats[0].events, ["start", "stop"])

    def test_claim_failure_is_emitted_and_does_not_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake()
            management.claim_error = SystemExit("403 forbidden")
            messages = []
            services = make_services(directory, management, emit=messages.append)
            self.assertFalse(work_once(services))
            self.assertTrue(any("403" in message for message in messages))

    def test_dry_run_does_not_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(dry_run_items=[generate_row()])
            services = make_services(directory, management)
            self.assertFalse(work_once(services, dry_run=True))
            self.assertFalse(
                any(call[1] == "/api/v1/requests/claim" for call in management.calls))
            self.assertTrue(
                any(call[1].startswith("/api/v1/requests?status=queued")
                    for call in management.calls))


class WorkLoopTest(unittest.TestCase):
    def test_once_claims_exactly_once_and_never_sleeps(self):
        with tempfile.TemporaryDirectory() as directory:
            row = generate_row()
            management = ManagementFake(claim_responses=[row])

            def fake_generate(path, generate_services, *, key_prefix=None, **kwargs):
                return {"generation_ids": []}

            def no_sleep(seconds):
                raise AssertionError("must not sleep when --once")

            services = make_services(
                directory, management, generate=fake_generate, sleep=no_sleep)
            work(services, once=True)
            claim_calls = [call for call in management.calls
                          if call[1] == "/api/v1/requests/claim"]
            self.assertEqual(len(claim_calls), 1)

    def test_sleeps_only_when_nothing_was_claimed(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[None])
            sleeps = []

            def sleep_then_stop(seconds):
                sleeps.append(seconds)
                raise KeyboardInterrupt

            services = make_services(directory, management, sleep=sleep_then_stop)
            work(services, once=False)  # must not raise
            self.assertEqual(sleeps, [30])

    def test_keyboard_interrupt_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[None])
            messages = []

            def interrupt(seconds):
                raise KeyboardInterrupt

            services = make_services(
                directory, management, sleep=interrupt, emit=messages.append)
            work(services, once=False)
            self.assertTrue(any("stopped" in message for message in messages))

    def test_keyboard_interrupt_releases_claims_after_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[None])

            def interrupt(seconds):
                raise KeyboardInterrupt

            services = make_services(directory, management, sleep=interrupt)
            with patch("comfyui_recipes.application.work.release_claims") as release:
                work(services, once=False, publish_catalog=False)
            self.assertEqual(release.call_count, 2)  # startup cleanup + shutdown cleanup


class PublishCatalogAtStartupTest(unittest.TestCase):
    def test_failing_catalog_publish_does_not_prevent_work_from_starting(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[None])
            messages = []

            def interrupt(seconds):
                raise KeyboardInterrupt

            services = make_services(
                directory, management, sleep=interrupt, emit=messages.append)
            work(services, once=False)  # must not raise
            self.assertTrue(
                any("catalog publish failed" in message for message in messages))

    def test_successful_publish_puts_to_the_branchs_catalog_before_the_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[None])
            calls = []
            management.put_catalog = lambda recipe_ref, catalog: calls.append(
                (recipe_ref, catalog)) or {}

            def interrupt(seconds):
                raise KeyboardInterrupt

            services = make_services(directory, management, sleep=interrupt)
            work(services, once=False)
            self.assertEqual(len(calls), 1)
            recipe_ref, catalog = calls[0]
            self.assertEqual(recipe_ref, "dev/requests-worker")
            self.assertEqual(catalog["git_branch"], "dev/requests-worker")

    def test_no_catalog_flag_skips_publish_entirely(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[None])
            calls = []
            management.put_catalog = lambda recipe_ref, catalog: calls.append(
                (recipe_ref, catalog)) or {}

            def interrupt(seconds):
                raise KeyboardInterrupt

            services = make_services(directory, management, sleep=interrupt)
            work(services, once=False, publish_catalog=False)
            self.assertEqual(calls, [])

    def test_dry_run_skips_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[None])
            calls = []
            management.put_catalog = lambda recipe_ref, catalog: calls.append(
                (recipe_ref, catalog)) or {}
            services = make_services(directory, management)
            work(services, once=True, dry_run=True)
            self.assertEqual(calls, [])


class ReleaseClaimsTest(unittest.TestCase):
    def _services(self, management, emit=None):
        with tempfile.TemporaryDirectory() as directory:
            return make_services(Path(directory), management, emit=emit)

    def test_release_hands_each_running_row_back_to_the_queue(self):
        management = ManagementFake()
        management.running_items = [{"id": "req-1"}, {"id": "req-2"}]
        release_claims(self._services(management))
        patched = [(path, payload) for method, path, payload, _ in management.calls
                   if method == "PATCH"]
        self.assertEqual(patched, [
            ("/api/v1/requests/req-1",
             {"status": "queued", "worker_id": "test-worker"}),
            ("/api/v1/requests/req-2",
             {"status": "queued", "worker_id": "test-worker"}),
        ])

    def test_release_asks_only_for_its_own_running_rows(self):
        management = ManagementFake()
        release_claims(self._services(management))
        queried = [path for method, path, _, _ in management.calls if method == "GET"]
        self.assertEqual(
            queried, ["/api/v1/requests?status=running&worker_id=test-worker"])

    def test_release_reports_the_status_the_row_landed_in(self):
        management = ManagementFake()
        management.running_items = [{"id": "req-1"}]

        def request(method, path, payload=None, multipart=None):
            management.calls.append((method, path, payload, multipart))
            if method == "GET":
                return {"items": management.running_items}
            return {"status": "failed", "error": "released after max attempts"}

        management.request = request
        emitted = []
        release_claims(self._services(management, emit=emitted.append))
        self.assertIn("released req-1: failed", emitted)

    def test_release_survives_an_unreachable_server(self):
        management = ManagementFake()

        def request(method, path, payload=None, multipart=None):
            raise SystemExit("connection refused")

        management.request = request
        emitted = []
        release_claims(self._services(management, emit=emitted.append))
        self.assertTrue(any("release query failed" in line for line in emitted))

    def test_dry_run_releases_nothing(self):
        management = ManagementFake()
        management.running_items = [{"id": "req-1"}]
        with tempfile.TemporaryDirectory() as directory:
            services = make_services(Path(directory), management)
            work(services, once=True, dry_run=True)
        self.assertFalse(any(method == "PATCH" for method, _, _, _ in management.calls))


class DrainTest(unittest.TestCase):
    def test_a_drained_worker_claims_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake()
            messages = []
            services = dataclasses.replace(
                make_services(Path(directory), management,
                              emit=messages.append),
                draining=lambda: True)
            work(services, publish_catalog=False)
            self.assertFalse(any(call[1] == "/api/v1/requests/claim"
                                 for call in management.calls))
            self.assertIn("draining: no new work claimed", messages)

    def test_the_request_in_flight_finishes_before_the_worker_leaves(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[generate_row()])
            finished = []
            asked = []

            def fake_generate(path, generate_services, **kwargs):
                asked.append(True)
                return {"generation_ids": ["g"]}

            services = dataclasses.replace(
                make_services(Path(directory), management,
                              generate=fake_generate),
                draining=lambda: bool(asked),
                drained=lambda: finished.append(True))
            work(services, publish_catalog=False)
            self.assertEqual(len(asked), 1)
            self.assertEqual(finished, [True])
            self.assertTrue(any(
                call[1].startswith("/api/v1/requests/") and call[2]
                and call[2].get("status") == "done"
                for call in management.calls))

    def test_leaving_without_a_drain_does_not_acknowledge_one(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[None])
            finished = []

            def interrupt(seconds):
                raise KeyboardInterrupt

            services = dataclasses.replace(
                make_services(Path(directory), management, sleep=interrupt),
                draining=lambda: False,
                drained=lambda: finished.append(True))
            work(services, publish_catalog=False)
            self.assertEqual(finished, [])

    def test_an_unreadable_drain_signal_does_not_stop_the_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[None])
            messages = []

            def explode():
                raise OSError("permission denied")

            def interrupt(seconds):
                raise KeyboardInterrupt

            services = dataclasses.replace(
                make_services(Path(directory), management, sleep=interrupt,
                              emit=messages.append),
                draining=explode)
            work(services, publish_catalog=False)
            self.assertTrue(any("drain check failed" in m for m in messages))
            self.assertTrue(any(call[1] == "/api/v1/requests/claim"
                                for call in management.calls))


class HeartbeatTest(unittest.TestCase):
    def test_sends_running_with_the_worker_id_until_stopped(self):
        import threading

        calls = []
        seen = threading.Event()

        class Management:
            def request(self, method, path, payload=None, multipart=None):
                calls.append((method, path, payload))
                seen.set()
                return {}

        heartbeat = Heartbeat(Management(), "req-1", "test-worker", interval=0.01)
        with heartbeat:
            self.assertTrue(seen.wait(2))
        self.assertEqual(calls[0], ("PATCH", "/api/v1/requests/req-1",
                                    {"status": "running", "worker_id": "test-worker"}))


class HubListenerTest(unittest.TestCase):
    def test_hello_sent_first_and_wake_set_after_connect(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = FakeHubConnection(script=[
                {"type": "hello_ack", "server_time": "t"},
            ])
            services = make_hub_services(
                directory, hub=HubFactory([connection]))
            wake = threading.Event()
            listener = HubListener(services, wake).start()
            try:
                self.assertTrue(wake.wait(2))
                self.assertEqual(connection.sent[0], {
                    "type": "hello", "worker_id": "test-worker",
                    "kinds": ["generate", "deliver"]})
            finally:
                listener.stop()
            self.assertTrue(connection.closed)

    def test_queued_message_sets_wake(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = FakeHubConnection(script=[
                {"type": "hello_ack", "server_time": "t"},
                {"type": "queued", "request_id": "r1", "kind": "generate",
                 "recipe_ref": "ref"},
            ])
            services = make_hub_services(directory, hub=HubFactory([connection]))
            wake = threading.Event()
            listener = HubListener(services, wake).start()
            try:
                self.assertTrue(wake.wait(2))
            finally:
                listener.stop()

    def test_unknown_message_types_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = FakeHubConnection(script=[
                {"type": "pong"}, {"type": "mystery", "foo": "bar"},
            ])
            services = make_hub_services(directory, hub=HubFactory([connection]))
            wake = threading.Event()
            listener = HubListener(services, wake).start()
            try:
                self.assertTrue(wake.wait(2))  # from the initial connect
                wake.clear()
                time.sleep(0.2)
                self.assertFalse(wake.is_set())
            finally:
                listener.stop()

    def test_ping_sent_on_recv_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = FakeHubConnection(script=[])
            services = make_hub_services(
                directory, hub=HubFactory([connection]), ping_interval=0.01)
            wake = threading.Event()
            listener = HubListener(services, wake).start()
            try:
                self.assertTrue(wake.wait(2))
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and not any(
                        message.get("type") == "ping" for message in connection.sent):
                    time.sleep(0.01)
                self.assertTrue(
                    any(message.get("type") == "ping" for message in connection.sent))
            finally:
                listener.stop()

    def test_reconnect_backoff_doubles_up_to_backoff_max(self):
        with tempfile.TemporaryDirectory() as directory:
            # The factory itself fails (the Upgrade is refused) every time, so
            # backoff never gets the post-connect reset -- it grows monotonically.
            factory = HubFactory([ConnectionError("refused") for _ in range(6)])
            sleeps = []
            holder: dict = {}

            def fake_sleep(seconds):
                sleeps.append(seconds)
                if len(sleeps) >= 5:
                    holder["listener"]._stop.set()

            services = make_hub_services(
                directory, hub=factory, sleep=fake_sleep, backoff_max=8)
            listener = HubListener(services, threading.Event())
            holder["listener"] = listener
            listener.start()
            listener._thread.join(2)
            self.assertEqual(sleeps, [1, 2, 4, 8, 8])

    def test_wake_set_again_after_a_reconnect(self):
        with tempfile.TemporaryDirectory() as directory:
            first = FakeHubConnection(script=[])
            second = FakeHubConnection(script=[])
            # A real, short pause between the closed first connection and the
            # reconnect attempt gives the test time to observe and clear the
            # first wake before the second one fires -- an instant retry
            # could collapse the two into a single observed set().
            services = make_hub_services(
                directory, hub=HubFactory([first, second]),
                sleep=lambda seconds: time.sleep(0.2))
            wake = threading.Event()
            listener = HubListener(services, wake).start()
            try:
                self.assertTrue(wake.wait(2))
                wake.clear()
                first.close()  # simulate the server dropping the first socket
                self.assertTrue(wake.wait(2))
            finally:
                listener.stop()

    def test_send_progress_is_a_no_op_when_not_connected(self):
        with tempfile.TemporaryDirectory() as directory:
            services = make_hub_services(directory)
            listener = HubListener(services, threading.Event())
            listener.send_progress("r1", "submit")  # must not raise

    def test_send_progress_forwards_the_message_when_connected(self):
        with tempfile.TemporaryDirectory() as directory:
            services = make_hub_services(directory)
            listener = HubListener(services, threading.Event())
            connection = FakeHubConnection()
            listener._connection = connection
            listener.send_progress("r1", "sampling", step=3, total=10)
            self.assertEqual(connection.sent, [{
                "type": "progress", "request_id": "r1", "phase": "sampling",
                "step": 3, "total": 10}])

    def test_send_progress_omits_none_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            services = make_hub_services(directory)
            listener = HubListener(services, threading.Event())
            connection = FakeHubConnection()
            listener._connection = connection
            listener.send_progress("r1", "submit")
            self.assertEqual(connection.sent, [{
                "type": "progress", "request_id": "r1", "phase": "submit"}])


class ProgressRelayTest(unittest.TestCase):
    def test_forwards_progress_only_while_current_is_set_and_with_the_id(self):
        with tempfile.TemporaryDirectory() as directory:
            feed = GatedHubConnection(
                pre_script=[{"step": 1, "total": 5}], gated_message={"step": 2, "total": 5})
            services = make_hub_services(
                directory, progress_feed=lambda: feed, sleep=lambda seconds: None)
            sent = []

            class RecordingListener:
                def send_progress(self, request_id, phase, **fields):
                    sent.append((request_id, phase, fields))

            relay = ProgressRelay(services, RecordingListener())
            relay.current = None
            relay.start()
            try:
                # First event (step 1) is delivered with current still None --
                # give the background thread a moment to have dropped it.
                time.sleep(0.2)
                self.assertEqual(sent, [])
                relay.current = "req-9"
                feed.release.set()
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and not sent:
                    time.sleep(0.01)
            finally:
                relay.stop()
            self.assertEqual(sent, [("req-9", "sampling", {"step": 2, "total": 5})])


class WorkWithHubTest(unittest.TestCase):
    def test_idle_wait_is_woken_by_a_queued_message_not_by_the_full_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            management = ManagementFake(claim_responses=[
                generate_row(), None, KeyboardInterrupt()])
            connection = GatedHubConnection(
                pre_script=[{"type": "hello_ack", "server_time": "t"}],
                gated_message={"type": "queued", "request_id": "req-1",
                              "kind": "generate", "recipe_ref": "dev/requests-worker"})
            claimed = []

            def fake_generate(path, generate_services, *, key_prefix=None, **kwargs):
                claimed.append(1)
                return {"generation_ids": []}

            services = make_services(directory, management, generate=fake_generate)
            services = dataclasses.replace(
                services, hub=HubFactory([connection]), ping_interval=0.05,
                backoff_max=1)

            done = threading.Event()

            def run():
                work(services, interval=30, once=False)
                done.set()

            worker = threading.Thread(target=run, daemon=True)
            start = time.monotonic()
            worker.start()
            # let pass 1 (claim generate_row) and pass 2 (claim None, enter
            # the idle wait) happen, then deliver the queued message.
            time.sleep(0.3)
            connection.release.set()
            self.assertTrue(done.wait(5))
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 5)
            self.assertEqual(len(claimed), 1)


class WorkOnceHubTest(unittest.TestCase):
    def test_sends_submit_phase_for_a_generate_row_after_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            row = generate_row()
            management = ManagementFake(claim_responses=[row])
            sent = []

            class RecordingListener:
                def send_progress(self, request_id, phase, **fields):
                    sent.append((request_id, phase))

            def fake_generate(path, generate_services, *, key_prefix=None, **kwargs):
                return {"generation_ids": []}

            services = make_services(directory, management, generate=fake_generate)
            work_once(services, listener=RecordingListener())
            self.assertIn(("req-1", "submit"), sent)

    def test_relay_current_is_set_during_execute_and_cleared_after(self):
        with tempfile.TemporaryDirectory() as directory:
            row = generate_row()
            management = ManagementFake(claim_responses=[row])
            observed = {}

            class FakeRelay:
                current = None

            relay = FakeRelay()

            def fake_generate(path, generate_services, *, key_prefix=None, **kwargs):
                observed["during"] = relay.current
                return {"generation_ids": []}

            services = make_services(directory, management, generate=fake_generate)
            work_once(services, relay=relay)
            self.assertEqual(observed["during"], "req-1")
            self.assertIsNone(relay.current)


if __name__ == "__main__":
    unittest.main()
