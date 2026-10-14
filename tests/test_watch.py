"""Tests for the pending-ExperimentRun poll loop.

All collaborators are fakes: this suite never opens a network socket and
never calls the real generate() use case (that is covered separately in
test_generate_application.py).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chagama.application.generate import GenerateServices, validate_request
from chagama.application.watch import (
    SkipRun,
    WatchServices,
    build_request,
    poll_once,
    watch,
)


class ManagementFake:
    def __init__(self, pending_items):
        self.pending_items = pending_items
        self.calls = []

    def request(self, method, path, payload=None, multipart=None):
        self.calls.append((method, path, payload, multipart))
        if path.startswith("/api/v1/experiment-runs?pending=true"):
            return {"items": self.pending_items}
        raise AssertionError(f"unexpected management call: {method} {path}")


class GenerateFake:
    def __init__(self):
        self.calls = []
        self.fail_for = set()
        self.raise_for: dict[str, BaseException] = {}

    def __call__(self, request_path, services, *, dry_run=False, force=False):
        self.calls.append((request_path, dry_run))
        if request_path.stem in self.fail_for:
            raise SystemExit("generate failed")
        if request_path.stem in self.raise_for:
            raise self.raise_for[request_path.stem]


def base_item(**overrides):
    item = {
        "id": "run-1",
        "experiment_id": "exp-1",
        "run_index": 2,
        "parent_run_id": None,
        "generation_id": None,
        "overrides": {"patches": [
            {"target": "prompt.positive", "op": "append",
             "value": ", distinct sock cuff", "reason": "test"}]},
        "objective": "ソックスの縁を明示する",
        "evaluation": None,
        "decision": None,
        "note": None,
        "experiment": {
            "id": "exp-1", "short_id": "abc123", "name": "sock-cuff",
            "status": "active", "base_recipe": "yukari",
            "base_parameters": {"pose": "lounge", "costume": "default", "count": 3},
        },
    }
    item.update(overrides)
    return item


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


class BuildRequestTest(unittest.TestCase):
    def test_matches_documented_shape(self):
        request = build_request(base_item())
        self.assertEqual(request, {
            "schema_version": 1,
            "request": {"instruction": "ソックスの縁を明示する", "count": 3},
            "generation": {"recipe": "yukari",
                            "parameters": {"pose": "lounge", "costume": "default"}},
            "semantic": {"summary": "ソックスの縁を明示する"},
            "experiment": {
                "experiment_id": "exp-1", "run_id": "run-1",
                "overrides": {"patches": [
                    {"target": "prompt.positive", "op": "append",
                     "value": ", distinct sock cuff", "reason": "test"}]},
            },
        })

    def test_null_objective_falls_back_and_still_validates(self):
        item = base_item(objective=None)
        request = build_request(item)
        self.assertEqual(request["request"]["instruction"], "sock-cuff run #2")
        self.assertEqual(request["semantic"]["summary"], "sock-cuff run #2")
        validate_request(request)  # must not raise

    def test_null_base_parameters_still_builds_a_well_formed_request(self):
        item = base_item()
        item["experiment"]["base_parameters"] = None
        request = build_request(item)
        self.assertEqual(request["generation"]["parameters"], {})
        self.assertEqual(request["request"]["count"], 1)
        self.assertEqual(request["schema_version"], 1)
        self.assertTrue(request["semantic"]["summary"])

    def test_missing_base_recipe_raises_skip_run(self):
        item = base_item()
        item["experiment"]["base_recipe"] = None
        with self.assertRaises(SkipRun):
            build_request(item)

    def test_non_mapping_base_parameters_raises_skip_run(self):
        item = base_item()
        item["experiment"]["base_parameters"] = ["pose", "lounge"]
        with self.assertRaises(SkipRun):
            build_request(item)

    def test_overrides_pass_through_verbatim(self):
        overrides = {"patches": [
            {"target": "render.cfg", "op": "set", "value": 4.5, "reason": "x"}]}
        item = base_item(overrides=overrides)
        request = build_request(item)
        self.assertEqual(request["experiment"]["overrides"], overrides)

    def test_count_moves_out_of_generation_parameters(self):
        request = build_request(base_item())
        self.assertNotIn("count", request["generation"]["parameters"])


class PollOnceTest(unittest.TestCase):
    def test_bad_run_is_skipped_and_a_later_valid_run_still_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = base_item(id="run-bad")
            bad["experiment"]["base_recipe"] = None
            good = base_item(id="run-good")
            management = ManagementFake([bad, good])
            fake_generate = GenerateFake()
            messages = []
            services = WatchServices(
                management=management,
                generate_services=make_generate_services(Path(directory)),
                generate=fake_generate,
                emit=messages.append,
            )
            poll_once(services)
            self.assertEqual(len(fake_generate.calls), 1)
            self.assertEqual(fake_generate.calls[0][0].stem, "run-good")
            self.assertTrue(any("run-bad" in message for message in messages))

    def test_generate_failure_is_reported_and_does_not_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            item = base_item(id="run-explodes")
            management = ManagementFake([item])
            fake_generate = GenerateFake()
            fake_generate.fail_for.add("run-explodes")
            messages = []
            services = WatchServices(
                management=management,
                generate_services=make_generate_services(Path(directory)),
                generate=fake_generate, emit=messages.append,
            )
            poll_once(services)
            self.assertTrue(any("run-explodes" in message for message in messages))

    def test_list_base_parameters_is_skipped_and_a_later_valid_run_still_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = base_item(id="run-bad-params")
            bad["experiment"]["base_parameters"] = ["pose", "lounge"]
            good = base_item(id="run-good")
            management = ManagementFake([bad, good])
            fake_generate = GenerateFake()
            messages = []
            services = WatchServices(
                management=management,
                generate_services=make_generate_services(Path(directory)),
                generate=fake_generate,
                emit=messages.append,
            )
            poll_once(services)
            self.assertEqual(len(fake_generate.calls), 1)
            self.assertEqual(fake_generate.calls[0][0].stem, "run-good")
            self.assertTrue(
                any("run-bad-params" in message for message in messages))

    def test_generate_runtime_error_is_reported_and_next_run_still_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            exploding = base_item(id="run-runtime-error")
            good = base_item(id="run-after-error")
            management = ManagementFake([exploding, good])
            fake_generate = GenerateFake()
            fake_generate.raise_for["run-runtime-error"] = RuntimeError("boom")
            messages = []
            services = WatchServices(
                management=management,
                generate_services=make_generate_services(Path(directory)),
                generate=fake_generate,
                emit=messages.append,
            )
            poll_once(services)
            self.assertEqual(len(fake_generate.calls), 2)
            self.assertEqual(fake_generate.calls[1][0].stem, "run-after-error")
            self.assertTrue(
                any("run-runtime-error" in message and "boom" in message
                    for message in messages))

    def test_request_file_path_is_stable_per_run(self):
        with tempfile.TemporaryDirectory() as directory:
            item = base_item(id="run-stable")
            management = ManagementFake([item])
            fake_generate = GenerateFake()
            services = WatchServices(
                management=management,
                generate_services=make_generate_services(Path(directory)),
                generate=fake_generate,
            )
            poll_once(services)
            poll_once(services)
            self.assertEqual(fake_generate.calls[0][0], fake_generate.calls[1][0])
            self.assertTrue(fake_generate.calls[0][0].exists())

    def test_dry_run_is_threaded_into_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            item = base_item(id="run-dry")
            management = ManagementFake([item])
            fake_generate = GenerateFake()
            services = WatchServices(
                management=management,
                generate_services=make_generate_services(Path(directory)),
                generate=fake_generate,
            )
            poll_once(services, dry_run=True)
            self.assertTrue(fake_generate.calls[0][1])

    def test_makes_no_chimera_call_beyond_the_pending_get(self):
        with tempfile.TemporaryDirectory() as directory:
            item = base_item(id="run-only-get")
            management = ManagementFake([item])
            fake_generate = GenerateFake()  # stands in for generate()'s own calls
            services = WatchServices(
                management=management,
                generate_services=make_generate_services(Path(directory)),
                generate=fake_generate,
            )
            poll_once(services)
            self.assertEqual(
                management.calls,
                [("GET", "/api/v1/experiment-runs?pending=true", None, None)])


class WatchLoopTest(unittest.TestCase):
    def test_once_polls_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            item = base_item(id="run-once")
            management = ManagementFake([item])
            fake_generate = GenerateFake()

            def no_sleep(seconds):
                raise AssertionError("must not sleep when --once")

            services = WatchServices(
                management=management,
                generate_services=make_generate_services(Path(directory)),
                generate=fake_generate, sleep=no_sleep,
            )
            watch(services, once=True)
            self.assertEqual(len(fake_generate.calls), 1)
            self.assertEqual(
                sum(1 for call in management.calls if call[1].startswith(
                    "/api/v1/experiment-runs")),
                1)

    def test_keyboard_interrupt_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            item = base_item(id="run-loop")
            management = ManagementFake([item])
            fake_generate = GenerateFake()

            def interrupt(seconds):
                raise KeyboardInterrupt

            messages = []
            services = WatchServices(
                management=management,
                generate_services=make_generate_services(Path(directory)),
                generate=fake_generate, sleep=interrupt, emit=messages.append,
            )
            watch(services, once=False)  # must not raise
            self.assertTrue(any("stopped" in message for message in messages))

    def test_keyboard_interrupt_from_generate_still_stops_the_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            item = base_item(id="run-ctrl-c")
            management = ManagementFake([item])
            fake_generate = GenerateFake()
            fake_generate.raise_for["run-ctrl-c"] = KeyboardInterrupt()

            def fail_if_reached(seconds):
                raise AssertionError("must not sleep: KeyboardInterrupt should "
                                      "have stopped the loop first")

            messages = []
            services = WatchServices(
                management=management,
                generate_services=make_generate_services(Path(directory)),
                generate=fake_generate, sleep=fail_if_reached,
                emit=messages.append,
            )
            watch(services, once=False)  # must not raise
            self.assertTrue(any("stopped" in message for message in messages))
            self.assertFalse(
                any("run-ctrl-c" in message for message in messages),
                "KeyboardInterrupt must not be reported as a skipped run")


if __name__ == "__main__":
    unittest.main()


class RequestPathGuardTest(unittest.TestCase):
    def test_a_run_id_that_would_escape_the_output_root_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            escaping = base_item(id="../../etc/passwd")
            management = ManagementFake([escaping])
            fake_generate = GenerateFake()
            messages = []
            services = WatchServices(
                management=management,
                generate_services=make_generate_services(Path(directory)),
                generate=fake_generate,
                emit=messages.append,
            )
            poll_once(services)
            self.assertEqual(fake_generate.calls, [])
            self.assertTrue(any("invalid run id" in message for message in messages))
