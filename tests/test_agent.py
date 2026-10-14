"""Tests for the worker's composition root and its ComfyUI node host.

comfy_nodes/yukari_worker cannot be imported through ComfyUI's own
custom_nodes layout in a test, so it is reached the same way
test_yukari_finalize_nodes.py reaches comfy_nodes/yukari_finalize: by
resolving this file's own path onto sys.path rather than relying on the
working directory.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chagama.interfaces.agent import (  # noqa: E402
    build_work_services,
    default_worker_id,
)
from comfy_nodes.yukari_worker import agent as worker_agent  # noqa: E402


class BuildWorkServicesTest(unittest.TestCase):
    @patch("chagama.interfaces.agent.socket.gethostname",
           return_value="gpu-box")
    @patch("chagama.interfaces.agent.os.getpid", return_value=1234)
    def test_default_worker_id_is_process_scoped(self, _getpid, _hostname):
        self.assertEqual(default_worker_id(), "gpu-box:1234")

    def test_draining_and_drained_operate_on_the_sentinel_file(self):
        with TemporaryDirectory() as tmp:
            repository = Path(tmp)
            services = build_work_services(
                repository, worker_id="worker-1", kinds=("generate",))
            drain_file = repository / ".local/_nogit/worker/drain"

            self.assertFalse(services.draining())
            drain_file.parent.mkdir(parents=True, exist_ok=True)
            drain_file.write_text("")
            self.assertTrue(services.draining())

            services.drained()
            self.assertFalse(drain_file.exists())
            self.assertFalse(services.draining())

    def test_hub_false_leaves_hub_unset_but_keeps_the_event_feed(self):
        with TemporaryDirectory() as tmp:
            services = build_work_services(
                Path(tmp), worker_id="worker-1", kinds=("generate",), hub=False)
            self.assertIsNone(services.hub)
            self.assertIsNotNone(services.progress_feed)
            self.assertIsNotNone(services.timings)

    def test_hub_true_sets_hub_and_progress_feed(self):
        with TemporaryDirectory() as tmp:
            services = build_work_services(
                Path(tmp), worker_id="worker-1", kinds=("generate",), hub=True)
            self.assertIsNotNone(services.hub)
            self.assertIsNotNone(services.progress_feed)


class WorkerAgentStartTest(unittest.TestCase):
    def setUp(self):
        worker_agent._thread = None
        self._env_patch = {}
        for key in ("COMFYUI_RECIPES_WORKER", "COMFYUI_RECIPES_WORKER_ID"):
            self._env_patch[key] = os.environ.pop(key, None)

    def tearDown(self):
        worker_agent._thread = None
        for key, value in self._env_patch.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_start_with_env_var_unset_starts_no_thread(self):
        started = threading.Event()

        def run(repository=None, **kwargs):
            started.set()

        worker_agent.start(run=run, ready=lambda: True)
        self.assertFalse(started.wait(timeout=0.2))
        self.assertIsNone(worker_agent._thread)

    def test_start_with_env_var_set_starts_exactly_one_thread(self):
        os.environ["COMFYUI_RECIPES_WORKER"] = "1"
        calls = []
        release = threading.Event()

        def run(repository=None, **kwargs):
            calls.append(kwargs)
            release.wait(timeout=2)

        worker_agent.start(run=run, ready=lambda: True)
        worker_agent.start(run=run, ready=lambda: True)
        time.sleep(0.1)
        release.set()

        self.assertEqual(len(calls), 1)

    def test_worker_id_env_var_is_threaded_through(self):
        os.environ["COMFYUI_RECIPES_WORKER"] = "yes"
        os.environ["COMFYUI_RECIPES_WORKER_ID"] = "box-7"
        seen = []
        done = threading.Event()

        def run(repository=None, *, worker_id=None, emit=None):
            seen.append(worker_id)
            done.set()

        worker_agent.start(run=run, ready=lambda: True)
        self.assertTrue(done.wait(timeout=2))
        self.assertEqual(seen, ["box-7"])

    def test_a_run_that_raises_does_not_propagate_or_kill_the_process(self):
        os.environ["COMFYUI_RECIPES_WORKER"] = "true"
        done = threading.Event()

        def run(repository=None, **kwargs):
            done.set()
            raise RuntimeError("boom")

        try:
            worker_agent.start(run=run, ready=lambda: True)
        except Exception as error:  # pragma: no cover - the assertion below
            self.fail(f"start() propagated an exception: {error!r}")
        self.assertTrue(done.wait(timeout=2))

    def test_start_failure_does_not_raise(self):
        class BrokenLock:
            def __enter__(self):
                raise RuntimeError("lock is broken")

            def __exit__(self, *exc_info):
                return False

        os.environ["COMFYUI_RECIPES_WORKER"] = "1"
        original_lock = worker_agent._lock
        worker_agent._lock = BrokenLock()
        try:
            worker_agent.start(run=lambda *a, **kw: None, ready=lambda: True)
        except Exception as error:  # pragma: no cover
            self.fail(f"start() propagated an exception: {error!r}")
        finally:
            worker_agent._lock = original_lock


class AwaitServerTest(unittest.TestCase):
    def _urlopen(self, failures):
        attempts = []

        def urlopen(url, timeout=None):
            attempts.append(url)
            if len(attempts) <= failures:
                raise OSError("connection refused")
            return contextlib.nullcontext()

        return urlopen, attempts

    def test_it_stops_asking_once_the_server_answers(self):
        urlopen, attempts = self._urlopen(failures=2)
        original = worker_agent.urllib.request.urlopen
        worker_agent.urllib.request.urlopen = urlopen
        try:
            ready = worker_agent._await_server(
                "http://box:8188", 100, lambda _: None, lambda: 0)
        finally:
            worker_agent.urllib.request.urlopen = original
        self.assertTrue(ready)
        self.assertEqual(len(attempts), 3)
        self.assertTrue(all(url.endswith("/system_stats") for url in attempts))

    def test_it_gives_up_at_the_deadline_rather_than_claiming(self):
        urlopen, _ = self._urlopen(failures=99)
        ticks = iter([0, 1, 2, 3, 4])
        original = worker_agent.urllib.request.urlopen
        worker_agent.urllib.request.urlopen = urlopen
        try:
            ready = worker_agent._await_server(
                "http://box:8188", 3, lambda _: None, lambda: next(ticks))
        finally:
            worker_agent.urllib.request.urlopen = original
        self.assertFalse(ready)

    def test_the_checkout_comes_from_the_pack_not_the_working_directory(self):
        os.environ["COMFYUI_RECIPES_WORKER"] = "1"
        worker_agent._thread = None
        seen = []
        done = threading.Event()

        def run(repository=None, **kwargs):
            seen.append(repository)
            done.set()

        try:
            worker_agent.start(run=run, ready=lambda: True)
            self.assertTrue(done.wait(timeout=2))
        finally:
            os.environ.pop("COMFYUI_RECIPES_WORKER", None)
            worker_agent._thread = None
        self.assertEqual(seen, [ROOT])

    def test_a_worker_that_never_sees_comfyui_does_not_claim(self):
        os.environ["COMFYUI_RECIPES_WORKER"] = "1"
        worker_agent._thread = None
        ran = threading.Event()
        try:
            worker_agent.start(run=lambda *a, **kw: ran.set(),
                               ready=lambda: False)
            worker_agent._thread.join(2)
        finally:
            os.environ.pop("COMFYUI_RECIPES_WORKER", None)
            worker_agent._thread = None
        self.assertFalse(ran.is_set())


if __name__ == "__main__":
    unittest.main()
