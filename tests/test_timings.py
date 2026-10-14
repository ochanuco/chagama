"""Generation timings: the event feed, the recorder, the attempt ledger and the PUT."""

from __future__ import annotations

import dataclasses
import json
import tempfile
import time
import unittest
from unittest.mock import MagicMock

from websockets.exceptions import ConnectionClosedError

from chagama.application.work import work_once
from chagama.application.worker_channels import ProgressRelay
from chagama.infrastructure.comfyui.client import ComfyUIClient
from chagama.infrastructure.comfyui.progress import FeedClosed, ProgressFeed
from chagama.infrastructure.comfyui.timings import (
    AttemptTimings,
    EnvCollector,
    TimingRecorder,
    attention_of,
    gpu_name_of,
    parse_history_status,
)

from test_hires_graph import plain_graph
from test_work import FakeHubConnection, ManagementFake, generate_row, make_hub_services, make_services


def ws_message(kind, **data):
    return json.dumps({"type": kind, "data": data})


class ScriptedSocket:
    def __init__(self, frames):
        self.frames = list(frames)

    def recv(self, timeout=None):
        frame = self.frames.pop(0)
        if isinstance(frame, BaseException):
            raise frame
        return frame

    def close(self):
        pass


def feed_of(frames):
    feed = ProgressFeed("http://example.invalid", clock_ms=lambda: 1000)
    feed._socket = ScriptedSocket(frames)
    return feed


class ProgressFeedTest(unittest.TestCase):
    def test_binary_frames_are_skipped_not_decoded(self):
        feed = feed_of([b"\x00\x00\x00\x01\x89PNG", ws_message("progress", value=1, max=4)])
        self.assertIsNone(feed.recv(1))
        self.assertEqual(feed.recv(1)["step"], 1)

    def test_every_typed_message_passes_through_with_a_receive_timestamp(self):
        feed = feed_of([
            ws_message("execution_start", prompt_id="p"),
            ws_message("execution_cached", prompt_id="p", nodes=["1", "2"]),
            ws_message("executing", prompt_id="p", node="3"),
            ws_message("progress", prompt_id="p", node="3", value=2, max=10),
            ws_message("executed", prompt_id="p", node="9", output={}),
            ws_message("execution_success", prompt_id="p"),
            ws_message("status", status={}),
            "not json",
        ])
        events = [feed.recv(1) for _ in range(8)]
        self.assertEqual([event and event["type"] for event in events], [
            "execution_start", "execution_cached", "executing", "progress",
            "executed", "execution_success", None, None])
        self.assertEqual(events[1]["nodes"], ["1", "2"])
        self.assertEqual(events[3], {
            "type": "progress", "prompt_id": "p", "node": "3", "nodes": None,
            "step": 2, "total": 10, "received_at": 1000})

    def test_closed_socket_raises_feed_closed(self):
        feed = feed_of([ConnectionClosedError(None, None)])
        with self.assertRaises(FeedClosed):
            feed.recv(1)


class ClientIdTest(unittest.TestCase):
    def test_feed_url_carries_the_client_id(self):
        feed = ProgressFeed("http://box:8188", client_id="abc")
        self.assertEqual(feed.url, "ws://box:8188/ws?clientId=abc")
        self.assertEqual(ProgressFeed("http://box:8188").url, "ws://box:8188/ws")

    def test_submit_body_includes_client_id_only_when_set(self):
        bodies = []
        for client_id in ("abc", None):
            client = ComfyUIClient("http://example.invalid", client_id=client_id)
            client.request = lambda path, payload=None: (
                bodies.append(payload) or {"prompt_id": "p"})
            client.submit(plain_graph())
        self.assertEqual(bodies[0]["client_id"], "abc")
        self.assertNotIn("client_id", bodies[1])

    def test_progress_state_passes_through_the_feed(self):
        nodes = {"3": {"value": 0.0, "max": 1.0, "state": "running"}}
        event = feed_of([ws_message("progress_state", prompt_id="p", nodes=nodes)]).recv(1)
        self.assertEqual((event["type"], event["nodes"]), ("progress_state", nodes))


class TimingRecorderTest(unittest.TestCase):
    def test_progress_state_sets_start_and_finished_wins_for_end(self):
        recorder = TimingRecorder()

        def send(kind, at, **fields):
            recorder.feed({"type": kind, "prompt_id": "p", "received_at": at, **fields})

        send("progress_state", 100, nodes={"3": {"state": "running"}})
        send("executing", 105, node="3")
        send("progress_state", 150, nodes={"3": {"state": "finished"}})
        send("executing", 190, node=None)
        send("progress_state", 200, nodes={"3": {"state": "finished"}})
        node = recorder.snapshot("p")["nodes"]["3"]
        self.assertEqual((node["started_at"], node["ended_at"]), (100, 150))


    def test_node_start_end_cached_and_step_intervals(self):
        recorder = TimingRecorder()

        def send(kind, at, **fields):
            recorder.feed({"type": kind, "prompt_id": "p", "received_at": at, **fields})

        send("execution_cached", 100, nodes=["1"])
        send("executing", 110, node="2")
        send("executing", 150, node="3")
        for at, step in ((200, 1), (230, 2), (270, 3)):
            send("progress", at, node="3", step=step, total=3)
        send("executing", 300, node=None)
        send("execution_success", 301)
        snapshot = recorder.snapshot("p")
        self.assertEqual(snapshot["cached"], {"1"})
        self.assertEqual((snapshot["nodes"]["2"]["started_at"],
                          snapshot["nodes"]["2"]["ended_at"]), (110, 150))
        sampler = snapshot["nodes"]["3"]
        self.assertEqual((sampler["started_at"], sampler["ended_at"]), (150, 300))
        self.assertEqual(sampler["step_ms"], [50, 30, 40])
        self.assertEqual(sampler["steps_total"], 3)

    def test_unknown_prompt_and_missing_prompt_id(self):
        recorder = TimingRecorder()
        recorder.feed({"type": "executing", "node": "1"})
        self.assertEqual(recorder.snapshot("x"), {"nodes": {}, "cached": set()})


def history_entry(messages, status_str="success", graph=None):
    entry = {"status": {"status_str": status_str, "messages": messages},
             "outputs": {"9": {"images": [{"filename": "a.png", "type": "output"}]}}}
    if graph is not None:
        entry["prompt"] = [1, "p", graph, {}, ["9"]]
    return entry


SUCCESS = [["execution_start", {"prompt_id": "p", "timestamp": 1000}],
           ["execution_cached", {"nodes": ["1", "2"], "prompt_id": "p", "timestamp": 1001}],
           ["execution_success", {"prompt_id": "p", "timestamp": 1900}]]


class HistoryStatusTest(unittest.TestCase):
    def test_success(self):
        self.assertEqual(parse_history_status(history_entry(SUCCESS)), {
            "start_at": 1000, "end_at": 1900, "cached": {"1", "2"}, "status": "success"})

    def test_error_and_interrupted(self):
        error = [SUCCESS[0], ["execution_error", {"prompt_id": "p", "timestamp": 1500}]]
        parsed = parse_history_status(history_entry(error, "error"))
        self.assertEqual((parsed["end_at"], parsed["status"]), (1500, "error"))
        stopped = [SUCCESS[0], ["execution_interrupted", {"timestamp": 1200}]]
        self.assertEqual(parse_history_status(history_entry(stopped))["status"], "interrupted")

    def test_without_messages_falls_back_to_status_str(self):
        parsed = parse_history_status({"status": {"status_str": "success"}})
        self.assertEqual((parsed["start_at"], parsed["status"]), (None, "success"))
        self.assertEqual(parse_history_status({})["status"], "unknown")


class EnvTest(unittest.TestCase):
    def test_attention_and_gpu_name(self):
        self.assertEqual(attention_of(["main.py", "--use-ck-attention"]), "ck")
        self.assertEqual(attention_of(["main.py", "--fast"]), "default")
        self.assertIsNone(attention_of(None))
        self.assertEqual(gpu_name_of("cuda:0 NVIDIA GeForce RTX 3060 : native"),
                         "NVIDIA GeForce RTX 3060")

    def test_collector_reads_system_stats_and_memoises_subprocess_lookups(self):
        comfyui = MagicMock()
        comfyui.request.return_value = {
            "system": {"comfyui_version": "0.37.0", "pytorch_version": "2.13.0+cu130",
                       "argv": ["main.py", "--fast", "--use-ck-attention"]},
            "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 3060 : native"}]}
        git = MagicMock(return_value={"commit": "abc", "dirty": True})
        run = MagicMock(return_value=MagicMock(stdout="581.42\n"))
        collector = EnvCollector(comfyui, git, run=run)
        env = collector()
        collector()
        self.assertEqual(env, {
            "comfyui_version": "0.37.0",
            "argv": ["main.py", "--fast", "--use-ck-attention"], "attention": "ck",
            "pytorch_version": "2.13.0+cu130", "worker_commit": "abc",
            "worker_dirty": True, "gpu_name": "NVIDIA GeForce RTX 3060",
            "gpu_driver": "581.42"})
        self.assertEqual((git.call_count, run.call_count), (1, 1))

    def test_unreachable_comfyui_and_missing_nvidia_smi_give_nulls(self):
        comfyui = MagicMock()
        comfyui.request.side_effect = OSError("down")
        collector = EnvCollector(comfyui, lambda: {"commit": "c", "dirty": False},
                                 run=MagicMock(side_effect=FileNotFoundError()))
        env = collector()
        self.assertIsNone(env["comfyui_version"])
        self.assertIsNone(env["gpu_driver"])
        self.assertEqual(env["worker_commit"], "c")


class Clock:
    def __init__(self, start=1_000):
        self.now = start

    def __call__(self):
        self.now += 10
        return self.now


class AttemptLedgerTest(unittest.TestCase):
    def make(self, env=None):
        timings = AttemptTimings(
            env_provider=(lambda: env) if env is not None else None, clock_ms=Clock())
        timings.begin(2)
        return timings

    def test_client_registers_submit_and_history_outputs(self):
        timings = self.make({"comfyui_version": "0.37.0"})
        client = ComfyUIClient("http://example.invalid", poll_interval=0, timings=timings)
        graph = plain_graph()
        submitted = {}

        def request(path, payload=None):
            if path == "/prompt":
                submitted.update(payload["prompt"])
                return {"prompt_id": "p"}
            return {"p": history_entry(SUCCESS)}

        client.request = request
        self.assertEqual(client.submit(graph, purpose="render"), "p")
        self.assertEqual(submitted["3"]["_meta"]["title"], "base_sampler")
        self.assertNotIn("_meta", graph["3"])
        client.wait_for("p")
        timings.recorder.feed({"type": "executing", "prompt_id": "p", "node": "3",
                               "received_at": 1100})
        timings.recorder.feed({"type": "executing", "prompt_id": "p", "node": None,
                               "received_at": 1400})
        timings.ingested("p")
        payload = timings.payload("w", "done")
        prompt = payload["prompts"][0]
        self.assertEqual((prompt["purpose"], prompt["resumed"], prompt["status"]),
                         ("render", False, "success"))
        self.assertEqual((prompt["execution_start_at"], prompt["execution_end_at"]),
                         (1000, 1900))
        for key in ("submitted_at", "outputs_ready_at", "ingested_at"):
            self.assertIsInstance(prompt[key], int)
        nodes = {node["node_id"]: node for node in prompt["nodes"]}
        self.assertTrue(nodes["1"]["cached"])
        self.assertEqual((nodes["3"]["role"], nodes["3"]["started_at"],
                          nodes["3"]["ended_at"], nodes["3"]["cached"]),
                         ("base_sampler", 1100, 1400, False))
        self.assertFalse(payload["cold_load"])

    def test_cold_load_when_a_loader_ran_uncached(self):
        timings = self.make()
        timings.register("p", plain_graph())
        entry = history_entry([SUCCESS[0], ["execution_cached", {"nodes": ["5"]}],
                               SUCCESS[2]])
        timings.outputs_ready("p", entry)
        self.assertTrue(timings.payload("w", "done")["cold_load"])

    def test_cold_load_unknown_without_data(self):
        timings = self.make()
        self.assertIsNone(timings.payload("w", "done")["cold_load"])

    def test_error_entry_is_recorded_before_the_wait_raises(self):
        timings = self.make()
        client = ComfyUIClient("http://example.invalid", poll_interval=0, timings=timings)
        client.request = lambda path, payload=None: (
            {"prompt_id": "p"} if path == "/prompt" else
            {"p": {"status": {"status_str": "error", "messages": [
                SUCCESS[0], ["execution_error", {"timestamp": 1500}]]}, "outputs": {}}})
        client.submit(plain_graph())
        with self.assertRaises(RuntimeError):
            client.wait_for("p")
        prompt = timings.payload("w", "failed")["prompts"][0]
        self.assertEqual((prompt["status"], prompt["execution_end_at"]), ("error", 1500))

    def test_resumed_prompt_takes_its_graph_from_history(self):
        timings = self.make()
        client = ComfyUIClient("http://example.invalid", poll_interval=0, timings=timings)
        graph = {"3": {"class_type": "KSampler", "inputs": {}, "_meta": {"title": "base_sampler"}}}
        client.request = lambda path, payload=None: {"p": history_entry(SUCCESS, graph=graph)}
        self.assertTrue(client.knows("p"))
        client.wait_for("p")
        prompt = timings.payload("w", "done")["prompts"][0]
        self.assertTrue(prompt["resumed"])
        self.assertIsNone(prompt["submitted_at"])
        self.assertEqual(prompt["nodes"][0]["role"], "base_sampler")

    def test_no_attempt_number_means_no_payload(self):
        timings = AttemptTimings()
        timings.begin(None)
        self.assertIsNone(timings.payload("w", "done"))


class RelayRecordingTest(unittest.TestCase):
    def test_relay_records_every_event_without_a_hub_and_forwards_only_progress(self):
        events = [
            {"type": "executing", "prompt_id": "p", "node": "3", "received_at": 100},
            {"type": "progress", "prompt_id": "p", "node": "3", "step": 1,
             "total": 2, "received_at": 140},
        ]
        feed = FakeHubConnection(script=events)
        with tempfile.TemporaryDirectory() as directory:
            services = make_hub_services(directory, progress_feed=lambda: feed,
                                         sleep=lambda seconds: None)
        timings = AttemptTimings()
        services = dataclasses.replace(services, timings=timings)
        sent = []

        class Listener:
            def send_progress(self, request_id, phase, **fields):
                sent.append((request_id, fields))

        relay = ProgressRelay(services, Listener())
        relay.current = "req-1"
        relay.start()
        try:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not sent:
                time.sleep(0.01)
        finally:
            relay.stop()
        self.assertEqual(sent, [("req-1", {"step": 1, "total": 2})])
        node = timings.recorder.snapshot("p")["nodes"]["3"]
        self.assertEqual((node["started_at"], node["step_ms"]), (100, [40]))

        silent = ProgressRelay(services, None)
        feed.script = [dict(events[0], prompt_id="q")]
        feed.closed = False
        silent.start()
        try:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not timings.recorder.snapshot("q")["nodes"]:
                time.sleep(0.01)
        finally:
            silent.stop()
        self.assertIn("3", timings.recorder.snapshot("q")["nodes"])


class WorkOnceTimingsTest(unittest.TestCase):
    def run_once(self, generate, management=None):
        directory = tempfile.mkdtemp()
        management = management or ManagementFake(claim_responses=[generate_row()])
        emitted = []
        services = make_services(directory, management, generate=generate,
                                 emit=emitted.append)
        timings = AttemptTimings(
            env_provider=lambda: {"comfyui_version": "0.37.0"}, clock_ms=Clock())
        services = dataclasses.replace(services, timings=timings)
        return services, management, emitted, timings

    def test_done_attempt_puts_timings_after_the_final_patch(self):
        def generate(path, generate_services, *, key_prefix=None, **kwargs):
            return {"generation_ids": ["g1"]}

        services, management, _, timings = self.run_once(generate)
        timings.register("p", plain_graph(), submitted_at=5)
        self.assertTrue(work_once(services))
        methods = [(call[0], call[1]) for call in management.calls]
        self.assertEqual(methods[-2:], [("PATCH", "/api/v1/requests/req-1"),
                                        ("PUT", "/api/v1/requests/req-1/timings")])
        payload = management.calls[-1][2]
        self.assertEqual({key: payload[key] for key in (
            "worker_id", "attempt", "version", "source", "status", "env")}, {
            "worker_id": "test-worker", "attempt": 1, "version": "v2",
            "source": "worker", "status": "done",
            "env": {"comfyui_version": "0.37.0"}})
        self.assertLess(payload["claimed_at"], payload["finished_at"])
        self.assertEqual(payload["prompts"], [])

    def test_failed_attempt_puts_failed_status(self):
        def generate(path, generate_services, *, key_prefix=None, **kwargs):
            raise RuntimeError("boom")

        services, management, _, _ = self.run_once(generate)
        work_once(services)
        self.assertEqual(management.calls[-1][1], "/api/v1/requests/req-1/timings")
        self.assertEqual(management.calls[-1][2]["status"], "failed")

    def test_a_timing_send_failure_does_not_change_the_outcome(self):
        class Failing(ManagementFake):
            def request(self, method, path, payload=None, multipart=None):
                if method == "PUT":
                    raise SystemExit("503")
                return super().request(method, path, payload, multipart)

        def generate(path, generate_services, *, key_prefix=None, **kwargs):
            return {"generation_ids": []}

        management = Failing(claim_responses=[generate_row()])
        services, management, emitted, _ = self.run_once(generate, management)
        self.assertTrue(work_once(services))
        patch_call = next(call for call in management.calls if call[0] == "PATCH")
        self.assertEqual(patch_call[2]["status"], "done")
        self.assertTrue(any("timings failed for req-1" in line for line in emitted))

    def test_row_without_attempt_skips_the_send(self):
        def generate(path, generate_services, *, key_prefix=None, **kwargs):
            return {"generation_ids": []}

        row = generate_row()
        del row["attempt"]
        services, management, emitted, _ = self.run_once(
            generate, ManagementFake(claim_responses=[row]))
        work_once(services)
        self.assertFalse(any(call[0] == "PUT" for call in management.calls))
        self.assertTrue(any("no attempt" in line for line in emitted))


if __name__ == "__main__":
    unittest.main()
