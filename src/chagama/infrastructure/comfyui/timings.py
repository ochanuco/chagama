"""Per-attempt generation timings: ComfyUI's event feed plus what the worker saw.

`TimingRecorder` folds the /ws event stream into per-node start, end and
step intervals. `AttemptTimings` is the worker-side ledger of one claimed
request: the prompts it submitted, when their outputs were seen and
ingested, and the environment they ran in. `payload()` is the body of
chimera's `PUT /api/v1/requests/:id/timings`.
"""

from __future__ import annotations

import functools
import subprocess
import threading
import time
from collections.abc import Callable, Mapping

from .roles import role_of, snake_case

VERSION = "v2"
SOURCE = "worker"
LOADER_CLASSES = frozenset({
    "UNETLoader", "CLIPLoader", "VAELoader", "CheckpointLoaderSimple",
    "DiffusersLoader", "LoraLoader", "LoraLoaderModelOnly", "ControlNetLoader",
})
ATTENTION_FLAGS = (
    ("--use-ck-attention", "ck"),
    ("--use-sage-attention", "sage"),
    ("--use-flash-attention", "flash"),
    ("--use-pytorch-cross-attention", "pytorch"),
    ("--use-split-cross-attention", "split"),
    ("--use-quad-cross-attention", "quad"),
)
HISTORY_END_MESSAGES = {
    "execution_success": "success",
    "execution_error": "error",
    "execution_interrupted": "interrupted",
}


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_history_status(entry: Mapping) -> dict:
    """Execution start/end, cached node ids and outcome from a /history entry."""
    status = entry.get("status") if isinstance(entry.get("status"), Mapping) else {}
    parsed: dict = {"start_at": None, "end_at": None, "cached": set(),
                    "status": "unknown"}
    for message in status.get("messages") or ():
        if not (isinstance(message, (list, tuple)) and len(message) == 2
                and isinstance(message[1], Mapping)):
            continue
        name, data = message
        stamp = data.get("timestamp")
        stamp = int(stamp) if isinstance(stamp, (int, float)) else None
        if name == "execution_start":
            parsed["start_at"] = stamp
        elif name == "execution_cached":
            parsed["cached"].update(str(node) for node in data.get("nodes") or ())
        elif name in HISTORY_END_MESSAGES:
            parsed["end_at"] = stamp
            parsed["status"] = HISTORY_END_MESSAGES[name]
    if parsed["status"] == "unknown" and status.get("status_str") in ("success", "error"):
        parsed["status"] = status["status_str"]
    return parsed


class TimingRecorder:
    """Thread-safe: fed by the feed thread, read by the worker thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompts: dict[str, dict] = {}

    def clear(self) -> None:
        with self._lock:
            self._prompts.clear()

    def feed(self, event: Mapping) -> None:
        prompt_id = event.get("prompt_id")
        if not prompt_id:
            return
        at = event.get("received_at")
        if at is None:
            at = now_ms()
        kind = event.get("type")
        with self._lock:
            prompt = self._prompts.setdefault(
                prompt_id, {"nodes": {}, "current": None, "cached": set()})
            if kind == "executing":
                self._close(prompt, at)
                node = event.get("node")
                if node is not None:
                    node = str(node)
                    state = self._node(prompt, node)
                    if state["started_at"] is None:
                        state["started_at"] = at
                    state["mark"] = at
                    prompt["current"] = node
            elif kind == "progress_state":
                for node, node_state in (event.get("nodes") or {}).items():
                    if not isinstance(node_state, dict):
                        continue
                    state = self._node(prompt, str(node))
                    if node_state.get("state") == "running" and state["started_at"] is None:
                        state["started_at"] = at
                        state["mark"] = at
                    elif node_state.get("state") == "finished" and not state["finished"]:
                        state["finished"] = True
                        state["ended_at"] = at
            elif kind == "execution_cached":
                prompt["cached"].update(str(node) for node in event.get("nodes") or ())
            elif kind == "progress":
                node = event.get("node")
                node = str(node) if node is not None else prompt["current"]
                if node is None:
                    return
                state = self._node(prompt, node)
                if state["mark"] is not None:
                    state["step_ms"].append(max(0, at - state["mark"]))
                state["mark"] = at
                total = event.get("total")
                if isinstance(total, (int, float)):
                    state["steps_total"] = max(state["steps_total"] or 0, int(total))
            elif kind in HISTORY_END_MESSAGES:
                self._close(prompt, at)

    @staticmethod
    def _node(prompt: dict, node: str) -> dict:
        return prompt["nodes"].setdefault(node, {
            "started_at": None, "ended_at": None, "steps_total": None,
            "step_ms": [], "mark": None, "finished": False})

    @staticmethod
    def _close(prompt: dict, at: int) -> None:
        current = prompt["current"]
        if current is not None:
            state = prompt["nodes"].get(current)
            if state is not None and state["ended_at"] is None and not state["finished"]:
                state["ended_at"] = at
        prompt["current"] = None

    def snapshot(self, prompt_id: str) -> dict:
        with self._lock:
            prompt = self._prompts.get(prompt_id)
            if prompt is None:
                return {"nodes": {}, "cached": set()}
            return {"nodes": {node: {**state, "step_ms": list(state["step_ms"])}
                              for node, state in prompt["nodes"].items()},
                    "cached": set(prompt["cached"])}


def _safe(method: Callable) -> Callable:
    """Timing capture never raises into the render path."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as error:
            self.errors.append(f"{method.__name__}: {error}")
            return None
    return wrapper


class AttemptTimings:
    def __init__(self, recorder: TimingRecorder | None = None,
                 env_provider: Callable[[], dict] | None = None,
                 clock_ms: Callable[[], int] = now_ms) -> None:
        self.recorder = recorder or TimingRecorder()
        self.errors: list[str] = []
        self._env_provider = env_provider
        self._clock_ms = clock_ms
        self._lock = threading.RLock()
        self._env: dict | None = None
        self._attempt: int | None = None
        self._claimed_at: int | None = None
        self._prompts: dict[str, dict] = {}

    @property
    def attempt(self) -> int | None:
        return self._attempt

    def invalidate_env(self) -> None:
        with self._lock:
            self._env = None

    @_safe
    def begin(self, attempt: object) -> None:
        with self._lock:
            self._attempt = attempt if isinstance(attempt, int) else None
            self._claimed_at = self._clock_ms()
            self._prompts = {}
        self.recorder.clear()

    @_safe
    def register(self, prompt_id: str, graph: Mapping | None, *,
                 purpose: str = "render", submitted_at: int | None = None,
                 resumed: bool = False) -> None:
        with self._lock:
            self._prompts[prompt_id] = {
                "graph": graph, "purpose": purpose, "resumed": resumed,
                "submitted_at": submitted_at, "execution_start_at": None,
                "execution_end_at": None, "outputs_ready_at": None,
                "ingested_at": None, "status": "unknown", "cached": set()}

    @_safe
    def resumed(self, prompt_id: str) -> None:
        with self._lock:
            if prompt_id not in self._prompts:
                self.register(prompt_id, None, resumed=True)

    @_safe
    def outputs_ready(self, prompt_id: str, entry: Mapping) -> None:
        with self._lock:
            if prompt_id not in self._prompts:
                self.register(prompt_id, None, resumed=True)
            prompt = self._prompts[prompt_id]
            parsed = parse_history_status(entry)
            prompt.update(outputs_ready_at=self._clock_ms(),
                          execution_start_at=parsed["start_at"],
                          execution_end_at=parsed["end_at"],
                          status=parsed["status"], cached=parsed["cached"])
            history_prompt = entry.get("prompt")
            if (prompt["graph"] is None and isinstance(history_prompt, (list, tuple))
                    and len(history_prompt) > 2 and isinstance(history_prompt[2], Mapping)):
                prompt["graph"] = history_prompt[2]

    @_safe
    def ingested(self, prompt_id: str) -> None:
        with self._lock:
            prompt = self._prompts.get(prompt_id)
            if prompt is not None:
                prompt["ingested_at"] = self._clock_ms()

    def _environment(self) -> dict | None:
        with self._lock:
            if self._env is None and self._env_provider is not None:
                try:
                    env = self._env_provider()
                except Exception as error:
                    self.errors.append(f"env: {error}")
                    env = None
                if env and env.get("comfyui_version"):
                    self._env = env
                return env
            return self._env

    def _nodes(self, prompt: Mapping, prompt_id: str) -> list[dict]:
        recorded = self.recorder.snapshot(prompt_id)
        cached_ids = prompt["cached"] | recorded["cached"]
        nodes = []
        for node_id, node in (prompt["graph"] or {}).items():
            if not isinstance(node, Mapping):
                continue
            class_type = str(node.get("class_type") or "unknown")
            cached = node_id in cached_ids
            state = {} if cached else recorded["nodes"].get(node_id, {})
            nodes.append({
                "node_id": node_id, "class_type": class_type,
                "role": role_of(node) or snake_case(class_type), "cached": cached,
                "started_at": state.get("started_at"),
                "ended_at": state.get("ended_at"),
                "steps_total": state.get("steps_total"),
                "step_ms": state.get("step_ms") or None})
        return nodes

    def payload(self, worker_id: str, status: str) -> dict | None:
        """The timings body for the attempt, or None when it has no attempt number."""
        with self._lock:
            if self._attempt is None:
                return None
            prompts = []
            known = []
            for prompt_id, prompt in self._prompts.items():
                nodes = self._nodes(prompt, prompt_id)
                prompts.append({
                    "prompt_id": prompt_id, "purpose": prompt["purpose"],
                    "resumed": prompt["resumed"],
                    "submitted_at": prompt["submitted_at"],
                    "execution_start_at": prompt["execution_start_at"],
                    "execution_end_at": prompt["execution_end_at"],
                    "outputs_ready_at": prompt["outputs_ready_at"],
                    "ingested_at": prompt["ingested_at"],
                    "status": prompt["status"], "nodes": nodes})
                if nodes and (prompt["status"] == "success"
                              or any(node["started_at"] for node in nodes)):
                    known.append(nodes)
            cold_load = None
            if known:
                cold_load = any(
                    node["class_type"] in LOADER_CLASSES and not node["cached"]
                    for nodes in known for node in nodes)
            return {
                "worker_id": worker_id, "attempt": self._attempt,
                "version": VERSION, "source": SOURCE, "status": status,
                "claimed_at": self._claimed_at, "finished_at": self._clock_ms(),
                "env": self._environment(), "cold_load": cold_load,
                "prompts": prompts}


def note_ingested(comfyui: object, prompt_id: str) -> None:
    timings = getattr(comfyui, "timings", None)
    if timings is not None:
        timings.ingested(prompt_id)


def attention_of(argv: object) -> str | None:
    if not isinstance(argv, list):
        return None
    for flag, name in ATTENTION_FLAGS:
        if flag in argv:
            return name
    return "default"


def gpu_name_of(device_name: object) -> str | None:
    if not isinstance(device_name, str):
        return None
    name = device_name.removeprefix("cuda:0 ").removesuffix(" : native")
    return name.strip() or None


class EnvCollector:
    """Reads ComfyUI's /system_stats; git and driver lookups are memoised
    because they spawn subprocesses and do not change under a running worker."""

    def __init__(self, comfyui: object, git_metadata: Callable[[], dict],
                 run: Callable = subprocess.run) -> None:
        self.comfyui = comfyui
        self.git_metadata = git_metadata
        self.run = run
        self._git: dict | None = None
        self._driver: tuple[str | None] | None = None

    def _gpu_driver(self) -> str | None:
        if self._driver is None:
            try:
                result = self.run(
                    ["nvidia-smi", "--query-gpu=driver_version",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10, check=True)
                lines = result.stdout.strip().splitlines()
                self._driver = (lines[0].strip() or None if lines else None,)
            except Exception:
                self._driver = (None,)
        return self._driver[0]

    def __call__(self) -> dict:
        env = {key: None for key in (
            "comfyui_version", "argv", "attention", "pytorch_version",
            "worker_commit", "worker_dirty", "gpu_name", "gpu_driver")}
        try:
            stats = self.comfyui.request("/system_stats")
        except Exception:
            stats = {}
        system = stats.get("system") if isinstance(stats.get("system"), Mapping) else {}
        devices = stats.get("devices") if isinstance(stats.get("devices"), list) else []
        argv = system.get("argv")
        env.update(
            comfyui_version=system.get("comfyui_version"),
            argv=argv if isinstance(argv, list) else None,
            attention=attention_of(argv),
            pytorch_version=system.get("pytorch_version"),
            gpu_name=gpu_name_of(devices[0].get("name")) if devices else None)
        if self._git is None:
            try:
                self._git = self.git_metadata()
            except Exception:
                self._git = {}
        env.update(worker_commit=self._git.get("commit"),
                   worker_dirty=self._git.get("dirty"),
                   gpu_driver=self._gpu_driver())
        return env
