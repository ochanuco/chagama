"""Starts the worker's claim loop as a thread inside ComfyUI's own process.

Off unless ``CHAGAMA_WORKER`` is set: importing a custom node pack
must never start background work on its own, and this is the switch that
keeps a plain ComfyUI install unaffected.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from chagama.envcompat import getenv

_STARTUP_TIMEOUT = 600.0

_lock = threading.Lock()
_thread: threading.Thread | None = None


def _enabled() -> bool:
    return (getenv("CHAGAMA_WORKER") or "").strip().lower() in {"1", "true", "yes"}


def _worker_id() -> str | None:
    return getenv("CHAGAMA_WORKER_ID") or None


def _repository() -> Path:
    """The checkout this pack is junctioned out of.

    ComfyUI runs from its own directory, so asking git about the working
    directory finds nothing; the pack's own path is what knows the checkout.
    """
    override = getenv("CHAGAMA_REPOSITORY")
    return Path(override) if override else Path(__file__).resolve().parents[2]


def _await_server(base_url: str, deadline: float, sleep: Callable[[float], None],
                  clock: Callable[[], float]) -> bool:
    """Block until ComfyUI answers, because the loop starts before it listens.

    Custom nodes are imported while the server is still coming up, so a claim
    made now would be submitted to a port with nothing behind it and the
    request would be failed for a reason that has nothing to do with it.
    """
    while clock() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/system_stats", timeout=3):
                return True
        except (urllib.error.URLError, OSError):
            sleep(2)
    return False


def _server_ready() -> bool:
    from chagama.infrastructure.comfyui.client import ComfyUIClient

    # The same URL the worker will submit to, resolved the same way.
    return _await_server(ComfyUIClient().base_url,
                         time.monotonic() + _STARTUP_TIMEOUT,
                         time.sleep, time.monotonic)


def _log(repository: Path) -> Callable[[str], None]:
    """Append to a file as well as ComfyUI's console.

    Standing inside ComfyUI, the loop has no console of its own to be read
    later; the standalone worker had watch.log and this replaces it.
    """
    path = repository / ".local/_nogit/worker/comfy-worker.log"

    def emit(message: str) -> None:
        print(f"[yukari_worker] {message}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{stamp} {message}\n")
        except OSError:
            pass

    return emit


def _run_guarded(run: Callable[..., None], ready: Callable[[], bool],
                 worker_id: str | None) -> None:
    try:
        if not ready():
            print("[yukari_worker] ComfyUI never answered; not claiming")
            return
        repository = _repository()
        run(repository, worker_id=worker_id, emit=_log(repository))
    except Exception as error:
        print(f"[yukari_worker] worker thread failed: {error!r}")
        try:
            _log(_repository())(f"worker thread failed: {error!r}")
        except Exception:
            pass


def start(run: Callable[..., None] | None = None,
          ready: Callable[[], bool] | None = None) -> None:
    """Start the claim-loop thread once, if enabled. Never raises."""
    try:
        if not _enabled():
            return
        if run is None:
            # Imported here, not at module scope: a disabled pack must cost
            # ComfyUI nothing but this file, and an import that fails would
            # otherwise report on every start of an install not using it.
            from chagama.interfaces.agent import run as run
        if ready is None:
            ready = _server_ready
        global _thread
        with _lock:
            if _thread is not None:
                return
            thread = threading.Thread(
                target=_run_guarded, args=(run, ready, _worker_id()),
                name="yukari-worker", daemon=True)
            thread.start()
            _thread = thread
    except Exception as error:
        print(f"[yukari_worker] failed to start worker thread: {error!r}")
