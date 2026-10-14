"""The threads that talk to chimera while a request runs: heartbeat, hub
socket and the ComfyUI progress relay onto it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .work import WorkServices


THREAD_STOP_TIMEOUT = 5.0


class Management(Protocol):
    def request(self, method: str, path: str, payload: dict | None = None,
                multipart: tuple[dict, str, str, bytes, str] | None = None) -> dict | None: ...


class Connection(Protocol):
    """What HubListener/ProgressRelay need from an open socket -- already open."""

    def send(self, message: dict) -> None: ...
    def recv(self, timeout: float) -> dict | None: ...
    def close(self) -> None: ...


class Heartbeat:
    """PATCHes {"status": "running"} every `interval` seconds until stopped."""

    def __init__(self, management: Management, row_id: str, worker_id: str, *,
                 interval: float = 30, emit: Callable[[str], None] = print) -> None:
        self.management = management
        self.row_id = row_id
        self.worker_id = worker_id
        self.interval = interval
        self.emit = emit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.management.request(
                    "PATCH", f"/api/v1/requests/{self.row_id}",
                    {"status": "running", "worker_id": self.worker_id})
            except (SystemExit, Exception) as error:
                self.emit(f"heartbeat failed for {self.row_id}: {error}")

    def start(self) -> "Heartbeat":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=THREAD_STOP_TIMEOUT)

    def __enter__(self) -> "Heartbeat":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


class HubListener:
    """Keeps one WorkerHub socket open: hello, ping, wake on `queued`.

    Reconnects with exponential backoff (1s doubling to `backoff_max`),
    reset once the hub sends a frame. send_progress() is a silent no-op
    while no socket is open.
    """

    def __init__(self, services: WorkServices, wake: threading.Event) -> None:
        self.services = services
        self.wake = wake
        self._stop = threading.Event()
        self._connection: Connection | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> "HubListener":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except (SystemExit, Exception):
                pass
        if self._thread is not None:
            self._thread.join(timeout=THREAD_STOP_TIMEOUT)

    def send_progress(self, request_id: str, phase: str, **fields: object) -> None:
        with self._lock:
            connection = self._connection
        if connection is None:
            return
        message = {"type": "progress", "request_id": request_id, "phase": phase}
        message.update({key: value for key, value in fields.items() if value is not None})
        try:
            connection.send(message)
        except (SystemExit, Exception):
            pass

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            connection: Connection | None = None
            try:
                connection = self.services.hub()
                with self._lock:
                    self._connection = connection
                connection.send({
                    "type": "hello", "worker_id": self.services.worker_id,
                    "kinds": list(self.services.kinds),
                })
                self.wake.set()
                next_ping = self.services.clock() + self.services.ping_interval
                while not self._stop.is_set():
                    timeout = max(0.0, next_ping - self.services.clock())
                    message = connection.recv(timeout)
                    if message is None:
                        connection.send({"type": "ping"})
                        next_ping = self.services.clock() + self.services.ping_interval
                        continue
                    backoff = 1.0
                    if message.get("type") == "queued":
                        self.wake.set()
            except (SystemExit, Exception) as error:
                self.services.emit(f"hub connection lost: {error}")
            finally:
                with self._lock:
                    if self._connection is connection:
                        self._connection = None
                if connection is not None:
                    try:
                        connection.close()
                    except (SystemExit, Exception):
                        pass
            if self._stop.is_set():
                return
            self.services.sleep(backoff)
            backoff = min(backoff * 2, self.services.backoff_max)


class ProgressRelay:
    """Reads ComfyUI's own /ws events: every one feeds the timing recorder,
    and `progress` events are relayed to the hub for `current`.

    `current` is the request_id being executed; progress outside that window
    is not relayed. With no listener the feed only records. Never raises
    into the main loop.
    """

    def __init__(self, services: WorkServices, listener: HubListener | None) -> None:
        self.services = services
        self.listener = listener
        self.current: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "ProgressRelay":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=THREAD_STOP_TIMEOUT)

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            feed: Connection | None = None
            try:
                feed = self.services.progress_feed()
                backoff = 1.0
                while not self._stop.is_set():
                    event = feed.recv(self.services.ping_interval)
                    if event is None:
                        continue
                    timings = self.services.timings
                    if timings is not None:
                        timings.recorder.feed(event)
                    if (self.current is not None and self.listener is not None
                            and event.get("type", "progress") == "progress"):
                        self.listener.send_progress(
                            self.current, "sampling",
                            step=event.get("step"), total=event.get("total"))
            except (SystemExit, Exception) as error:
                self.services.emit(f"progress feed lost: {error}")
                if self.services.timings is not None:
                    self.services.timings.invalidate_env()
            finally:
                if feed is not None:
                    try:
                        feed.close()
                    except (SystemExit, Exception):
                        pass
            if self._stop.is_set():
                return
            self.services.sleep(backoff)
            backoff = min(backoff * 2, self.services.backoff_max)
