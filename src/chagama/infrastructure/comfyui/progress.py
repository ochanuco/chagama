"""ComfyUI's own broadcast socket, read for execution and sampling events.

on a socket scoped by `clientId`: ComfyUI sends the `executing` family only
to the client that submitted the prompt, so the worker submits with the same
id. Binary frames (previews) are skipped.
"""

from __future__ import annotations

import json
import time
import urllib.parse

from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

from ..ws import WebSocketClosed, websocket_url


EVENT_TYPES = frozenset({
    "execution_start", "execution_cached", "executing", "progress", "executed",
    "execution_success", "execution_error", "execution_interrupted",
    "progress_state",
})


def now_ms() -> int:
    return int(time.time() * 1000)


class FeedClosed(WebSocketClosed):
    pass


class ProgressFeed:
    def __init__(self, base_url: str, *, client_id: str | None = None,
                 clock_ms=now_ms) -> None:
        self.clock_ms = clock_ms
        self.url = websocket_url(base_url, "/ws")
        if client_id:
            self.url += "?" + urllib.parse.urlencode({"clientId": client_id})
        self._socket = None

    def open(self) -> "ProgressFeed":
        try:
            self._socket = connect(self.url, open_timeout=20, extensions=[])
        except WebSocketException as error:
            raise FeedClosed(str(error)) from error
        return self

    def recv(self, timeout: float) -> dict | None:
        try:
            raw = self._socket.recv(timeout=timeout)
        except TimeoutError:
            return None
        except WebSocketException as error:
            raise FeedClosed(str(error)) from error
        if not isinstance(raw, str):
            return None
        received_at = self.clock_ms()
        try:
            message = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(message, dict) or message.get("type") not in EVENT_TYPES:
            return None
        data = message.get("data")
        data = data if isinstance(data, dict) else {}
        return {"type": message["type"], "prompt_id": data.get("prompt_id"),
                "node": data.get("node"), "nodes": data.get("nodes"),
                "step": data.get("value"), "total": data.get("max"),
                "received_at": received_at}

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except WebSocketException:
                pass
