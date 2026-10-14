"""One websocket to chimera's WorkerHub; the contract is chimera's docs/worker-protocol.md."""

from __future__ import annotations

import json

from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

from ..ws import WebSocketClosed, websocket_url


class HubClosed(WebSocketClosed):
    pass


def hub_url(base_url: str) -> str:
    return websocket_url(base_url, "/api/v1/worker/ws")


def encode(message: dict) -> str:
    """Compact JSON: the hub's hibernation auto-response matches {"type":"ping"} byte for byte."""
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False)


class HubConnection:
    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self.url = url
        self.headers = headers
        self._socket = None

    def open(self) -> "HubConnection":
        try:
            self._socket = connect(
                self.url, additional_headers=self.headers,
                open_timeout=20, extensions=[])
        except WebSocketException as error:
            raise HubClosed(str(error)) from error
        return self

    def send(self, message: dict) -> None:
        try:
            self._socket.send(encode(message))
        except WebSocketException as error:
            raise HubClosed(str(error)) from error

    def recv(self, timeout: float) -> dict | None:
        try:
            raw = self._socket.recv(timeout=timeout)
        except TimeoutError:
            return None
        except WebSocketException as error:
            raise HubClosed(str(error)) from error
        try:
            message = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return message if isinstance(message, dict) else None

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except WebSocketException:
                pass

