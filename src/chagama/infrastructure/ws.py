"""Shared pieces of the thin websocket wrappers (hub, comfyui progress)."""

from __future__ import annotations


class WebSocketClosed(Exception):
    """A Connection's socket is closed or broken; the caller should reconnect."""


def websocket_url(base_url: str, path: str) -> str:
    url = base_url.rstrip("/")
    if url.startswith("https://"):
        return "wss://" + url[len("https://"):] + path
    if url.startswith("http://"):
        return "ws://" + url[len("http://"):] + path
    raise ValueError(f"unsupported base_url scheme: {base_url!r}")
