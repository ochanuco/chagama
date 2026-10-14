"""ComfyUI HTTP adapter."""

from __future__ import annotations

import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image

from .roles import label_roles
from .timings import AttemptTimings, now_ms


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def as_png(data: bytes) -> bytes:
    # LoadImage decodes through PyAV, which reads WebP via YUV and shifts pixels.
    if data.startswith(PNG_SIGNATURE):
        return data
    output = io.BytesIO()
    Image.open(io.BytesIO(data)).save(output, "PNG")
    return output.getvalue()


def images_of(history_entry: dict) -> list[dict]:
    images = []
    for node_output in history_entry.get("outputs", {}).values():
        for image in node_output.get("images", []):
            if image.get("type") == "output":
                images.append(image)
    return images


class ComfyUIClient:
    def __init__(self, base_url: str | None = None, *, poll_interval: int = 10,
                 poll_timeout: int = 20 * 60,
                 timings: AttemptTimings | None = None,
                 client_id: str | None = None) -> None:
        self.timings = timings
        self.client_id = client_id
        self.base_url = (base_url or
                         f"http://{os.environ.get('COMFYUI_HOST', '127.0.0.1')}:"
                         f"{os.environ.get('COMFYUI_PORT', '8188')}").rstrip("/")
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

    def request(self, path: str, payload: dict | None = None) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode() if payload else None,
            headers={"Content-Type": "application/json"} if payload else {},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    def submit(self, graph: dict, purpose: str = "render") -> str:
        try:
            graph = label_roles(graph)
        except Exception:
            pass
        submitted_at = now_ms()
        body = {"prompt": graph}
        if self.client_id:
            body["client_id"] = self.client_id
        prompt_id = self.request("/prompt", body)["prompt_id"]
        if self.timings is not None:
            self.timings.register(prompt_id, graph, purpose=purpose,
                                  submitted_at=submitted_at)
        return prompt_id

    def knows(self, prompt_id: str) -> bool:
        try:
            if self.request(f"/history/{prompt_id}").get(prompt_id):
                known = True
            else:
                queue = self.request("/queue")
                queued_ids = {
                    entry[1] for entry in
                    queue.get("queue_running", []) + queue.get("queue_pending", [])
                }
                known = prompt_id in queued_ids
        except urllib.error.URLError:
            return True
        if known and self.timings is not None:
            self.timings.resumed(prompt_id)
        return known

    def _wait_for_entry(self, prompt_id: str) -> dict:
        deadline = time.time() + self.poll_timeout
        while time.time() < deadline:
            try:
                entry = self.request(f"/history/{prompt_id}").get(prompt_id)
            except urllib.error.URLError:
                entry = None
            if entry:
                status = entry.get("status", {}).get("status_str")
                finished = status in ("error", "success") or images_of(entry)
                if finished and self.timings is not None:
                    self.timings.outputs_ready(prompt_id, entry)
                if status == "error":
                    raise RuntimeError(f"comfy job {prompt_id} failed")
                if finished:
                    return entry
            time.sleep(self.poll_interval)
        raise RuntimeError(f"comfy job {prompt_id} timed out")

    def wait_for(self, prompt_id: str) -> list[dict]:
        return images_of(self._wait_for_entry(prompt_id))

    def wait_for_outputs(self, prompt_id: str) -> dict:
        """Like `wait_for`, but the raw node-keyed outputs instead of only
        images -- e.g. a DWPreprocessor node's `openpose_json` text output,
        which has no image to be found by `wait_for`."""
        return self._wait_for_entry(prompt_id).get("outputs", {})

    def fetch(self, image: dict) -> bytes:
        query = urllib.parse.urlencode({
            "filename": image["filename"],
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        })
        with urllib.request.urlopen(
                self.base_url + "/view?" + query, timeout=120) as response:
            return response.read()

    def upload_image(self, name: str, data: bytes) -> str:
        """Returns the filename ComfyUI stored it as."""
        boundary = uuid.uuid4().hex
        body = b"".join([
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{name}"\r\nContent-Type: image/png\r\n\r\n'.encode(),
            as_png(data),
            f'\r\n--{boundary}\r\nContent-Disposition: form-data; '
            f'name="overwrite"\r\n\r\ntrue\r\n'
            f'--{boundary}--\r\n'.encode(),
        ])
        request = urllib.request.Request(
            self.base_url + "/upload/image", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())["name"]
