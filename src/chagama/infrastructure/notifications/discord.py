"""Discord webhook notification adapter."""

from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from ..chimera.client import USER_AGENT


class DiscordNotifier:
    def __init__(self, repository: Path) -> None:
        self.webhook_file = repository / ".local/discord-webhook"

    def _webhook(self) -> str:
        url = os.environ.get("DISCORD_WEBHOOK", "").strip()
        if url:
            return url
        if self.webhook_file.exists():
            return self.webhook_file.read_text(encoding="utf-8").strip()
        raise SystemExit(
            f"no webhook: set $DISCORD_WEBHOOK or write one to {self.webhook_file}")

    def send(self, content: str, filename: str, image: bytes) -> None:
        try:
            url = self._webhook()
        except SystemExit:
            print("  ! no Discord webhook configured, skipping notification")
            return
        boundary = uuid.uuid4().hex
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body = b"".join([
            f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n'
            f'Content-Type: application/json\r\n\r\n'
            f'{json.dumps({"content": content})}\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="files[0]"; '
            f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode(),
            image,
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        request = urllib.request.Request(url, data=body, headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": USER_AGENT,
        })
        try:
            with urllib.request.urlopen(request, timeout=120):
                pass
        except urllib.error.HTTPError as error:
            print(f"  ! Discord: HTTP {error.code} {error.read()[:200]!r}")
        except (urllib.error.URLError, OSError) as error:
            print(f"  ! Discord: {error}")


class NullNotifier:
    def send(self, content: str, filename: str, image: bytes) -> None:
        return None
