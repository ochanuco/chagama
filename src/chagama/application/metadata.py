"""Generation metadata use cases."""

from __future__ import annotations

import json
from pathlib import Path


def put_semantic(client, generation_id: str, path: Path) -> None:
    client.put_semantic(generation_id, json.loads(path.read_text(encoding="utf-8")))


def add_tag(client, generation_id: str, name: str) -> None:
    client.add_tag(generation_id, name)


def record_publication(client, generation_id: str, url: str | None = None,
                       idempotency_key: str | None = None) -> dict:
    return client.record_publication(
        generation_id, url=url, idempotency_key=idempotency_key)


def upload_asset(client, generation_id: str, role: str, path: Path,
                 region: str = "") -> dict:
    return client.upload_asset(generation_id, role, path, region)


def list_assets(client, generation_id: str) -> list[dict]:
    return client.list_assets(generation_id)
