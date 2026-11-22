"""Chimera Management API adapter."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from ..locations import config_dir

USER_AGENT = "chagama-generate/1.0 (+local)"
OP_ITEM = "yml6r5qgx3zt57pryokgi3xdqy"
# chimera answers 503 d1_overloaded while its D1 is unavailable, which lasts a minute or two.
OVERLOAD_BUDGET_SECONDS = 180


def overload_wait(error: urllib.error.HTTPError, detail: bytes) -> float | None:
    """Seconds chimera asked to wait when its database was briefly unavailable."""
    if error.code != 503:
        return None
    try:
        code = (json.loads(detail).get("error") or {}).get("code")
    except (ValueError, AttributeError):
        return None
    if code != "d1_overloaded":
        return None
    try:
        return max(1.0, float(error.headers.get("Retry-After") or 30))
    except ValueError:
        return 30.0


class ChimeraClient:
    def __init__(self, repository: Path, base_url: str | None = None) -> None:
        self.repository = repository
        self.base_url = (base_url or os.environ.get(
            "CHIMERA_BASE_URL", "https://chimera.chanu.co")).rstrip("/")
        self.token_cache = config_dir(repository) / "chimera-token"
        self._credentials: dict[str, str] | None = None
        self._preset_cache: dict[tuple[str, str, str, int], dict] = {}

    @staticmethod
    def _retryable(method: str, payload: dict | None,
                   multipart: tuple[dict, str, str, bytes, str] | None) -> bool:
        """Only retry requests whose side effect is explicitly repeatable."""
        if method in {"GET", "HEAD", "OPTIONS", "PUT", "DELETE"}:
            return True
        if payload and payload.get("idempotency_key"):
            return True
        return bool(multipart and multipart[0].get("idempotency_key"))

    def credentials(self) -> dict[str, str]:
        if self._credentials is not None:
            return self._credentials
        client_id = os.environ.get("CHIMERA_CF_CLIENT_ID", "").strip()
        secret = os.environ.get("CHIMERA_CF_CLIENT_SECRET", "").strip()
        if not (client_id and secret) and self.token_cache.exists():
            client_id, secret, *_ = (
                self.token_cache.read_text(encoding="utf-8").splitlines() + ["", ""])
            client_id, secret = client_id.strip(), secret.strip()
        if not (client_id and secret):
            def field(label: str) -> str:
                return subprocess.run(
                    ["op", "item", "get", OP_ITEM, "--fields", f"label={label}",
                     "--reveal"], check=True, capture_output=True, text=True,
                ).stdout.strip()
            client_id = field("CF-Access-Client-Id")
            secret = field("CF-Access-Client-Secret")
            self.token_cache.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.token_cache.parent.chmod(0o700)
            self.token_cache.write_text(f"{client_id}\n{secret}\n", encoding="utf-8")
            self.token_cache.chmod(0o600)
        self._credentials = {
            "CF-Access-Client-Id": client_id,
            "CF-Access-Client-Secret": secret,
        }
        return self._credentials

    def request(self, method: str, path: str, payload: dict | None = None,
                multipart: tuple[dict, str, str, bytes, str] | None = None) -> dict | None:
        method = method.upper()
        headers = {**self.credentials(), "User-Agent": USER_AGENT}
        if multipart:
            meta, field, filename, data, content_type = multipart
            boundary = uuid.uuid4().hex
            body = b"".join([
                f'--{boundary}\r\nContent-Disposition: form-data; '
                f'name="metadata"\r\nContent-Type: application/json\r\n\r\n'
                f'{json.dumps(meta)}\r\n'.encode(),
                f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
                f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode(),
                data,
                f"\r\n--{boundary}--\r\n".encode(),
            ])
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        elif payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        else:
            body = None
        retryable = self._retryable(method, payload, multipart)
        attempts = 3 if retryable else 1
        overload_waited = 0.0
        attempt = 0
        while attempt < attempts:
            request = urllib.request.Request(
                self.base_url + path, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    raw = response.read()
                    if response.status == 204 or not raw:
                        return None
                    if "json" not in (response.headers.get("Content-Type") or ""):
                        raise SystemExit(
                            f"{method} {path}: non-JSON response -- Cloudflare "
                            "Access rejected the service token")
                    return json.loads(raw)
            except urllib.error.HTTPError as error:
                detail = error.read()[:300]
                if error.code < 500:
                    raise SystemExit(
                        f"{method} {path}: HTTP {error.code} {detail!r}")
                last = f"HTTP {error.code} {detail!r}"
                wait = overload_wait(error, detail) if retryable else None
                if wait is not None and overload_waited + wait <= OVERLOAD_BUDGET_SECONDS:
                    time.sleep(wait)
                    overload_waited += wait
                    continue
            except urllib.error.URLError as error:
                last = str(error)
            attempt += 1
            if attempt < attempts:
                time.sleep(2 ** attempt)
        raise SystemExit(f"{method} {path}: giving up after retries ({last})")

    def fetch_generation_image(self, generation_id: str) -> bytes:
        return self._fetch_generation_bytes(generation_id, "image")

    def fetch_generation_preview(self, generation_id: str) -> bytes:
        return self._fetch_generation_bytes(generation_id, "preview")

    def _read_waiting_out_overload(self, url: str) -> bytes:
        """GET `url`, waiting while chimera's database is briefly unavailable."""
        waited = 0.0
        while True:
            request = urllib.request.Request(
                url, headers={**self.credentials(), "User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                detail = error.read()[:300]
                wait = overload_wait(error, detail)
                if wait is None or waited + wait > OVERLOAD_BUDGET_SECONDS:
                    raise
                time.sleep(wait)
                waited += wait

    def _fetch_generation_bytes(self, generation_id: str, kind: str) -> bytes:
        return self._read_waiting_out_overload(
            f"{self.base_url}/g/{generation_id}/{kind}")

    def _reference_image_request(self, request_id: str, method: str
                                 ) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}/api/v1/requests/"
            f"{urllib.parse.quote(request_id, safe='')}/reference-image",
            headers={**self.credentials(), "User-Agent": USER_AGENT}, method=method)

    def fetch_reference_image(self, request_id: str) -> bytes | None:
        """The uploaded reference picture, or None when chimera holds none."""
        try:
            with urllib.request.urlopen(
                    self._reference_image_request(request_id, "GET"),
                    timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise

    def delete_reference_image(self, request_id: str) -> None:
        with urllib.request.urlopen(
                self._reference_image_request(request_id, "DELETE"), timeout=60):
            pass

    def fetch_asset(self, generation_id: str, role: str) -> bytes | None:
        """The stored bytes of the generation's asset, or None when it has none."""
        try:
            return self._read_waiting_out_overload(
                f"{self.base_url}/g/{generation_id}/assets/{urllib.parse.quote(role, safe='')}")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise

    def put_semantic(self, generation_id: str, semantic: dict) -> dict:
        semantic.setdefault("schema_version", 1)
        semantic.setdefault("generated_by", {
            "provider": "claude-code",
            "model": os.environ.get("CLAUDE_MODEL", "unspecified"),
        })
        return self.request(
            "PUT", f"/api/v1/generations/{generation_id}/semantic", semantic)

    def put_safety(self, generation_id: str, safety: dict) -> dict:
        return self.request(
            "PUT", f"/api/v1/generations/{generation_id}/safety", safety)

    def put_catalog(self, recipe_ref: str, catalog: dict) -> dict:
        return self.request(
            "PUT",
            f"/api/v1/catalogs/{urllib.parse.quote(recipe_ref, safe='/')}",
            catalog)

    def add_tag(self, generation_id: str, name: str) -> dict:
        return self.request(
            "POST", f"/api/v1/generations/{generation_id}/tags",
            {"name": name, "created_by": "claude"})

    def record_publication(self, generation_id: str, url: str | None = None,
                           published_at: str | None = None,
                           idempotency_key: str | None = None) -> dict:
        payload = {"idempotency_key": idempotency_key or str(uuid.uuid4())}
        if url is not None:
            payload["url"] = url
        if published_at is not None:
            payload["published_at"] = published_at
        return self.request(
            "POST", f"/api/v1/generations/{generation_id}/publications", payload)

    def upload_asset(self, generation_id: str, role: str, path: Path,
                     region: str = "", idempotency_key: str | None = None) -> dict:
        content_types = {
            ".png": "image/png", ".json": "application/json",
            ".psd": "image/vnd.adobe.photoshop",
        }
        metadata = {"role": role,
                    "idempotency_key": idempotency_key or str(uuid.uuid4())}
        if region:
            metadata["region"] = region
        return self.request(
            "POST", f"/api/v1/generations/{generation_id}/assets",
            multipart=(metadata, "file", path.name, path.read_bytes(),
                       content_types.get(path.suffix.lower(),
                                         "application/octet-stream")))

    def list_assets(self, generation_id: str) -> list[dict]:
        return self.request(
            "GET", f"/api/v1/generations/{generation_id}/assets").get("assets", [])

    def get_preset(self, recipe: str, kind: str, name: str, version: int) -> dict:
        # Preset versions are immutable, so the cache never needs invalidating.
        key = (recipe, kind, name, version)
        if key not in self._preset_cache:
            segments = "/".join(
                urllib.parse.quote(str(part), safe="")
                for part in (recipe, kind, name, version))
            self._preset_cache[key] = self.request(
                "GET", f"/api/v1/presets/{segments}")
        return self._preset_cache[key]

    def resolve_character(self, name: str) -> str:
        listing = self.request("GET", "/api/v1/characters")
        items = listing.get("items", listing if isinstance(listing, list) else [])
        for item in items:
            if item.get("name") == name or name in (item.get("aliases") or []):
                return item["id"]
        return self.request(
            "POST", "/api/v1/characters", {"name": name, "aliases": []})["id"]
