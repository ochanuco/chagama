#!/usr/bin/env python3
"""Put the Hugging Face models in manifests/worker-models.toml on the GPU box.

    python scripts/worker_models.py --comfy-root <ComfyUI dir> [--dry-run]

Every file is checked upstream before anything is downloaded, so a wrong path
fails the run without a partial fetch. A download resumes from its `.part`,
is hashed whole and lands under its final name only when the hash matches.
Stdlib only: it runs in the checkout's venv on the box.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tomllib
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

HF = "https://huggingface.co"
CHUNK = 8 * 1024 * 1024
# Head room left on the disk after every pending download has landed.
FREE_MARGIN = 2 * 1024**3

Opener = Callable[[urllib.request.Request], object]


@dataclass(frozen=True)
class Model:
    dest: str
    repo: str
    file: str
    revision: str
    sha256: str | None = None


@dataclass(frozen=True)
class Upstream:
    commit: str
    size: int
    sha256: str


class FetchError(RuntimeError):
    pass


def load_manifest(path: Path) -> list[Model]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    models = []
    for entry in data.get("model", []):
        model = Model(dest=entry["dest"], repo=entry["repo"], file=entry["file"],
                      revision=entry.get("revision", "main"),
                      sha256=(entry.get("sha256") or None))
        dest = PurePosixPath(model.dest)
        if dest.is_absolute() or ".." in dest.parts or len(dest.parts) < 2:
            raise FetchError(f"dest must be <folder>/<file> under models/: {model.dest}")
        models.append(model)
    dests = [m.dest for m in models]
    if len(dests) != len(set(dests)):
        raise FetchError("a dest is listed twice")
    return models


def _request(url: str, *, data: bytes | None = None,
             headers: dict[str, str] | None = None) -> urllib.request.Request:
    merged = {"User-Agent": "comfyui-recipes-worker-models"}
    token = os.environ.get("HF_TOKEN")
    if token:
        merged["Authorization"] = f"Bearer {token}"
    merged.update(headers or {})
    return urllib.request.Request(url, data=data, headers=merged)


def _json(opener: Opener, request: urllib.request.Request):
    with opener(request) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_commit(repo: str, revision: str, opener: Opener) -> str:
    quoted = urllib.parse.quote(revision, safe="")
    return _json(opener, _request(f"{HF}/api/models/{repo}/revision/{quoted}"))["sha"]


def resolve(models: list[Model], opener: Opener) -> dict[str, Upstream]:
    """Upstream size and SHA256 of every model, keyed by dest; raises listing
    every file Hugging Face does not have."""
    commits: dict[tuple[str, str], str] = {}
    found: dict[str, Upstream] = {}
    missing = []
    for model in models:
        key = (model.repo, model.revision)
        if key not in commits:
            commits[key] = resolve_commit(model.repo, model.revision, opener)
        commit = commits[key]
        body = json.dumps({"paths": [model.file]}).encode("utf-8")
        infos = _json(opener, _request(
            f"{HF}/api/models/{model.repo}/paths-info/{commit}", data=body,
            headers={"Content-Type": "application/json"}))
        info = next((i for i in infos if i.get("path") == model.file), None)
        lfs = (info or {}).get("lfs") or {}
        if not info or not lfs.get("oid"):
            missing.append(f"{model.repo}@{commit[:12]} :: {model.file}")
            continue
        if model.sha256 and model.sha256.lower() != lfs["oid"].lower():
            raise FetchError(f"{model.dest}: pinned sha256 {model.sha256[:12]} but "
                             f"{model.repo}@{commit[:12]} publishes {lfs['oid'][:12]}")
        found[model.dest] = Upstream(commit=commit, size=int(lfs.get("size", info.get("size", 0))),
                                     sha256=lfs["oid"].lower())
    if missing:
        raise FetchError("not on Hugging Face as an LFS file:\n  " + "\n  ".join(missing))
    return found


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def download(model: Model, upstream: Upstream, target: Path, opener: Opener) -> None:
    """Fetch into `<target>.part`, resuming what is there, and rename it to
    `target` once its SHA256 matches."""
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    if have > upstream.size:
        part.unlink()
        have = 0
    if have < upstream.size:
        file = urllib.parse.quote(model.file)
        request = _request(f"{HF}/{model.repo}/resolve/{upstream.commit}/{file}",
                           headers={"Range": f"bytes={have}-"} if have else None)
        with opener(request) as response:
            status = getattr(response, "status", 200)
            mode = "ab" if have and status == 206 else "wb"
            with part.open(mode) as handle:
                while block := response.read(CHUNK):
                    handle.write(block)
    got = _hash(part)
    if got != upstream.sha256:
        part.unlink()
        raise FetchError(f"{model.dest}: sha256 {got[:12]} != {upstream.sha256[:12]}; "
                         "the partial file was removed")
    os.replace(part, target)


def pending(models: list[Model], found: dict[str, Upstream],
            models_dir: Path) -> list[Model]:
    """Models not yet at their dest at the published size."""
    out = []
    for model in models:
        target = models_dir / model.dest
        if not (target.exists() and target.stat().st_size == found[model.dest].size):
            out.append(model)
    return out


def _keep_awake() -> contextlib.AbstractContextManager:
    if sys.platform != "win32":
        return contextlib.nullcontext()
    from comfyui_recipes.infrastructure.power.windows import keep_awake
    return keep_awake()


def sync(models: list[Model], models_dir: Path, *, opener: Opener,
         dry_run: bool = False, disk_free: Callable[[Path], int] | None = None,
         echo: Callable[[str], None] = print) -> bool:
    found = resolve(models, opener)
    for repo, commit in sorted({(m.repo, found[m.dest].commit) for m in models}):
        echo(f"{repo} @ {commit}")
    todo = pending(models, found, models_dir)
    for model in models:
        if model not in todo:
            echo(f"{model.dest} ok")
    if not todo:
        echo("changed: False")
        return False
    need = 0
    for model in todo:
        part = models_dir / (model.dest + ".part")
        need += found[model.dest].size - (part.stat().st_size if part.exists() else 0)
    models_dir.mkdir(parents=True, exist_ok=True)
    free = (disk_free or (lambda p: shutil.disk_usage(p).free))(models_dir)
    if free < need + FREE_MARGIN:
        raise FetchError(f"{need / 1024**3:.1f} GB to fetch, {free / 1024**3:.1f} GB free")
    for model in todo:
        echo(f"{model.dest} {'would fetch' if dry_run else 'fetching'} "
             f"{found[model.dest].size / 1024**3:.2f} GB")
    if dry_run:
        echo("changed: False")
        return False
    with _keep_awake():
        for model in todo:
            upstream = found[model.dest]
            download(model, upstream, models_dir / model.dest, opener)
            echo(f"{model.dest} fetched {upstream.sha256[:12]}")
    echo("changed: True")
    return True


def _urlopen(request: urllib.request.Request):
    return urllib.request.urlopen(request, timeout=60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comfy-root", required=True, type=Path,
                        help="the ComfyUI directory holding models/")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "manifests" / "worker-models.toml")
    parser.add_argument("--dry-run", action="store_true",
                        help="check upstream and disk, download nothing")
    args = parser.parse_args(argv)
    models_dir = args.comfy_root / "models"
    if not args.comfy_root.is_dir():
        print(f"not found: {args.comfy_root}", file=sys.stderr)
        return 2
    try:
        sync(load_manifest(args.manifest), models_dir, opener=_urlopen,
             dry_run=args.dry_run)
    except (FetchError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
