#!/usr/bin/env python3
"""Put the Hugging Face and Civitai models in manifests/worker-models.toml on the GPU box.

    python scripts/worker_models.py --comfy-root <ComfyUI dir> [--dry-run]

Every file is checked upstream before anything is downloaded, so a wrong path
fails the run without a partial fetch. A download resumes from its `.part`,
is hashed whole and lands under its final name only when the hash matches
(the SHA256 Hugging Face publishes for an LFS file, the git blob SHA1 for a
small file kept outside LFS).
A Civitai download needs CIVITAI_TOKEN; Hugging Face reads HF_TOKEN when set.
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
CIVITAI = "https://civitai.com"
CHUNK = 8 * 1024 * 1024
# Head room left on the disk after every pending download has landed.
FREE_MARGIN = 2 * 1024**3

Opener = Callable[[urllib.request.Request], object]


@dataclass(frozen=True)
class Model:
    dest: str
    repo: str | None = None
    file: str | None = None
    revision: str = "main"
    sha256: str | None = None
    civitai_model: int | None = None
    civitai_version: int | None = None

    @property
    def is_civitai(self) -> bool:
        return self.civitai_version is not None

    def source(self, commit: str) -> str:
        if self.is_civitai:
            return f"civitai {self.civitai_model}/{self.civitai_version}"
        return f"{self.repo} @ {commit}"


@dataclass(frozen=True)
class Upstream:
    """`digest` is the SHA256 of the file, or its git blob SHA1 when Hugging
    Face keeps the file outside LFS and publishes no SHA256."""
    commit: str
    size: int
    digest: str
    git_blob: bool = False


class FetchError(RuntimeError):
    pass


HF_KEYS = {"repo", "file", "revision"}
CIVITAI_KEYS = {"civitai_model", "civitai_version"}


def _entry_model(entry: dict) -> Model:
    dest = entry.get("dest")
    if not isinstance(dest, str):
        raise FetchError("a model entry has no dest")
    hf, civitai = HF_KEYS & entry.keys(), CIVITAI_KEYS & entry.keys()
    if hf and civitai:
        raise FetchError(f"{dest}: Hugging Face and Civitai keys are mixed")
    sha256 = entry.get("sha256") or None
    if civitai:
        if civitai != CIVITAI_KEYS:
            raise FetchError(f"{dest}: needs civitai_model and civitai_version")
        return Model(dest=dest, sha256=sha256, civitai_model=int(entry["civitai_model"]),
                     civitai_version=int(entry["civitai_version"]))
    if not {"repo", "file"} <= hf:
        raise FetchError(f"{dest}: needs repo and file, or civitai_model and civitai_version")
    return Model(dest=dest, repo=entry["repo"], file=entry["file"],
                 revision=entry.get("revision", "main"), sha256=sha256)


def load_manifest(path: Path) -> list[Model]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    models = []
    for entry in data.get("model", []):
        model = _entry_model(entry)
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
    merged = {"User-Agent": "chagama-worker-models"}
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


def _civitai_request(url: str, *, headers: dict[str, str] | None = None
                     ) -> urllib.request.Request:
    return urllib.request.Request(
        url, headers={"User-Agent": "chagama-worker-models", **(headers or {})})


def _resolve_civitai(model: Model, opener: Opener) -> Upstream:
    info = _json(opener, _civitai_request(
        f"{CIVITAI}/api/v1/model-versions/{model.civitai_version}"))
    if info.get("modelId") != model.civitai_model:
        raise FetchError(f"{model.dest}: civitai version {model.civitai_version} belongs to "
                         f"model {info.get('modelId')}, not {model.civitai_model}")
    primary = next((f for f in info.get("files", []) if f.get("primary")), None)
    sha256 = ((primary or {}).get("hashes") or {}).get("SHA256")
    if not primary or not sha256:
        raise FetchError(f"{model.dest}: civitai version {model.civitai_version} "
                         "has no primary file with a SHA256")
    sha256 = sha256.lower()
    if model.sha256 and model.sha256.lower() != sha256:
        raise FetchError(f"{model.dest}: pinned sha256 {model.sha256[:12]} but civitai "
                         f"version {model.civitai_version} publishes {sha256[:12]}")
    return Upstream(commit=str(model.civitai_version),
                    size=round(primary["sizeKB"] * 1024), digest=sha256)


def resolve(models: list[Model], opener: Opener) -> dict[str, Upstream]:
    """Upstream size and digest of every model, keyed by dest; raises listing
    every file Hugging Face does not have."""
    commits: dict[tuple[str, str], str] = {}
    found: dict[str, Upstream] = {}
    missing = []
    for model in models:
        if model.is_civitai:
            found[model.dest] = _resolve_civitai(model, opener)
            continue
        key = (model.repo, model.revision)
        if key not in commits:
            commits[key] = resolve_commit(model.repo, model.revision, opener)
        commit = commits[key]
        body = json.dumps({"paths": [model.file]}).encode("utf-8")
        infos = _json(opener, _request(
            f"{HF}/api/models/{model.repo}/paths-info/{commit}", data=body,
            headers={"Content-Type": "application/json"}))
        info = next((i for i in infos if i.get("path") == model.file), None)
        if not info:
            missing.append(f"{model.repo}@{commit[:12]} :: {model.file}")
            continue
        lfs = info.get("lfs") or {}
        if lfs.get("oid"):
            upstream = Upstream(commit=commit, size=int(lfs.get("size", info.get("size", 0))),
                                digest=lfs["oid"].lower())
        else:
            upstream = Upstream(commit=commit, size=int(info.get("size", 0)),
                                digest=info["oid"].lower(), git_blob=True)
        if model.sha256 and model.sha256.lower() != upstream.digest:
            raise FetchError(f"{model.dest}: pinned sha256 {model.sha256[:12]} but "
                             f"{model.repo}@{commit[:12]} publishes {upstream.digest[:12]}")
        found[model.dest] = upstream
    if missing:
        raise FetchError("not on Hugging Face:\n  " + "\n  ".join(missing))
    return found


def _hash(path: Path, git_blob: bool = False) -> str:
    if git_blob:
        digest = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    else:
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
        ranged = {"Range": f"bytes={have}-"} if have else None
        if model.is_civitai:
            token = urllib.parse.quote(os.environ["CIVITAI_TOKEN"], safe="")
            request = _civitai_request(
                f"{CIVITAI}/api/download/models/{model.civitai_version}?token={token}",
                headers=ranged)
        else:
            file = urllib.parse.quote(model.file)
            request = _request(f"{HF}/{model.repo}/resolve/{upstream.commit}/{file}",
                               headers=ranged)
        with opener(request) as response:
            status = getattr(response, "status", 200)
            mode = "ab" if have and status == 206 else "wb"
            with part.open(mode) as handle:
                while block := response.read(CHUNK):
                    handle.write(block)
    got = _hash(part, upstream.git_blob)
    if got != upstream.digest:
        part.unlink()
        raise FetchError(f"{model.dest}: digest {got[:12]} != {upstream.digest[:12]}; "
                         "the partial file was removed")
    sidecar = _sidecar(target)
    sidecar.unlink(missing_ok=True)
    os.replace(part, target)
    sidecar.write_text(got + "\n", encoding="ascii")


def _sidecar(target: Path) -> Path:
    return target.with_name(target.name + ".sha256")


def _is_current(target: Path, upstream: Upstream) -> bool:
    """The file at `target` is the upstream content: the published size, and a
    `<target>.sha256` sidecar holding the upstream digest (written here after
    hashing the file once when it is missing)."""
    if not (target.exists() and target.stat().st_size == upstream.size):
        return False
    sidecar = _sidecar(target)
    if sidecar.exists():
        return sidecar.read_text(encoding="ascii").strip() == upstream.digest
    if _hash(target, upstream.git_blob) != upstream.digest:
        return False
    sidecar.write_text(upstream.digest + "\n", encoding="ascii")
    return True


def pending(models: list[Model], found: dict[str, Upstream],
            models_dir: Path) -> list[Model]:
    """Models whose dest does not hold the published content."""
    return [m for m in models if not _is_current(models_dir / m.dest, found[m.dest])]


def _keep_awake() -> contextlib.AbstractContextManager:
    if sys.platform != "win32":
        return contextlib.nullcontext()
    from chagama.infrastructure.power.windows import keep_awake
    return keep_awake()


def sync(models: list[Model], models_dir: Path, *, opener: Opener,
         dry_run: bool = False, disk_free: Callable[[Path], int] | None = None,
         echo: Callable[[str], None] = print) -> bool:
    found = resolve(models, opener)
    for line in sorted({m.source(found[m.dest].commit) for m in models}):
        echo(line)
    todo = pending(models, found, models_dir)
    if not dry_run and not os.environ.get("CIVITAI_TOKEN") \
            and any(m.is_civitai for m in todo):
        raise FetchError("CIVITAI_TOKEN is not set; a Civitai download needs it")
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
            echo(f"{model.dest} fetched {upstream.digest[:12]}")
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
