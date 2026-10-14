"""Content rating from the WD tagger, as raw probabilities for chimera."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from chagama.envcompat import getenv

MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
MODEL_REVISION = "627aef95638667ddcaa3ac8ae625e88ea5b02f51"
MODEL_ID = f"wd-swinv2-tagger-v3@{MODEL_REVISION}"
FILES = {
    "model.onnx": "e6774bff34d43bd49f75a47db4ef217dce701c9847b546523eb85ff6dbba1db1",
    "selected_tags.csv":
        "298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217",
}
MODEL_DIR_ENV = "CHAGAMA_MODEL_DIR"
RATING_CATEGORY = "9"
GENERAL_CATEGORY = "0"
RATING_KEYS = ("general", "sensitive", "questionable", "explicit")
TAG_CUTOFF = 0.05
USER_AGENT = "chagama-safety/1.0"

_lock = threading.Lock()
_loaded: tuple | None = None


class SafetyUnavailable(RuntimeError):
    """The tagger cannot run here: missing runtime, download or file."""


def model_dir() -> Path:
    override = getenv(MODEL_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / ".local/_nogit/models/wd-tagger"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(name: str, target: Path) -> None:
    url = (f"https://huggingface.co/{MODEL_REPO}/resolve/"
           f"{MODEL_REVISION}/{name}")
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, \
                partial.open("wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise SafetyUnavailable(f"could not download {url}: {error}") from error
    if _sha256(partial) != FILES[name]:
        partial.unlink(missing_ok=True)
        raise SafetyUnavailable(f"{name} downloaded with an unexpected sha256")
    partial.replace(target)


def ensure_files(directory: Path | None = None) -> Path:
    directory = directory or model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = directory / name
        if target.exists() and _sha256(target) == expected:
            continue
        _fetch(name, target)
    return directory


def load_tag_table(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [(row["name"], row["category"]) for row in csv.DictReader(handle)]


def _load() -> tuple:
    global _loaded
    with _lock:
        if _loaded is not None:
            return _loaded
        try:
            import onnxruntime
        except ImportError as error:
            raise SafetyUnavailable("onnxruntime is not installed") from error
        directory = ensure_files()
        session = onnxruntime.InferenceSession(
            str(directory / "model.onnx"), providers=["CPUExecutionProvider"])
        size = int(session.get_inputs()[0].shape[1])
        table = load_tag_table(directory / "selected_tags.csv")
        _loaded = (session, size, table)
        return _loaded


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """Square white-padded BGR float32 NHWC batch, as the WD tagger expects."""
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "transparency" in image.info
                              or image.mode in ("LA", "PA") else "RGB")
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), "white")
    offset = ((side - width) // 2, (side - height) // 2)
    mask = image.getchannel("A") if image.mode == "RGBA" else None
    canvas.paste(image.convert("RGB"), offset, mask)
    if side != size:
        canvas = canvas.resize((size, size), Image.BICUBIC)
    array = np.asarray(canvas, dtype=np.float32)[:, :, ::-1]
    return np.ascontiguousarray(array)[None]


def build_payload(probabilities, table: list[tuple[str, str]]) -> dict:
    rating: dict[str, float] = {}
    tags: dict[str, float] = {}
    for (name, category), probability in zip(table, probabilities):
        if category == RATING_CATEGORY:
            rating[name] = float(probability)
        elif category == GENERAL_CATEGORY and probability >= TAG_CUTOFF:
            tags[name] = round(float(probability), 3)
    return {"model": MODEL_ID,
            "rating": {key: rating[key] for key in RATING_KEYS},
            "tags": tags}


def rate_image(source: Image.Image | Path | str | bytes) -> dict:
    if isinstance(source, bytes):
        source = Image.open(io.BytesIO(source))
    elif not isinstance(source, Image.Image):
        source = Image.open(source)
    session, size, table = _load()
    batch = preprocess(source, size)
    name = session.get_inputs()[0].name
    probabilities = session.run(None, {name: batch})[0][0]
    return build_payload(probabilities, table)
