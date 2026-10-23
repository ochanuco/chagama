"""Anima Tagger: the dbv4 caformer danbooru tagger projected onto the Anima
caption vocabulary, with the checkpoint's thresholds and tag groups.

Vendored from sorryhyun/anime_tools (MIT, see LICENSE.anime_tools beside this
file) at commit d8e8fcb46eeddd098d95978007b31b53853ad0bb, from
anime_tools/tagger/tagger.py and dbv4_backend.py, the dbv4 backend only.
`kept` stops after the per-tag thresholds and the group argmax: the character,
copyright and artist caps that follow upstream do not touch general tags.
"""

from __future__ import annotations

import csv
import gc
import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .models import TAGGER_HEAD, TAGGER_TRUNK, ReferenceUnavailable, model_path

HEAD_FILES = ("config.json", "vocab.json", "groups.yaml", "rules.yaml",
              "thresholds.safetensors", "sidecar.json", "sidecar.safetensors")
TRUNK_FILES = ("model.safetensors", "selected_tags.csv", "meta.json")
RATING_CATEGORY = 9
UNSUPPORTED_LOGIT = -30.0
PROB_EPS = 1e-6
SOFTMAX_MODES = ("softmax", "softmax_when_solo")
SINGLE_COUNT_NAMES = frozenset({"solo", "1girl", "1boy", "1other"})
COUNT_RE = re.compile(
    r"^(?:\d+\+?(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$")
DEFAULT_ARCH = "caformer_b36"
DEFAULT_IMG_SIZE = 384


@dataclass(frozen=True)
class TagGroup:
    name: str
    mode: str
    tags: tuple[str, ...]
    escape: tuple[str, ...] = ()
    sentinel: bool = False


@dataclass(frozen=True)
class Tagging:
    """`scores` holds every vocabulary tag's probability, `kept` the tags the
    checkpoint emits, `groups` the checkpoint's tag groups."""
    scores: dict[str, float]
    kept: dict[str, float]
    groups: tuple[TagGroup, ...]


def parse_groups(raw: Mapping) -> tuple[TagGroup, ...]:
    """`groups.yaml` as loaded: `{name: {mode, tags, escape?, sentinel?}}`."""
    return tuple(
        TagGroup(name=name, mode=str(body.get("mode", "")),
                 tags=tuple(str(tag) for tag in body.get("tags") or ()),
                 escape=tuple(str(tag) for tag in body.get("escape") or ()),
                 sentinel=bool(body.get("sentinel", False)))
        for name, body in raw.items() if name != "version")


def sentinel_name(group: str) -> str:
    return f"<none:{group}>"


def is_sentinel(name: str) -> bool:
    return name.startswith("<none:") and name.endswith(">")


def is_solo(names: Collection[str]) -> bool:
    names = set(names)
    multi = any(name not in SINGLE_COUNT_NAMES and COUNT_RE.match(name)
                for name in names)
    return bool(names & SINGLE_COUNT_NAMES) and not multi


def rename_recovery(replacements: Mapping[str, str]) -> dict[str, str]:
    """`rules.yaml` renames `src -> tgt`; aligning needs `tgt -> src`."""
    return {target: source for source, target in replacements.items()}


def align(names: Sequence[str], columns: Mapping[str, int],
          recovery: Mapping[str, str]) -> tuple[list[int], list[int]]:
    """Vocabulary indices the trunk can emit, and their trunk columns."""
    ours, ext = [], []
    for index, name in enumerate(names):
        column = columns.get(name)
        if column is None and name in recovery:
            column = columns.get(recovery[name])
        if column is not None:
            ours.append(index)
            ext.append(column)
    return ours, ext


def pad_square_white(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """`[3, size, size]` float32 in [0, 1]; the trunk normalises it."""
    square = pad_square_white(image.convert("RGB")).resize(
        (size, size), Image.BICUBIC)
    return (np.asarray(square, dtype=np.float32) / 255.0).transpose(2, 0, 1)


def to_logits(probs: np.ndarray, supported: np.ndarray) -> np.ndarray:
    clamped = np.clip(probs, PROB_EPS, 1.0 - PROB_EPS)
    logits = np.log(clamped) - np.log1p(-clamped)
    logits[~supported] = UNSUPPORTED_LOGIT
    return logits


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def decide(logits: np.ndarray, names: Sequence[str], thresholds: np.ndarray,
           groups: Sequence[TagGroup]) -> dict[str, float]:
    """The emitted tags: every tag over its threshold, then one argmax winner
    per applicable softmax group, which must clear its own threshold too."""
    probs = sigmoid(np.asarray(logits, dtype=np.float64))
    kept = {name: float(p) for name, p, threshold in zip(names, probs, thresholds)
            if p >= threshold and not is_sentinel(name)}
    kept_names = set(kept)
    solo = is_solo(kept_names)
    index_of = {name: index for index, name in enumerate(names)}
    for group in groups:
        if group.mode not in SOFTMAX_MODES:
            continue
        members = [index_of[tag] for tag in group.tags if tag in index_of]
        if not members:
            continue
        sentinel_local = None
        if group.sentinel and sentinel_name(group.name) in index_of:
            sentinel_local = len(members)
            members.append(index_of[sentinel_name(group.name)])
        escaped = bool(kept_names & set(group.escape))
        applicable = not escaped and (solo or group.mode == "softmax")
        if not applicable:
            continue
        winner_local = int(np.argmax(np.asarray(logits)[members]))
        for tag in group.tags:
            kept.pop(tag, None)
        if winner_local == sentinel_local:
            continue
        winner = members[winner_local]
        if probs[winner] < thresholds[winner]:
            continue
        kept[names[winner]] = float(probs[winner])
    return kept


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AnimaTagger:
    def __init__(self, head: Path, trunk: Path, device: str | None = None) -> None:
        try:
            import timm
            import torch
            import yaml
            from safetensors.torch import load_file
        except ImportError as error:
            raise ReferenceUnavailable(
                f"the Anima Tagger needs torch, timm, safetensors and pyyaml: {error}"
            ) from error
        self._torch = torch
        self.device = torch.device(device or default_device())
        # bf16 only pays where the hardware has kernels for it.
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        config = json.loads((head / "config.json").read_text(encoding="utf-8"))
        if config.get("backend") != "dbv4":
            raise ReferenceUnavailable(f"{head} is not a dbv4 tagger checkpoint")
        dbv4 = config.get("dbv4") or {}
        vocab = json.loads((head / "vocab.json").read_text(encoding="utf-8"))
        entries = sorted(vocab["tags"], key=lambda entry: int(entry["index"]))
        self.names = [entry["name"] for entry in entries]
        rules = yaml.safe_load((head / "rules.yaml").read_text(encoding="utf-8")) or {}
        self.groups = parse_groups(
            yaml.safe_load((head / "groups.yaml").read_text(encoding="utf-8")) or {})
        self.thresholds = load_file(str(head / "thresholds.safetensors"))[
            "thresholds"].float().numpy()

        with (trunk / "selected_tags.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        columns = {row["name"].replace("_", " "): column
                   for column, row in enumerate(rows)
                   if int(row["category"]) != RATING_CATEGORY}
        self.ours, self.ext = align(
            self.names, columns, rename_recovery(rules.get("replacements") or {}))
        meta = json.loads((trunk / "meta.json").read_text(encoding="utf-8"))
        model_args = meta.get("model_args") or {}
        kwargs = ({"act_layer": model_args["act_layer"]}
                  if "act_layer" in model_args else {})
        model = timm.create_model(dbv4.get("arch", DEFAULT_ARCH), pretrained=False,
                                  num_classes=len(rows), **kwargs)
        model.load_state_dict(load_file(str(trunk / "model.safetensors")))
        model.to(self.device, dtype=self.dtype).eval()
        cfg = model.pretrained_cfg
        self._mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(
            1, 3, 1, 1).to(self.device)
        self._std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(
            1, 3, 1, 1).to(self.device)
        self._model = model
        self.img_size = int(dbv4.get("img_size", DEFAULT_IMG_SIZE))

        sidecar = json.loads((head / "sidecar.json").read_text(encoding="utf-8"))
        weights = load_file(str(head / "sidecar.safetensors"))
        self._sidecar_weight = weights["fc.weight"].float().to(self.device)
        self._sidecar_bias = weights["fc.bias"].float().to(self.device)
        self.bce = [int(index) for index in sidecar.get("bce_indices", [])]
        self.supported = np.zeros(len(self.names), dtype=bool)
        self.supported[self.ours] = True
        self.supported[self.bce] = True

    def predict(self, image: Image.Image) -> Tagging:
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(preprocess(image, self.img_size))[None].to(
                self.device, dtype=torch.float32)
            x = ((x - self._mean) / self._std).to(self.dtype)
            model = self._model
            pooled = model.forward_head(model.forward_features(x), pre_logits=True)
            fc = model.head.fc
            hidden = fc.norm(fc.act(fc.fc1(pooled)))
            trunk_probs = fc.fc2(hidden).float().sigmoid()[0].cpu().numpy()
            bce = (hidden.float() @ self._sidecar_weight.T + self._sidecar_bias)
            bce_probs = bce.sigmoid()[0, :len(self.bce)].cpu().numpy()
        probs = np.zeros(len(self.names), dtype=np.float64)
        probs[self.ours] = trunk_probs[self.ext]
        probs[self.bce] = bce_probs
        logits = to_logits(probs, self.supported)
        scores = dict(zip(self.names, sigmoid(logits).tolist()))
        kept = decide(logits, self.names, self.thresholds, self.groups)
        return Tagging(scores=scores, kept=kept, groups=self.groups)

    def close(self) -> None:
        cuda = self.device.type == "cuda"
        for name in ("_model", "_sidecar_weight", "_sidecar_bias", "_mean", "_std"):
            self.__dict__.pop(name, None)
        gc.collect()
        if cuda:
            self._torch.cuda.empty_cache()


def load_tagger(device: str | None = None) -> AnimaTagger:
    return AnimaTagger(model_path(TAGGER_HEAD, HEAD_FILES),
                       model_path(TAGGER_TRUNK, TRUNK_FILES), device=device)
