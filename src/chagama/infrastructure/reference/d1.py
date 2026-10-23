"""d1: a yes/no check of each borderline tag against the picture."""

from __future__ import annotations

import gc
import re
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from .models import VERIFIER, ReferenceUnavailable, model_path

VERIFIER_FILES = ("config.json", "model.safetensors", "modeling_d1.py",
                  "tokenizer.json", "processor_config.json")
QUESTION = ("Does the Danbooru tag '{tag}' accurately describe this illustration "
            "(camera angle, framing, the character's pose, gesture or expression)?")
MIN_TRANSFORMERS = (5, 14)
LONG_SIDE = 1024


def version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:2])


def questions(tags: Sequence[str]) -> dict[str, dict]:
    return {tag: {"type": "noul", "instructions": QUESTION.format(tag=tag)}
            for tag in tags}


class D1Verifier:
    def __init__(self, model_dir: Path, device: str | None = None) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModel
        except ImportError as error:
            raise ReferenceUnavailable(
                f"d1 needs torch and transformers: {error}") from error
        if version_tuple(transformers.__version__) < MIN_TRANSFORMERS:
            raise ReferenceUnavailable(
                f"d1 needs transformers>={'.'.join(map(str, MIN_TRANSFORMERS))}, "
                f"this Python has {transformers.__version__}")
        from .anima_tagger import default_device

        self._torch = torch
        self.device = torch.device(device or default_device())
        self._model = AutoModel.from_pretrained(
            str(model_dir), trust_remote_code=True, dtype=torch.bfloat16,
        ).to(self.device)

    def verify(self, image: Image.Image, tags: Sequence[str]) -> dict[str, float]:
        """P(yes) per tag, all asked in one pass over the picture."""
        if not tags:
            return {}
        picture = image.convert("RGB")
        picture.thumbnail((LONG_SIDE, LONG_SIDE))
        answers = self._model.system_one(
            None, questions(tags), images=[picture])["answers"]
        return {tag: float(answers[tag]["noul"]) for tag in tags}

    def close(self) -> None:
        self.__dict__.pop("_model", None)
        gc.collect()
        if self.device.type == "cuda":
            self._torch.cuda.empty_cache()


def load_verifier(device: str | None = None) -> D1Verifier:
    return D1Verifier(model_path(VERIFIER, VERIFIER_FILES), device=device)
