"""CLIPSeg asked for "hair": where the reference's hair lies, so the depth
leaves it out. Yukari's hair comes from her identity tags, never from the
reference, whose hair would otherwise be drawn as whatever else fits."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from .models import HAIR, model_path

FILES = ("config.json", "model.safetensors", "preprocessor_config.json",
         "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
PROMPT = "hair"


def load_estimator() -> Callable[[Image.Image], np.ndarray]:
    """A function from a picture to its float 0..1 hair likelihood, the
    picture's size."""
    import cv2
    import torch
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    folder = model_path(HAIR, FILES)
    processor = CLIPSegProcessor.from_pretrained(folder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPSegForImageSegmentation.from_pretrained(folder).to(device).eval()

    def estimate(image: Image.Image) -> np.ndarray:
        inputs = processor(text=[PROMPT], images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        likelihood = torch.sigmoid(logits).reshape(logits.shape[-2:]).float().cpu().numpy()
        return cv2.resize(likelihood, image.size, interpolation=cv2.INTER_LINEAR)
    return estimate
