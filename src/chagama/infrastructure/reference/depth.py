"""Depth Anything V2 Large: the figure's depth, white near, the picture the
Anima VACE depth ControlNet was trained to read."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from .models import DEPTH, model_path

FILES = ("config.json", "model.safetensors", "preprocessor_config.json")


def load_estimator() -> Callable[[Image.Image], np.ndarray]:
    """A function from a picture to its relative depth, larger nearer, the
    picture's size."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    folder = model_path(DEPTH, FILES)
    processor = AutoImageProcessor.from_pretrained(folder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForDepthEstimation.from_pretrained(folder).to(device).eval()

    def estimate(image: Image.Image) -> np.ndarray:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            predicted = model(**inputs).predicted_depth
        resized = torch.nn.functional.interpolate(
            predicted[None], size=image.size[::-1], mode="bicubic")
        return resized[0, 0].float().cpu().numpy()
    return estimate
