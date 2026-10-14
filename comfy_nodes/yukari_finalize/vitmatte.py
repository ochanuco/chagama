"""The matting model and foreground estimator `imaging.matting` is handed.

torch, transformers and pymatting are imported on first use, so the node
pack still imports where they are absent.
"""

from __future__ import annotations

import numpy as np

from chagama.domain.yukari import delivery_style

_loaded: dict = {}


def _device():
    try:
        import comfy.model_management as model_management
        return model_management.get_torch_device()
    except ImportError:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model():
    if "model" not in _loaded:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        name, revision = delivery_style.MATTING_MODEL, delivery_style.MATTING_REVISION
        _loaded["processor"] = VitMatteImageProcessor.from_pretrained(
            name, revision=revision)
        _loaded["model"] = VitMatteForImageMatting.from_pretrained(
            name, revision=revision).eval()
    return _loaded["processor"], _loaded["model"]


def predict(rgb: np.ndarray, known: np.ndarray) -> np.ndarray:
    import torch
    from PIL import Image
    processor, model = _model()
    device = _device()
    model.to(device)
    inputs = processor(images=Image.fromarray(rgb), trimaps=Image.fromarray(known),
                       return_tensors="pt").to(device)
    with torch.no_grad():
        alpha = model(**inputs).alphas[0, 0].float().cpu().numpy()
    height, width = known.shape
    return alpha[:height, :width]


def release() -> None:
    """Back to the CPU: ComfyUI's own memory management does not see this
    model, and the drawing model needs the GPU next."""
    if "model" in _loaded:
        import torch
        _loaded["model"].to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def foreground(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    from pymatting import estimate_foreground_ml
    return estimate_foreground_ml(rgb, alpha)
