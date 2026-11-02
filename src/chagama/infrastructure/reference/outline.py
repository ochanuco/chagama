"""The figure's silhouette, lineart and depth: control pictures cut to the
figure. The silhouette and lineart are black on white for the Anima any-test
LLLite; the depth is white near on black for the Anima VACE depth.

skytnt's anime-seg masks the figure, and BiRefNet lite does where anime-seg
finds no one (a photo); the lineart is the gap between the
picture's grey and its dilation, kept inside the slightly widened mask; the
depth is spread over `DEPTH_RANGE` between the figure's nearest and farthest
percentile, so the farthest of the figure still stands off the black.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Callable, Sequence

import cv2
import numpy as np
from PIL import Image

from .models import ANIME_SEG, PHOTO_SEG, model_path

Segment = Callable[[np.ndarray], np.ndarray]

SEGMENTER = "isnetis.onnx"
SEGMENTER_SIDE = 1024
PHOTO_SEGMENTER = "model.onnx"
PHOTO_SIDE = 1024
PHOTO_MEAN = (0.485, 0.456, 0.406)
PHOTO_STD = (0.229, 0.224, 0.225)
LONG_SIDE = 768
MASK_THRESHOLD = 0.5
LINE_KERNEL = 3
LINE_GAIN = 4.0
MASK_MARGIN = 9
DEPTH_RANGE = (0.15, 1.0)
DEPTH_PERCENTILES = (1, 99)


def _session(folder: str, name: str):
    import onnxruntime
    path = model_path(folder, (name,)) / name
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def load_segmenters() -> list[tuple[str, Segment]]:
    """`(name, segment)` pairs, tried in order until one finds a figure; each
    model is loaded on its first use."""
    sessions: dict[str, object] = {}

    def anime(rgb: np.ndarray) -> np.ndarray:
        if "anime" not in sessions:
            sessions["anime"] = _session(ANIME_SEG, SEGMENTER)
        return figure_mask(rgb, sessions["anime"])

    def photo(rgb: np.ndarray) -> np.ndarray:
        if "photo" not in sessions:
            sessions["photo"] = _session(PHOTO_SEG, PHOTO_SEGMENTER)
        return photo_mask(rgb, sessions["photo"])
    return [("anime", anime), ("photo", photo)]


def photo_mask(rgb: np.ndarray, segmenter) -> np.ndarray:
    """BiRefNet's float 0..1 mask of the figure, the picture's size."""
    height, width = rgb.shape[:2]
    scaled = cv2.resize(rgb, (PHOTO_SIDE, PHOTO_SIDE),
                        interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    scaled = (scaled - np.float32(PHOTO_MEAN)) / np.float32(PHOTO_STD)
    feed = {segmenter.get_inputs()[0].name: scaled.transpose(2, 0, 1)[None]}
    logits = segmenter.run(None, feed)[-1][0, 0]
    return cv2.resize(1 / (1 + np.exp(-logits)), (width, height))


def figure_mask(rgb: np.ndarray, segmenter) -> np.ndarray:
    """A float 0..1 mask of the figure, the picture's size."""
    height, width = rgb.shape[:2]
    scale = SEGMENTER_SIDE / max(height, width)
    inner_h, inner_w = round(height * scale), round(width * scale)
    top, left = (SEGMENTER_SIDE - inner_h) // 2, (SEGMENTER_SIDE - inner_w) // 2
    canvas = np.zeros((SEGMENTER_SIDE, SEGMENTER_SIDE, 3), np.float32)
    canvas[top:top + inner_h, left:left + inner_w] = cv2.resize(
        rgb, (inner_w, inner_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    (mask,) = segmenter.run(None, {"img": canvas.transpose(2, 0, 1)[None]})[0]
    return cv2.resize(mask[0, top:top + inner_h, left:left + inner_w], (width, height))


def silhouette(mask: np.ndarray) -> np.ndarray:
    return np.where(mask >= MASK_THRESHOLD, 0, 255).astype(np.uint8)


def lineart(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((LINE_KERNEL, LINE_KERNEL), np.uint8)
    lines = np.clip((cv2.dilate(grey, kernel).astype(np.float32) - grey) * LINE_GAIN, 0, 255)
    inside = cv2.dilate((mask >= MASK_THRESHOLD).astype(np.uint8),
                        np.ones((MASK_MARGIN, MASK_MARGIN), np.uint8))
    return (255 - lines * inside).astype(np.uint8)


def depth(relative: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inside = mask >= MASK_THRESHOLD
    far, near = np.percentile(relative[inside], DEPTH_PERCENTILES)
    spread = np.clip((relative - far) / max(near - far, 1e-6), 0, 1)
    low, high = DEPTH_RANGE
    return np.where(inside, (low + spread * (high - low)) * 255, 0).astype(np.uint8)


def png_base64(picture: np.ndarray) -> str:
    height, width = picture.shape[:2]
    scale = min(1.0, LONG_SIDE / max(height, width))
    image = Image.fromarray(picture)
    if scale < 1:
        image = image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def outlines(image: Image.Image, segmenters: Sequence[tuple[str, Segment]],
             estimate_depth) -> dict:
    """`{cut, silhouette, lineart, depth}`: `cut` names the segmenter that
    found the figure, and each picture is a base64 greyscale PNG, the long
    side at most `LONG_SIDE`; just `{cut: None}` when none found it.
    `estimate_depth` maps the picture to its relative depth, larger nearer."""
    rgb = np.asarray(image.convert("RGB"))
    for name, segment in segmenters:
        mask = segment(rgb)
        if (mask >= MASK_THRESHOLD).any():
            break
    else:
        return {"cut": None}
    return {"cut": name, "silhouette": png_base64(silhouette(mask)),
            "lineart": png_base64(lineart(rgb, mask)),
            "depth": png_base64(depth(estimate_depth(image.convert("RGB")), mask))}
