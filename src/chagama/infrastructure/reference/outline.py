"""The figure's silhouette and lineart: control pictures for the Anima
any-test LLLite, both black on white and cut to the figure.

skytnt's anime-seg masks the figure; the lineart is the gap between the
picture's grey and its dilation, kept inside the slightly widened mask.
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from .models import ANIME_SEG, model_path

SEGMENTER = "isnetis.onnx"
SEGMENTER_SIDE = 1024
LONG_SIDE = 768
MASK_THRESHOLD = 0.5
LINE_KERNEL = 3
LINE_GAIN = 4.0
MASK_MARGIN = 9


def load_segmenter():
    import onnxruntime
    path = model_path(ANIME_SEG, (SEGMENTER,)) / SEGMENTER
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


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


def png_base64(picture: np.ndarray) -> str:
    height, width = picture.shape[:2]
    scale = min(1.0, LONG_SIDE / max(height, width))
    image = Image.fromarray(picture)
    if scale < 1:
        image = image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def outlines(image: Image.Image, segmenter) -> dict:
    """`{silhouette, lineart}`: each a base64 greyscale PNG, the long side at
    most `LONG_SIDE`."""
    rgb = np.asarray(image.convert("RGB"))
    mask = figure_mask(rgb, segmenter)
    return {"silhouette": png_base64(silhouette(mask)),
            "lineart": png_base64(lineart(rgb, mask))}
