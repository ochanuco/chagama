"""DWPose: the COCO-WholeBody pose of the picture's main figure.

rtmlib's `Wholebody` in its `balanced` mode is what the Anima pose adapter's
skeletons were detected with; it runs on the CPU through onnxruntime.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .models import DWPOSE, model_path

DETECTOR = "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"
ESTIMATOR = "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx"
DETECTOR_INPUT = (640, 640)
ESTIMATOR_INPUT = (192, 256)
BODY = slice(0, 23)


def load_estimator():
    from rtmlib import Wholebody
    folder = model_path(DWPOSE, (DETECTOR, ESTIMATOR))
    return Wholebody(det=str(folder / DETECTOR), det_input_size=DETECTOR_INPUT,
                     pose=str(folder / ESTIMATOR), pose_input_size=ESTIMATOR_INPUT,
                     to_openpose=False, backend="onnxruntime", device="cpu")


def main_figure(keypoints: np.ndarray, scores: np.ndarray) -> int | None:
    """The figure whose body and feet are most surely seen."""
    if len(keypoints) == 0:
        return None
    return int(np.argmax(scores[:, BODY].sum(axis=1)))


def detect(image: Image.Image, estimator) -> dict | None:
    """`{points, aspect}`: 133 `[x, y, score]`, x and y as fractions of the
    picture, or None when no figure is found."""
    width, height = image.size
    keypoints, scores = estimator(np.asarray(image)[..., ::-1].copy())
    keypoints, scores = np.asarray(keypoints), np.asarray(scores)
    figure = main_figure(keypoints, scores)
    if figure is None:
        return None
    points = [[round(float(x) / width, 4), round(float(y) / height, 4),
               round(float(score), 3)]
              for (x, y), score in zip(keypoints[figure], scores[figure])]
    return {"points": points, "aspect": width / height}
