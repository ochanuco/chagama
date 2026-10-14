"""ComfyUI graph for the DWPose pass that locates a repair source's hands/feet."""

from __future__ import annotations

import json
from collections.abc import Mapping


def pose_graph(image_name: str, prefix: str = "pose") -> dict:
    """DWPreprocessor on an already-uploaded image; the SaveImage exists only
    so the graph has an output node to wait on."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "DWPreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": "enable", "detect_body": "enable",
            "detect_face": "disable", "resolution": 1024,
            # The yolox bbox detectors return zero people on this worker;
            # "None" falls back to the whole-frame pose estimator.
            "bbox_detector": "None",
            "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
            "scale_stick_for_xinsr_cn": "disable"}},
        "3": {"class_type": "SaveImage", "inputs": {
            "images": ["2", 0], "filename_prefix": f"{prefix}-pose"}},
    }


def pose_from_outputs(outputs: Mapping) -> Mapping:
    """The first frame of DWPreprocessor node 2's `openpose_json` text output."""
    try:
        frames = json.loads(outputs["2"]["openpose_json"][0])
    except (KeyError, IndexError) as error:
        raise SystemExit(
            "the pose pass returned no keypoints (DWPreprocessor node 2 "
            "produced no openpose_json; a cached node reports none)") from error
    return frames[0]
