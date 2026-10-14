"""Tests for the DWPose pass graph and its output parsing."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from chagama.infrastructure.comfyui.pose_graph import (
    pose_from_outputs,
    pose_graph,
)

FIXTURES = Path(__file__).parent / "fixtures"


class PoseGraphTest(unittest.TestCase):
    def test_loads_the_given_image(self):
        graph = pose_graph("staged.png")
        self.assertEqual(graph["1"]["class_type"], "LoadImage")
        self.assertEqual(graph["1"]["inputs"]["image"], "staged.png")

    def test_dwpreprocessor_uses_the_none_bbox_detector(self):
        graph = pose_graph("staged.png")
        node = graph["2"]
        self.assertEqual(node["class_type"], "DWPreprocessor")
        self.assertEqual(node["inputs"]["image"], ["1", 0])
        self.assertEqual(node["inputs"]["bbox_detector"], "None")
        self.assertEqual(node["inputs"]["detect_hand"], "enable")
        self.assertEqual(node["inputs"]["detect_body"], "enable")
        self.assertEqual(node["inputs"]["detect_face"], "disable")

    def test_save_image_prefix_uses_the_given_prefix(self):
        graph = pose_graph("staged.png", prefix="rep-abc")
        self.assertEqual(graph["3"]["class_type"], "SaveImage")
        self.assertEqual(graph["3"]["inputs"]["images"], ["2", 0])
        self.assertEqual(graph["3"]["inputs"]["filename_prefix"], "rep-abc-pose")


class PoseFromOutputsTest(unittest.TestCase):
    def test_parses_the_first_frame_of_the_openpose_json_text_output(self):
        frame = json.loads((FIXTURES / "repair-kps-raw.json").read_text())
        outputs = {"2": {"openpose_json": [json.dumps([frame])]}}
        parsed = pose_from_outputs(outputs)
        self.assertEqual(parsed, frame)
        self.assertEqual(parsed["canvas_width"], 832)
        self.assertEqual(len(parsed["people"]), 1)


if __name__ == "__main__":
    unittest.main()
