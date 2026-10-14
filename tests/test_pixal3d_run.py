from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

import pixal3d_run


def _sheet(spans: list[tuple[int, int]], width: int = 600, height: int = 200) -> Image.Image:
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    for x0, x1 in spans:
        pixels[20:180, x0:x1] = (90, 60, 120)
    return Image.fromarray(pixels)


class SplitFiguresTest(unittest.TestCase):
    def test_cuts_at_empty_columns(self):
        figures = pixal3d_run.split_figures(_sheet([(10, 100), (130, 250), (400, 520)]))
        self.assertEqual(len(figures), 3)
        for (x0, x1), (a, b) in zip(figures, [(10, 100), (130, 250), (400, 520)]):
            self.assertLessEqual(x0, a)
            self.assertGreaterEqual(x1, b)
        for left, right in zip(figures, figures[1:]):
            self.assertLessEqual(left[1], right[0])

    def test_a_narrow_gap_still_splits(self):
        figures = pixal3d_run.split_figures(_sheet([(10, 200), (203, 400)]))
        self.assertEqual(len(figures), 2)

    def test_a_stray_strand_joins_its_figure(self):
        figures = pixal3d_run.split_figures(_sheet([(10, 200), (205, 210), (300, 450)]))
        self.assertEqual(len(figures), 2)
        self.assertGreaterEqual(figures[0][1], 210)


class ParseViewsTest(unittest.TestCase):
    def test_default(self):
        self.assertEqual(pixal3d_run.parse_views(pixal3d_run.DEFAULT_VIEWS),
                         {"front": 0, "left": 2, "back": 4, "right": 5})

    def test_rig_order_whatever_the_input_order(self):
        self.assertEqual(list(pixal3d_run.parse_views("back=3,front=0")), ["front", "back"])

    def test_front_required_and_names_checked(self):
        for text in ("left=1,back=2", "front=0,top=1", "front=x"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                pixal3d_run.parse_views(text)


class BuildGraphTest(unittest.TestCase):
    def setUp(self):
        self.files = {name: f"{name}.png" for name in pixal3d_run.VIEW_ORDER}
        self.graph = pixal3d_run.build_graph(self.files, prefix="pixal3d/test")

    def test_every_link_points_at_a_node_output(self):
        for node_id, node in self.graph.items():
            for name, value in node["inputs"].items():
                if isinstance(value, list):
                    with self.subTest(node=node_id, input=name):
                        self.assertIn(value[0], self.graph)
                        self.assertIsInstance(value[1], int)

    def test_every_view_reaches_the_conditioning(self):
        cond = self.graph["conditioning"]["inputs"]
        for name in pixal3d_run.VIEW_ORDER:
            self.assertEqual(cond[name], [f"crop_{name}", 0])
            self.assertEqual(self.graph[f"load_{name}"]["inputs"]["image"], f"{name}.png")

    def test_missing_views_are_left_unconnected(self):
        graph = pixal3d_run.build_graph({"front": "f.png", "back": "b.png"}, prefix="p")
        cond = graph["conditioning"]["inputs"]
        self.assertNotIn("left", cond)
        self.assertNotIn("load_left", graph)

    def test_ends_in_a_glb_save(self):
        saves = [n for n in self.graph.values() if n["class_type"] == "SaveGLB"]
        self.assertEqual(len(saves), 1)
        self.assertEqual(saves[0]["inputs"]["filename_prefix"], "pixal3d/test/model")


class SavedFilesTest(unittest.TestCase):
    def test_collects_files_under_any_ui_key(self):
        outputs = {
            "save_glb": {"3d": [{"filename": "model_00001_.glb", "subfolder": "pixal3d/x"}]},
            "save_front": {"images": [{"filename": "front_00001_.png", "type": "output"}]},
            "other": {"text": ["not a file"]},
        }
        names = sorted(f["filename"] for f in pixal3d_run.saved_files(outputs))
        self.assertEqual(names, ["front_00001_.png", "model_00001_.glb"])


if __name__ == "__main__":
    unittest.main()
