"""cut_graph / deliver_graph: the native-alpha guide wiring."""

from __future__ import annotations

import unittest

from chagama.domain.yukari import delivery_style
from chagama.infrastructure.comfyui.deliver_graph import cut_graph, deliver_graph

from test_deliver_application import GRAPH_KWARGS

MATTE = delivery_style.MATTE_MODEL


def matting(graph: dict) -> dict:
    return next(node for node in graph.values()
                if node["class_type"] == "YukariMatting")


class CutGuideGraphTest(unittest.TestCase):
    def test_a_cut_without_a_guide_has_no_guide_input(self):
        graph = cut_graph("src.png", MATTE, "p", (64, 64))
        self.assertEqual(set(matting(graph)["inputs"]), {"image", "matte"})

    def test_a_cut_with_a_guide_wires_the_upload_into_the_matting(self):
        graph = cut_graph("src.png", MATTE, "p", (64, 64),
                          guide_image="guide.png")
        guide_ref = matting(graph)["inputs"]["guide"]
        to_mask = graph[guide_ref[0]]
        self.assertEqual(to_mask["class_type"], "ImageToMask")
        self.assertEqual(to_mask["inputs"]["channel"], "red")
        load = graph[to_mask["inputs"]["image"][0]]
        self.assertEqual(load["class_type"], "LoadImage")
        self.assertEqual(load["inputs"]["image"], "guide.png")

    def test_a_fresh_deliver_with_a_guide_wires_it_too(self):
        graph = deliver_graph("src.png", MATTE, "dlv-g",
                              guide_image="guide.png", **GRAPH_KWARGS)
        self.assertIn("guide", matting(graph)["inputs"])

    def test_a_loaded_alpha_ignores_the_guide(self):
        graph = deliver_graph("src.png", MATTE, "dlv-g",
                              alpha_image="alpha.png", guide_image="guide.png",
                              **GRAPH_KWARGS)
        classes = {node["class_type"] for node in graph.values()}
        self.assertNotIn("YukariMatting", classes)


if __name__ == "__main__":
    unittest.main()
