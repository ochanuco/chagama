"""cut_graph / deliver_graph: the carried-alpha wiring."""

from __future__ import annotations

import unittest

from chagama.domain.yukari import delivery_style
from chagama.infrastructure.comfyui.deliver_graph import cut_graph, deliver_graph

from test_deliver_application import GRAPH_KWARGS

MATTE = delivery_style.MATTE_MODEL


def classes(graph: dict) -> set:
    return {node["class_type"] for node in graph.values()}


def saved_prefixes(graph: dict) -> set:
    return {node["inputs"]["filename_prefix"] for node in graph.values()
            if node["class_type"] == "SaveImage"}


class CutCarryGraphTest(unittest.TestCase):
    def test_a_cut_without_a_carried_alpha_is_matted(self):
        graph = cut_graph("src.png", MATTE, "p", (64, 64))
        self.assertIn("YukariMatting", classes(graph))

    def test_a_cut_with_a_carried_alpha_takes_it_as_the_alpha(self):
        graph = cut_graph("src.png", MATTE, "p", (64, 64),
                          carried_image="carried.png")
        self.assertNotIn("YukariMatting", classes(graph))
        self.assertNotIn("BiRefNetRMBG", classes(graph))
        loads = {node["inputs"]["image"] for node in graph.values()
                 if node["class_type"] == "LoadImage"}
        self.assertEqual(loads, {"src.png", "carried.png"})
        self.assertEqual(saved_prefixes(graph),
                         {"p-alpha", "p-depth", "p-foreground"})

    def test_a_fresh_deliver_with_a_carried_alpha_takes_it_too(self):
        graph = deliver_graph("src.png", MATTE, "dlv-g",
                              carried_image="carried.png", **GRAPH_KWARGS)
        self.assertNotIn("YukariMatting", classes(graph))
        self.assertIn("dlv-g-alpha", saved_prefixes(graph))

    def test_a_loaded_alpha_is_reused_without_being_saved_again(self):
        graph = deliver_graph("src.png", MATTE, "dlv-g",
                              alpha_image="alpha.png", **GRAPH_KWARGS)
        self.assertNotIn("YukariMatting", classes(graph))
        self.assertNotIn("dlv-g-alpha", saved_prefixes(graph))


if __name__ == "__main__":
    unittest.main()
