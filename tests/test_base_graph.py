"""Tests for reading a stored base graph, including repair outputs' stitched shape.

`repair-graph-raw.json` is a plain generation (one SaveImage).
`repair-graph-redraw.json` is a redraw output graph: a second sampling pass on
a bigger canvas whose VAEDecode feeds the one SaveImage. Both were captured
from a real worker submission.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from chagama.infrastructure.comfyui.base_graph import base_roles, source_prompts

FIXTURES = Path(__file__).parent / "fixtures"
RAW = json.loads((FIXTURES / "repair-graph-raw.json").read_text())
REDRAW = json.loads((FIXTURES / "repair-graph-redraw.json").read_text())


class SourcePromptsTest(unittest.TestCase):
    def test_raw_graph_reads_the_only_prompt_pair(self):
        positive, negative = source_prompts(RAW)
        self.assertEqual(positive, RAW["6"]["inputs"]["text"])
        self.assertEqual(negative, RAW["7"]["inputs"]["text"])

    def test_redraw_graph_reads_the_redraw_prompt_pair_not_pass1(self):
        # The fixture's pass-1 and redraw prompts happen to carry the same
        # text; what matters is which node source_prompts reads them from.
        positive, negative = source_prompts(REDRAW)
        self.assertEqual(positive, REDRAW["15"]["inputs"]["text"])
        self.assertEqual(negative, REDRAW["16"]["inputs"]["text"])


class BaseRolesTest(unittest.TestCase):
    def test_a_plain_save_is_not_stitched(self):
        self.assertFalse(base_roles(RAW).stitched)

    def test_a_save_behind_an_inpaint_stitch_is_stitched(self):
        graph = copy.deepcopy(RAW)
        save_id = next(key for key, node in graph.items()
                       if node["class_type"] == "SaveImage")
        decode_ref = graph[save_id]["inputs"]["images"]
        graph["900"] = {"class_type": "InpaintStitchImproved",
                        "inputs": {"inpainted_image": decode_ref}}
        graph[save_id]["inputs"]["images"] = ["900", 0]
        roles = base_roles(graph)
        self.assertEqual(roles.save_id, save_id)
        self.assertTrue(roles.stitched)
