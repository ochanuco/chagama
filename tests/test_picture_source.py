"""What the picture under a saturation stands for as a delivery."""

from __future__ import annotations

import unittest

from chagama.application.picture_source import (
    CUT_REDRAW_DELIVERY,
    delivery_parameters,
    has_backdrop_layer,
)


def context_of(kind, parameters):
    return {"request": {"kind": kind, "parameters": parameters}}


class Records:
    def __init__(self, contexts):
        self.contexts = contexts

    def request(self, method, path):
        return self.contexts[path.split("/")[4]]


class SaturationSourceTest(unittest.TestCase):
    def test_a_saturation_of_a_drawing_stands_on_a_transparent_cut_with_no_outlines(self):
        for drawing in (context_of("generate", {"kind": "generate"}),
                        context_of("redraw", {"kind": "redraw", "method": "hires"}),
                        context_of("redraw", {"kind": "redraw", "cut": True})):
            records = Records({"drw": drawing})
            saturated = context_of("saturation", {"kind": "saturation",
                                                  "base_generation": "drw", "strength": 1})
            with self.subTest(drawing=drawing):
                self.assertEqual(delivery_parameters(records, saturated), CUT_REDRAW_DELIVERY)
                self.assertFalse(has_backdrop_layer(records, saturated))

    def test_a_saturation_of_a_delivery_stands_on_that_delivery(self):
        delivery = {"kind": "deliver", "outlines": [], "transparent": False}
        records = Records({"dlv": context_of("deliver", delivery)})
        saturated = context_of("saturation", {"kind": "saturation",
                                              "base_generation": "dlv", "strength": 1})
        self.assertEqual(delivery_parameters(records, saturated), delivery)
        self.assertTrue(has_backdrop_layer(records, saturated))


if __name__ == "__main__":
    unittest.main()
