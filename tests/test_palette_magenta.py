from __future__ import annotations

import unittest

from chagama.infrastructure.imaging.palette import palette_window, repin
from test_palette_cyan import center_hsv, swatch

PURPLE = palette_window("purple")["hue_target"]


class MagentaPullTest(unittest.TestCase):
    def test_a_saturated_light_pink_stays_pink_without_magenta(self):
        rgb, _ = repin(swatch((238, 150, 205)))
        self.assertGreater(center_hsv(rgb)[..., 0].mean(), 225)

    def test_magenta_eases_a_saturated_light_pink_to_purple_and_compresses_it(self):
        rgb, _ = repin(swatch((238, 150, 205)), magenta=1.0)
        center = center_hsv(rgb)
        self.assertAlmostEqual(center[..., 0].mean(), PURPLE, delta=3)
        self.assertLess(center[..., 1].mean(), 80)

    def test_magenta_leaves_skin_pale_pink_and_the_dark_mouth_red_alone(self):
        for hsv in ((20, 60, 230), (238, 60, 230), (252, 170, 110)):
            with self.subTest(hsv=hsv):
                plain, _ = repin(swatch(hsv))
                pulled, _ = repin(swatch(hsv), magenta=1.0)
                self.assertTrue((abs(center_hsv(plain) - center_hsv(pulled)) <= 1).all())


if __name__ == "__main__":
    unittest.main()
