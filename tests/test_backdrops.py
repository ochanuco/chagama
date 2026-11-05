from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from chagama.domain.yukari import delivery_style
from chagama.infrastructure.imaging.backdrops import (
    PATTERNS,
    is_backdrop,
    render,
    stripes,
    thumbnail,
)
from chagama.infrastructure.imaging.delivery import parse_color

PALETTE = delivery_style.BACKDROP_PALETTE

PATTERN_NAMES = (
    "stripes", "waveform", "ears", "phases", "hatching", "torn",
    "stickers", "ornament", "dots", "gingham", "moons", "halftone",
    "sunburst", "chevron", "checker",
)


class StripesTest(unittest.TestCase):
    def test_shape_and_range(self):
        img = stripes(64, 48, PALETTE)
        self.assertEqual(img.shape, (64, 48, 3))
        self.assertTrue((img >= 0).all())
        self.assertTrue((img <= 255).all())

    def test_deterministic(self):
        np.testing.assert_array_equal(stripes(40, 40, PALETTE), stripes(40, 40, PALETTE))

    def test_not_flat(self):
        img = stripes(64, 64, PALETTE)
        self.assertGreater(float(img.std()), 0.0)


class RenderTest(unittest.TestCase):
    def test_none_matches_the_flat_default_backdrop(self):
        img = render(None, 10, 12, PALETTE)
        expected = np.array(parse_color(delivery_style.BACKDROP), dtype=float)
        self.assertTrue((img == expected).all())

    def test_a_hex_colour_is_flat(self):
        img = render("#ff0000", 10, 12, PALETTE)
        self.assertTrue((img == np.array([255.0, 0.0, 0.0])).all())

    def test_a_pattern_key_dispatches_to_its_function(self):
        np.testing.assert_array_equal(render("stripes", 20, 24, PALETTE),
                                      stripes(20, 24, PALETTE))

    def test_an_unknown_name_raises(self):
        with self.assertRaises(SystemExit):
            render("plaid", 10, 10, PALETTE)

    def test_shape_is_always_height_width_three(self):
        for backdrop in (None, "#112233", "stripes"):
            self.assertEqual(render(backdrop, 5, 7, PALETTE).shape, (5, 7, 3))


class IsBackdropTest(unittest.TestCase):
    def test_a_pattern_key_is_valid(self):
        self.assertTrue(is_backdrop("stripes"))

    def test_a_hex_colour_is_valid(self):
        self.assertTrue(is_backdrop("#c7e5e9"))

    def test_an_unknown_name_is_invalid(self):
        self.assertFalse(is_backdrop("plaid"))

    def test_patterns_registry_holds_every_name_in_display_order(self):
        self.assertEqual(tuple(PATTERNS), PATTERN_NAMES)


class EveryPatternTest(unittest.TestCase):
    def test_shape_and_range(self):
        for name in PATTERN_NAMES:
            with self.subTest(name=name):
                img = PATTERNS[name](64, 48, PALETTE)
                self.assertEqual(img.shape, (64, 48, 3))
                self.assertTrue((img >= 0).all())
                self.assertTrue((img <= 255).all())

    def test_deterministic(self):
        for name in PATTERN_NAMES:
            with self.subTest(name=name):
                np.testing.assert_array_equal(
                    PATTERNS[name](40, 40, PALETTE), PATTERNS[name](40, 40, PALETTE))

    def test_not_flat(self):
        for name in PATTERN_NAMES:
            with self.subTest(name=name):
                img = PATTERNS[name](64, 64, PALETTE)
                self.assertGreater(float(img.std()), 0.0)


class PaletteTest(unittest.TestCase):
    def test_every_pattern_paints_in_the_palette_it_is_given(self):
        other = PALETTE._replace(base="#f7d3b5", deep_stroke="#c27138")
        for name in PATTERN_NAMES:
            with self.subTest(name=name):
                self.assertFalse(np.array_equal(PATTERNS[name](48, 32, PALETTE),
                                                PATTERNS[name](48, 32, other)))

    def test_the_accent_reaches_the_motifs_that_use_it(self):
        other = PALETTE._replace(accent="#00ff00")
        for name in ("ears", "stickers", "ornament"):
            with self.subTest(name=name):
                self.assertFalse(np.array_equal(PATTERNS[name](200, 200, PALETTE),
                                                PATTERNS[name](200, 200, other)))

    def test_the_base_is_the_ground_of_a_flat_ground_pattern(self):
        other = PALETTE._replace(base="#f7d3b5")
        corner = PATTERNS["dots"](64, 64, other)[0, 0]
        np.testing.assert_array_equal(corner, parse_color("#f7d3b5"))

    def test_a_flat_colour_ignores_the_palette(self):
        other = PALETTE._replace(base="#000000")
        np.testing.assert_array_equal(render("#112233", 4, 4, PALETTE),
                                      render("#112233", 4, 4, other))


class BackdropLabelsTest(unittest.TestCase):
    def test_every_pattern_has_a_label(self):
        self.assertEqual(set(delivery_style.BACKDROP_LABELS), set(PATTERNS))

    def test_labels_are_non_empty_strings(self):
        for name, label in delivery_style.BACKDROP_LABELS.items():
            with self.subTest(name=name):
                self.assertIsInstance(label, str)
                self.assertTrue(label)


class ThumbnailTest(unittest.TestCase):
    def test_every_pattern_has_a_120x192_png_thumbnail(self):
        for name in PATTERN_NAMES:
            with self.subTest(name=name):
                data = thumbnail(name, PALETTE)
                self.assertIsInstance(data, bytes)
                image = Image.open(io.BytesIO(data))
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (120, 192))

    def test_deterministic(self):
        self.assertEqual(thumbnail("stripes", PALETTE), thumbnail("stripes", PALETTE))


if __name__ == "__main__":
    unittest.main()
