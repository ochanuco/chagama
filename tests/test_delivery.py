from __future__ import annotations

import io
import json
import re
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from scipy import ndimage

from chagama.domain.yukari import delivery_style
from chagama.infrastructure.imaging.palette import repin_skin_png
from chagama.infrastructure.imaging.delivery import (
    background_mask,
    band_alphas,
    clean_background,
    cut_figure,
    deliver_pngs,
    despill,
    down2,
    drawn_outline,
    enclosed_cut,
    figure_rim,
    fill_unkeyed_holes,
    frame_window,
    graph_from_png,
    graph_from_png_or_none,
    keep_scene,
    key_holes,
    keyed_coverage,
    parse_color,
    refine_matte,
    shadow_cut,
    sticker,
    stroke_alpha,
    transparent,
    unpremultiply,
)


PURPLE = "#885b80"
PURPLE_WIDTH_PCT = 1.04
PURPLE_ONLY = [{"color": PURPLE, "width": PURPLE_WIDTH_PCT}]
WHITE_PURPLE = [{"color": "#ffffff", "width": 0.4}, *PURPLE_ONLY]


def png(pixels: np.ndarray, prompt: str | None = None) -> bytes:
    output = io.BytesIO()
    metadata = PngInfo()
    if prompt is not None:
        metadata.add_text("prompt", prompt)
    Image.fromarray(pixels.astype(np.uint8)).save(
        output, "PNG", pnginfo=metadata)
    return output.getvalue()


def matte(shape, box):
    """A hard matte with one rectangular figure, as the worker would send it."""
    mask = np.zeros(shape, dtype=np.uint8)
    top, bottom, left, right = box
    mask[top:bottom, left:right] = 255
    return png(mask)


class DeliveryTest(unittest.TestCase):
    # These cases pin the purple band straight against the figure; the
    # recipe default's white band is covered by OutlinesTest.
    def setUp(self):
        patcher = mock.patch.object(delivery_style, "OUTLINES", PURPLE_ONLY)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_graph_metadata_validation(self):
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaisesRegex(SystemExit, "no ComfyUI prompt"):
            graph_from_png(png(pixels))
        with self.assertRaisesRegex(SystemExit, "invalid ComfyUI prompt"):
            graph_from_png(png(pixels, "not-json"))
        self.assertEqual(graph_from_png(png(pixels, json.dumps({"3": {}}))),
                         {"3": {}})

    def test_lenient_graph_reader_returns_none_without_metadata(self):
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertIsNone(graph_from_png_or_none(png(pixels)))
        self.assertEqual(
            graph_from_png_or_none(png(pixels, json.dumps({"3": {}}))), {"3": {}})
        with self.assertRaisesRegex(SystemExit, "invalid ComfyUI prompt"):
            graph_from_png_or_none(png(pixels, "not-json"))
        with self.assertRaisesRegex(SystemExit, "invalid ComfyUI prompt"):
            graph_from_png_or_none(png(pixels, "[]"))

    def test_parse_color_rejects_non_hexadecimal_input(self):
        self.assertEqual(parse_color("#c7e5e9"), (199, 229, 233))
        with self.assertRaisesRegex(SystemExit, "6-digit hex"):
            parse_color("gggggg")

    def test_background_mask_claims_every_region_reaching_the_frame(self):
        # A figure that crosses the frame cuts the backdrop into pieces. Each
        # piece that still reaches an edge is backdrop; keeping only the
        # corner's piece is what left half a head crop at the render's own
        # colour, with the stroke drawn along the seam between the two halves.
        pixels = np.zeros((12, 12, 3), dtype=int)
        pixels[:, 5:7] = 255
        mask = background_mask(pixels, 0)
        self.assertEqual(int(mask.sum()), 12 * 12 - 24)
        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[0, 11])
        self.assertFalse(mask[0, 5])

    def test_background_mask_leaves_regions_walled_off_from_the_frame(self):
        # Backdrop enclosed by the figure is `enclosed_mask`'s job, on its own
        # much tighter tolerance; the flood must not reach it.
        pixels = np.zeros((12, 12, 3), dtype=int)
        pixels[4:9, 4:9] = 255
        pixels[6, 6] = 0
        mask = background_mask(pixels, 0)
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[6, 6])

    def test_refine_matte_retraces_the_edge_by_colour_and_drops_grain(self):
        pixels = np.full((40, 40, 3), (58, 67, 81), dtype=int)
        pixels[10:30, 10:30] = (190, 170, 220)
        # A strand the matte lost, a strand the matte cut short, and a
        # backdrop speck the matte never had.
        pixels[20, 30:33] = (190, 170, 220)
        pixels[24, 8:10] = (190, 170, 220)
        pixels[2, 2] = (200, 200, 200)
        figure = np.zeros((40, 40), dtype=bool)
        figure[10:30, 10:30] = True
        figure[24, 9] = True
        refined = refine_matte(pixels, figure, 3, 20)
        self.assertTrue(refined[20, 30:33].all())
        self.assertTrue(refined[24, 8:10].all())
        self.assertFalse(refined[2, 2])
        self.assertTrue(refined[10:30, 10:30].all())
        self.assertFalse(refined[0:5, 10:40].any())

    def test_refine_matte_reads_the_backdrop_locally_under_a_gradient(self):
        # The backdrop darkens by 40 toward the figure: a single corner
        # reference would claim the shaded ring as figure.
        ramp = np.linspace(0, 40, 60)
        pixels = np.zeros((60, 60, 3), dtype=int) + (58, 67, 81)
        pixels = pixels + ramp[None, :, None].astype(int)
        pixels[20:40, 20:40] = (190, 170, 220)
        figure = np.zeros((60, 60), dtype=bool)
        figure[20:40, 20:40] = True
        refined = refine_matte(pixels, figure, 3, 20)
        self.assertTrue((refined == figure).all())

    def test_refine_matte_is_identity_below_one_pixel_of_band(self):
        pixels = np.zeros((8, 8, 3), dtype=int)
        figure = np.zeros((8, 8), dtype=bool)
        figure[2:6, 2:6] = True
        self.assertIs(refine_matte(pixels, figure, 0, 20), figure)

    def test_keep_scene_returns_the_redraw_untouched(self):
        pixels = np.full((32, 32, 3), (210, 230, 235), dtype=np.uint8)
        data = png(pixels)
        kept, tag = keep_scene(data, matte(pixels.shape[:2], (8, 24, 10, 22)))
        self.assertEqual(kept, data)
        self.assertEqual(tag, "scene")

    def test_clean_background_preserves_size_and_clean_width_tag(self):
        pixels = np.full((32, 32, 3), (210, 230, 235), dtype=np.uint8)
        pixels[8:24, 10:22] = (40, 40, 40)
        cleaned, tag = clean_background(png(pixels), matte(pixels.shape[:2],
                                                           (8, 24, 10, 22)))
        self.assertEqual(Image.open(io.BytesIO(cleaned)).size, (32, 32))
        self.assertRegex(tag, r"^clean-o\d+-cut0\.5$")

    def test_clean_background_keeps_the_retrace_inside_the_soft_matte(self):
        pixels = np.full((1024, 1024, 3), (218, 214, 218), dtype=np.uint8)
        pixels[256:768, 256:640] = (40, 40, 40)
        pixels[256:768, 640:656] = (180, 178, 180)  # a cast shadow on the floor
        soft = np.zeros((1024, 1024), dtype=np.uint8)
        soft[256:768, 256:640] = 255
        cleaned, _ = clean_background(png(pixels), png(soft))
        arr = np.array(Image.open(io.BytesIO(cleaned)))
        # The shadow is band, not figure: the purple stroke paints over it
        # right up to the soft matte's edge.
        np.testing.assert_array_equal(arr[512, 640], parse_color(PURPLE))
        np.testing.assert_array_equal(arr[512, 638], (40, 40, 40))

    def test_shadow_cut_drops_the_cast_shadow_the_matte_kept_and_no_more(self):
        pixels = np.full((1024, 1024, 3), (218, 214, 218), dtype=np.uint8)
        pixels[256:640, 256:640] = (40, 40, 40)
        pixels[400:500, 300:340] = (170, 170, 170)   # a grey patch inside
        pixels[640:680, 256:640] = (180, 178, 180)   # a cast shadow under it
        soft = np.zeros((1024, 1024), dtype=np.uint8)
        soft[256:640, 256:640] = 255
        soft[640:680, 256:640] = 180                  # the model kept the shadow
        figure = soft > 127
        cut = shadow_cut(pixels.astype(float), figure, soft, 6)
        self.assertFalse(cut[640:680, 256:640].any())
        self.assertTrue(cut[400:500, 300:340].all())
        self.assertTrue(cut[256:640, 256:640].all())

    def test_enclosed_cut_drops_the_key_pocket_the_matte_was_certain_of(self):
        pixels = np.full((256, 256, 3), (196, 220, 151), dtype=np.uint8)
        pixels[64:192, 64:192] = (215, 200, 240)
        pixels[100:120, 100:120] = (195, 218, 153)   # backdrop a loop of hair closes
        pixels[150:153, 150:153] = (196, 220, 151)   # a speck under the minimum area
        figure = np.zeros((256, 256), dtype=bool)
        figure[64:192, 64:192] = True
        cut = enclosed_cut(pixels.astype(float), figure, 20)
        self.assertFalse(cut[100:120, 100:120].any())
        self.assertTrue(cut[150:153, 150:153].all())
        self.assertEqual(int((figure & ~cut).sum()), 400)

    def test_enclosed_cut_drops_a_shaded_key_pocket_only_where_it_is_enclosed(self):
        pixels = np.full((256, 256, 3), (138, 196, 138), dtype=np.uint8)
        pixels[64:192, 64:192] = (170, 140, 210)
        pixels[100:130, 100:106] = (122, 168, 120)   # the key deep in a loop of hair
        pixels[64:192, 64:67] = (120, 170, 120)      # the key's shade on the edge
        pixels[150:160, 150:160] = (60, 85, 60)      # darker than the key's shade
        figure = np.zeros((256, 256), dtype=bool)
        figure[64:192, 64:192] = True
        cut = enclosed_cut(pixels.astype(float), figure, 20)
        self.assertFalse(cut[100:130, 100:106].any())
        self.assertTrue(cut[64:192, 64:67].all())
        self.assertTrue(cut[150:160, 150:160].all())
        self.assertEqual(int((figure & ~cut).sum()), 180)

    def test_key_holes_are_the_key_coloured_pixels_of_the_figures_holes(self):
        pixels = np.full((128, 128, 3), (138, 196, 138), dtype=np.uint8)
        pixels[32:96, 32:96] = (170, 140, 210)
        pixels[50:60, 50:54] = (122, 168, 120)
        pixels[70:80, 70:74] = (60, 40, 70)          # a hole the matte left dark
        figure = np.zeros((128, 128), dtype=bool)
        figure[32:96, 32:96] = True
        figure[50:60, 50:54] = False
        figure[70:80, 70:74] = False
        holes = key_holes(pixels.astype(float), figure, 20)
        self.assertTrue(holes[50:60, 50:54].all())
        self.assertEqual(int(holes.sum()), 40)
        pixels[:] = (210, 230, 235)
        self.assertFalse(key_holes(pixels.astype(float), figure, 20).any())

    def test_cut_figure_fills_a_matte_hole_that_holds_no_key(self):
        pixels = np.full((256, 256, 3), (89, 166, 119), dtype=np.uint8)
        pixels[64:192, 64:192] = (60, 40, 70)
        pixels[100:140, 100:140] = (42, 45, 71)      # cushion between the legs
        pixels[150:170, 150:170] = (89, 166, 119)    # backdrop a loop closes
        soft = np.zeros((256, 256), dtype=np.uint8)
        soft[64:192, 64:192] = 255
        soft[100:140, 100:140] = 0
        soft[150:170, 150:170] = 0
        figure = cut_figure(pixels.astype(float), soft)
        self.assertTrue(figure[100:140, 100:140].all())
        self.assertFalse(figure[152:168, 152:168].any())
        self.assertFalse(figure[:32].any())

    def test_fill_unkeyed_holes_is_a_no_op_unless_the_backdrop_is_a_green_key(self):
        for backdrop in ((218, 214, 218), (210, 230, 235)):
            with self.subTest(backdrop=backdrop):
                pixels = np.full((256, 256, 3), backdrop, dtype=np.uint8)
                pixels[64:192, 64:192] = (40, 40, 40)
                soft = np.zeros((256, 256), dtype=np.uint8)
                soft[64:192, 64:192] = 255
                soft[100:140, 100:140] = 0
                np.testing.assert_array_equal(
                    fill_unkeyed_holes(pixels.astype(float), soft), soft)

    def test_enclosed_cut_is_a_no_op_unless_the_backdrop_is_a_green_key(self):
        figure = np.zeros((256, 256), dtype=bool)
        figure[64:192, 64:192] = True
        for backdrop in ((218, 214, 218), (210, 230, 235)):
            with self.subTest(backdrop=backdrop):
                pixels = np.full((256, 256, 3), backdrop, dtype=np.uint8)
                pixels[64:192, 64:192] = (40, 40, 40)
                pixels[100:120, 100:120] = backdrop   # the figure's own pale passage
                np.testing.assert_array_equal(
                    enclosed_cut(pixels.astype(float), figure, 20), figure)

    def test_enclosed_cut_takes_the_key_from_a_pocket_when_the_corners_are_white(self):
        pixels = np.full((256, 256, 3), (255, 255, 255), dtype=np.uint8)
        pixels[64:192, 64:192] = (196, 220, 151)     # green closed off by a frame line
        pixels[60:196, 60:64] = (20, 20, 20)
        pixels[60:196, 192:196] = (20, 20, 20)
        pixels[60:64, 60:196] = (20, 20, 20)
        pixels[192:196, 60:196] = (20, 20, 20)
        pixels[100:160, 100:160] = (215, 200, 240)   # the figure inside it
        figure = np.zeros((256, 256), dtype=bool)
        figure[60:196, 60:196] = True
        cut = enclosed_cut(pixels.astype(float), figure, 20)
        self.assertFalse(cut[70:90, 70:90].any())
        self.assertTrue(cut[100:160, 100:160].all())
        self.assertTrue(cut[60:196, 60:64].all())

    def test_clean_background_paints_the_backdrop_only_inside_the_frame(self):
        pixels = np.full((1024, 1024, 3), (255, 255, 255), dtype=np.uint8)
        pixels[256:768, 256:768] = (196, 220, 151)
        pixels[250:774, 250:256] = (20, 20, 20)
        pixels[250:774, 768:774] = (20, 20, 20)
        pixels[250:256, 250:774] = (20, 20, 20)
        pixels[768:774, 250:774] = (20, 20, 20)
        pixels[400:624, 400:624] = (215, 200, 240)
        soft = np.zeros((1024, 1024), dtype=np.uint8)
        soft[400:624, 400:624] = 255
        cleaned, _ = clean_background(png(pixels), png(soft), backdrop="#102030")
        arr = np.array(Image.open(io.BytesIO(cleaned))).astype(int)
        purple = np.array(parse_color(PURPLE))
        self.assertTrue((np.abs(arr[300:340, 300:340] - (16, 32, 48)).max(axis=2) <= 2).all())
        self.assertTrue((arr[:64, :64] == 255).all())
        self.assertTrue((arr[-64:, -64:] == 255).all())
        np.testing.assert_array_equal(arr[250:774, 250:256], pixels[250:774, 250:256])
        np.testing.assert_array_equal(arr[768:774, 250:774], pixels[768:774, 250:774])
        near = np.abs(arr[512, 340:400] - purple).max(axis=1) <= 40
        self.assertTrue(near.any())
        outside = np.ones((1024, 1024), dtype=bool)
        outside[256:768, 256:768] = False
        distance = np.abs(arr - purple).max(axis=2)
        self.assertFalse(((distance <= 40) & outside).any())

    def test_clean_background_keeps_a_grey_bezel_and_the_figure_over_it(self):
        pixels = np.full((1024, 1024, 3), (60, 60, 60), dtype=np.uint8)
        pixels[256:768, 256:768] = (196, 220, 151)
        pixels[600:900, 400:624] = (215, 200, 240)    # steps out over the bottom bezel
        soft = np.zeros((1024, 1024), dtype=np.uint8)
        soft[600:900, 400:624] = 255
        cleaned, _ = clean_background(png(pixels), png(soft), backdrop="#102030")
        arr = np.array(Image.open(io.BytesIO(cleaned))).astype(int)
        outside = np.ones((1024, 1024), dtype=bool)
        outside[256:768, 256:768] = False
        np.testing.assert_array_equal(arr[outside], pixels.astype(int)[outside])
        self.assertTrue((np.abs(arr[300:340, 300:340] - (16, 32, 48)).max(axis=2) <= 2).all())

    def test_clean_background_paints_green_outside_the_frame_with_the_backdrop(self):
        pixels = np.full((1024, 1024, 3), (20, 20, 20), dtype=np.uint8)
        pixels[:200, :200] = (196, 220, 151)          # green beyond the black band
        pixels[256:768, 256:768] = (196, 220, 151)
        pixels[400:624, 400:624] = (215, 200, 240)
        soft = np.zeros((1024, 1024), dtype=np.uint8)
        soft[400:624, 400:624] = 255
        cleaned, _ = clean_background(png(pixels), png(soft), backdrop="#102030")
        arr = np.array(Image.open(io.BytesIO(cleaned))).astype(int)
        purple = np.array(parse_color(PURPLE))
        self.assertTrue((np.abs(arr[:190, :190] - (16, 32, 48)).max(axis=2) <= 2).all())
        self.assertTrue((np.abs(arr[300:340, 300:340] - (16, 32, 48)).max(axis=2) <= 2).all())
        np.testing.assert_array_equal(arr[200:256, 300:700], pixels.astype(int)[200:256, 300:700])
        outside = np.ones((1024, 1024), dtype=bool)
        outside[256:768, 256:768] = False
        self.assertFalse(((np.abs(arr - purple).max(axis=2) <= 40) & outside).any())

    def test_clean_background_takes_a_window_that_runs_off_the_canvas(self):
        pixels = np.full((1024, 1024, 3), (20, 20, 20), dtype=np.uint8)
        pixels[:100, :100] = (196, 220, 151)          # outer green, far from the figure
        pixels[:768, 256:768] = (196, 220, 151)       # the window reaches the top edge
        pixels[400:624, 400:624] = (215, 200, 240)
        soft = np.zeros((1024, 1024), dtype=np.uint8)
        soft[400:624, 400:624] = 255
        cleaned, _ = clean_background(png(pixels), png(soft), backdrop="#102030")
        arr = np.array(Image.open(io.BytesIO(cleaned))).astype(int)
        self.assertTrue((np.abs(arr[:90, :90] - (16, 32, 48)).max(axis=2) <= 2).all())
        self.assertTrue((np.abs(arr[20:60, 300:340] - (16, 32, 48)).max(axis=2) <= 2).all())
        np.testing.assert_array_equal(arr[200:700, 100:200], pixels.astype(int)[200:700, 100:200])
        purple = np.array(parse_color(PURPLE))
        self.assertTrue((np.abs(arr[512, 340:400] - purple).max(axis=1) <= 40).any())
        outside = np.ones((1024, 1024), dtype=bool)
        outside[:768, 256:768] = False
        self.assertFalse(((np.abs(arr - purple).max(axis=2) <= 40) & outside).any())

    def test_clean_background_takes_both_sides_of_a_window_the_figure_splits(self):
        pixels = np.full((1024, 1024, 3), (255, 255, 255), dtype=np.uint8)
        pixels[256:768, 256:768] = (196, 220, 151)
        pixels[250:774, 250:256] = (20, 20, 20)
        pixels[250:774, 768:774] = (20, 20, 20)
        pixels[250:256, 250:774] = (20, 20, 20)
        pixels[768:774, 250:774] = (20, 20, 20)
        pixels[256:640, 480:544] = (215, 200, 240)    # reaches the top and splits the green
        soft = np.zeros((1024, 1024), dtype=np.uint8)
        soft[256:640, 480:544] = 255
        cleaned, _ = clean_background(png(pixels), png(soft), backdrop="#102030")
        arr = np.array(Image.open(io.BytesIO(cleaned))).astype(int)
        purple = np.array(parse_color(PURPLE))
        for cols in (slice(280, 330), slice(700, 750)):
            self.assertTrue((np.abs(arr[300:340, cols] - (16, 32, 48)).max(axis=2) <= 2).all())
        self.assertTrue((np.abs(arr[400, 400:480] - purple).max(axis=1) <= 40).any())
        self.assertTrue((np.abs(arr[400, 544:640] - purple).max(axis=1) <= 40).any())

    def test_a_full_green_screen_is_not_a_frame(self):
        pixels = np.full((1024, 1024, 3), (196, 220, 151), dtype=np.uint8)
        pixels[256:768, 256:768] = (215, 200, 240)
        self.assertIsNone(frame_window(pixels.astype(float)))
        soft = np.zeros((1024, 1024), dtype=np.uint8)
        soft[256:768, 256:768] = 255
        cleaned, _ = clean_background(png(pixels), png(soft), backdrop="#102030")
        arr = np.array(Image.open(io.BytesIO(cleaned))).astype(int)
        self.assertTrue((np.abs(arr[:64, :64] - (16, 32, 48)).max(axis=2) <= 2).all())
        purple = np.array(parse_color(PURPLE))
        self.assertTrue((np.abs(arr[512, 150:256] - purple).max(axis=1) <= 40).any())

    def test_enclosed_cut_ignores_a_few_green_pixels_on_a_white_backdrop(self):
        pixels = np.full((256, 256, 3), (255, 255, 255), dtype=np.uint8)
        pixels[64:192, 64:192] = (40, 40, 40)
        pixels[100:110, 100:110] = (196, 220, 151)   # under ENCLOSED_POCKET_MIN_AREA
        figure = np.zeros((256, 256), dtype=bool)
        figure[64:192, 64:192] = True
        np.testing.assert_array_equal(
            enclosed_cut(pixels.astype(float), figure, 20), figure)

    def test_clean_background_and_transparent_open_the_enclosed_key_pocket(self):
        pixels = np.full((1024, 1024, 3), (196, 220, 151), dtype=np.uint8)
        pixels[256:768, 256:768] = (215, 200, 240)
        pixels[480:544, 480:544] = (196, 220, 151)
        soft = np.zeros((1024, 1024), dtype=np.uint8)
        soft[256:768, 256:768] = 255
        cleaned, _ = clean_background(png(pixels), png(soft), backdrop="#102030")
        arr = np.array(Image.open(io.BytesIO(cleaned)))
        self.assertFalse((np.abs(arr[480:544, 480:544].astype(int)
                                 - (196, 220, 151)).max(axis=2) <= 20).any())
        cut, _ = transparent(png(pixels), png(soft))
        rgba = np.array(Image.open(io.BytesIO(cut)))
        self.assertFalse((np.abs(rgba[480:544, 480:544, :3].astype(int)
                                 - (196, 220, 151)).max(axis=2) <= 20).any())

    def outlined_figure(self, ring: bool):
        """A lavender square, its black line, then (with `ring`) a drawn
        white outline and a key-tinted halo, on a green key."""
        pixels = np.full((512, 512, 3), (150, 196, 164), dtype=np.uint8)
        soft = np.zeros((512, 512), dtype=np.uint8)
        if ring:
            pixels[147:365, 147:365] = (194, 215, 200)
            pixels[150:362, 150:362] = (255, 255, 255)
            soft[147:365, 147:365] = 255
        pixels[156:356, 156:356] = (40, 30, 50)
        pixels[158:354, 158:354] = (215, 200, 240)
        if not ring:
            soft[156:356, 156:356] = 255
        return pixels, soft

    def test_drawn_outline_takes_the_white_ring_and_halo_up_to_the_line(self):
        pixels, soft = self.outlined_figure(ring=True)
        figure = soft > 127
        outline = drawn_outline(pixels.astype(float), figure, 6)
        self.assertTrue(outline[147:156, 147:365].all())
        self.assertFalse(outline[156:356, 156:356].any())

    def test_drawn_outline_reaches_past_key_pixels_the_matte_overshot(self):
        pixels, soft = self.outlined_figure(ring=True)
        key = np.all(pixels == (150, 196, 164), axis=2)
        pixels[key] = (109, 213, 81)
        pixels[147:365, 147:365][np.all(pixels[147:365, 147:365] == (194, 215, 200), axis=2)] = (170, 223, 158)
        soft[145:367, 145:367] = 255
        figure = soft > 127
        outline = drawn_outline(pixels.astype(float), figure, 6)
        self.assertTrue(outline[145:156, 145:367].all())
        self.assertFalse(outline[156:356, 156:356].any())

    def test_drawn_outline_takes_the_specks_left_inside_it(self):
        pixels, soft = self.outlined_figure(ring=True)
        pixels[151:153, 200:202] = (90, 80, 110)
        outline = drawn_outline(pixels.astype(float), soft > 127, 6)
        self.assertTrue(outline[151:153, 200:202].all())
        self.assertFalse(outline[156:356, 156:356].any())

    def test_drawn_outline_reaches_past_a_darker_key_fringe(self):
        pixels, soft = self.outlined_figure(ring=True)
        pixels[147:365, 147:365][np.all(pixels[147:365, 147:365] == (194, 215, 200), axis=2)] = (128, 167, 140)
        soft[145:367, 145:367] = 255
        pixels[145:367, 145:367][np.all(pixels[145:367, 145:367] == (150, 196, 164), axis=2)] = (128, 167, 140)
        outline = drawn_outline(pixels.astype(float), soft > 127, 6)
        self.assertTrue(outline[145:156, 145:367].all())
        self.assertFalse(outline[156:356, 156:356].any())

    def test_drawn_outline_takes_a_key_pocket_the_figure_lines_close_off(self):
        pixels, soft = self.outlined_figure(ring=True)
        pixels[200:222, 200:222] = (40, 30, 50)
        pixels[202:220, 202:220] = (255, 255, 255)
        pixels[208:214, 208:214] = (150, 196, 164)
        pixels[207, 208:214] = (120, 160, 110)
        outline = drawn_outline(pixels.astype(float), soft > 127, 6)
        self.assertTrue(outline[202:220, 202:220].all())
        self.assertFalse(outline[200, 200:222].any())
        self.assertFalse(outline[240:350, 240:350].any())

    def test_drawn_outline_leaves_a_white_patch_without_key_inside(self):
        pixels, soft = self.outlined_figure(ring=True)
        pixels[200:230, 200:230] = (40, 30, 50)
        pixels[202:228, 202:228] = (255, 255, 255)
        outline = drawn_outline(pixels.astype(float), soft > 127, 6)
        self.assertFalse(outline[202:228, 202:228].any())

    def test_drawn_outline_ignores_where_the_figure_runs_off_the_canvas(self):
        pixels = np.full((512, 512, 3), (150, 196, 164), dtype=np.uint8)
        pixels[147:, :] = (194, 215, 200)
        pixels[150:, :] = (255, 255, 255)
        pixels[156:, :] = (40, 30, 50)
        pixels[158:, :] = (215, 200, 240)
        figure = np.zeros((512, 512), dtype=bool)
        figure[147:, :] = True
        outline = drawn_outline(pixels.astype(float), figure, 6)
        self.assertTrue(outline[147:156, 100:400].all())
        self.assertFalse(outline[160:, :].any())

    def test_drawn_outline_is_empty_without_a_ring_around_most_of_the_edge(self):
        pixels, soft = self.outlined_figure(ring=False)
        self.assertFalse(drawn_outline(pixels.astype(float), soft > 127, 6).any())
        pixels, soft = self.outlined_figure(ring=True)
        pixels[147:156, 147:365] = (40, 30, 50)
        pixels[147:365, 147:156] = (40, 30, 50)
        pixels[147:365, 356:365] = (40, 30, 50)
        self.assertFalse(drawn_outline(pixels.astype(float), soft > 127, 6).any())

    def test_clean_background_and_transparent_paint_the_drawn_outline_white(self):
        pixels, soft = self.outlined_figure(ring=True)
        cleaned, tag = clean_background(png(pixels), png(soft), backdrop="#102030")
        arr = np.array(Image.open(io.BytesIO(cleaned))).astype(int)
        self.assertTrue(tag.endswith("-outline"))
        self.assertTrue((arr[147:154, 200:300] == 255).all())
        self.assertTrue((arr[200:300, 200:300] == (215, 200, 240)).all())
        cut, tag = transparent(png(pixels), png(soft))
        rgba = np.array(Image.open(io.BytesIO(cut))).astype(int)
        self.assertTrue(tag.endswith("-outline"))
        self.assertTrue((rgba[147:154, 200:300] == 255).all())

    def test_clean_background_keeps_the_drawn_outline_white_inside_a_bezel(self):
        pixels, soft = self.outlined_figure(ring=True)
        bezel = np.ones((512, 512), dtype=bool)
        bezel[40:472, 40:472] = False
        pixels[bezel] = (60, 60, 60)
        soft[bezel] = 255
        cleaned, tag = clean_background(png(pixels), png(soft), backdrop="#102030")
        arr = np.array(Image.open(io.BytesIO(cleaned))).astype(int)
        self.assertIn("-outline", tag)
        self.assertTrue((arr[150:154, 200:300] == 255).all())

    def test_clean_background_without_a_drawn_outline_has_no_outline_tag(self):
        pixels, soft = self.outlined_figure(ring=False)
        _, tag = clean_background(png(pixels), png(soft))
        self.assertNotIn("-outline", tag)
        _, tag = transparent(png(pixels), png(soft))
        self.assertNotIn("-outline", tag)

    def test_clean_background_band_widths_derive_from_the_longest_side(self):
        pixels = np.full((30, 50, 3), (210, 230, 235), dtype=np.uint8)
        pixels[8:24, 15:35] = (20, 20, 20)
        _, tag = clean_background(png(pixels), matte(pixels.shape[:2],
                                                     (8, 24, 15, 35)))
        match = re.match(r"^clean-o(\d+)-cut0\.5$", tag)
        self.assertIsNotNone(match)
        purple_w = max(30, 50) * PURPLE_WIDTH_PCT / 100
        self.assertEqual(match.group(1), f"{purple_w:.0f}")
        _, tag = clean_background(png(pixels), matte(pixels.shape[:2], (8, 24, 15, 35)),
                                  outlines=WHITE_PURPLE)
        self.assertRegex(tag, r"^clean-o0\+1-cut0\.5$")
        _, tag = clean_background(png(pixels), matte(pixels.shape[:2], (8, 24, 15, 35)),
                                  outlines=[])
        self.assertRegex(tag, r"^clean-nooutline-cut0\.5$")

    @mock.patch.object(delivery_style, "BACKDROP", "#c7e5e9")
    def test_clean_background_composites_purple_directly_against_the_figure(self):
        # Walking outward from the figure's edge should cross the purple
        # stroke and then reach the flat backdrop, with no white band between
        # the figure and the stroke. Pinned at eps 0 (the smooth ramp): the
        # stroke is only a few px wide on this canvas, too thin for the
        # polygon cut to resolve a pure colour.
        height = width = 240
        pixels = np.full((height, width, 3), (233, 229, 199), dtype=np.uint8)
        pixels[80:160, 80:160] = (10, 10, 10)
        with mock.patch.object(delivery_style, "STROKE_CUT_EPS_PCT", 0):
            cleaned, _ = clean_background(png(pixels), matte(pixels.shape[:2],
                                                             (80, 160, 80, 160)))
        arr = np.array(Image.open(io.BytesIO(cleaned)).convert("RGB")).astype(int)

        backdrop = np.array(parse_color(delivery_style.BACKDROP))
        purple = np.array(parse_color(PURPLE))
        white = np.array([255, 255, 255])

        row = height // 2
        cols = np.arange(160, width)
        strip = arr[row, cols]

        def first_match(color, tolerance=20):
            distance = np.abs(strip - color).sum(axis=1)
            hits = np.where(distance <= tolerance)[0]
            self.assertTrue(hits.size, f"strip never reaches {color}")
            return cols[hits[0]]

        purple_at = first_match(purple)
        backdrop_at = first_match(backdrop)
        self.assertLessEqual(purple_at, 162)
        self.assertLess(purple_at, backdrop_at)
        self.assertGreater(np.abs(strip - white).sum(axis=1).min(), 20)

    def test_transparent_frames_the_cutout_with_the_sticker_bands(self):
        # Pinned at eps 0 (the smooth ramp), same reason as the clean
        # background layer-order test above.
        pixels = np.full((256, 256, 3), (210, 230, 235), dtype=np.uint8)
        pixels[64:192, 64:160] = (40, 40, 40)
        with mock.patch.object(delivery_style, "STROKE_CUT_EPS_PCT", 0):
            cut, tag = transparent(png(pixels), matte(pixels.shape[:2], (64, 192, 64, 160)))
        image = Image.open(io.BytesIO(cut))
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.size, (256, 256))
        arr = np.array(image)
        purple = np.array(parse_color(PURPLE))
        white = np.array([255, 255, 255])
        # Inside the figure: its own pixels, opaque.
        np.testing.assert_array_equal(arr[128, 100, :3], (40, 40, 40))
        self.assertEqual(arr[128, 100, 3], 255)
        # 256 * 1.04% = 2.7px of purple straight against the figure, then nothing.
        np.testing.assert_array_equal(arr[128, 161, :3], purple)
        self.assertEqual(arr[128, 161, 3], 255)
        self.assertEqual(arr[128, 164, 3], 0)
        self.assertEqual(arr[128, 172, 3], 0)
        self.assertEqual(arr[0, 0, 3], 0)
        opaque = arr[..., 3] > 0
        self.assertFalse((arr[..., :3] == white).all(axis=2)[opaque].any())
        self.assertEqual(tag, "transparent-o3")

    def test_transparent_keeps_the_retrace_inside_the_soft_matte(self):
        pixels = np.full((256, 256, 3), (210, 230, 235), dtype=np.uint8)
        pixels[64:192, 64:160] = (40, 40, 40)
        pixels[64:192, 160:161] = (150, 150, 150)   # backdrop shading at the edge
        pixels[120:136, 158:160] = (215, 228, 232)  # a light passage at the edge
        soft = np.zeros((256, 256), dtype=np.uint8)
        soft[64:192, 64:160] = 255
        cut, _ = transparent(png(pixels), png(soft))
        arr = np.array(Image.open(io.BytesIO(cut)))
        # The shading column is band, not figure: the purple stroke paints
        # over it, and the figure's own edge stays where the soft matte put it.
        purple = parse_color(PURPLE)
        np.testing.assert_array_equal(arr[128, 161, :3], purple)
        np.testing.assert_array_equal(arr[128, 158, :3], (215, 228, 232))
        self.assertEqual(arr[128, 158, 3], 255)

    def test_stroke_alpha_ramps_over_one_pixel_at_the_outer_edge(self):
        mask = np.ones((1, 12), dtype=bool)
        mask[0, 0] = False
        # edge_smooth=0.0: gaussian_filter is a no-op, isolating the raw ramp
        # math this test checks from the boundary-rounding blur.
        alpha = stroke_alpha(mask, 0.0, 5.0, 0.0)
        self.assertEqual(alpha[0, 1], 1.0)
        self.assertEqual(alpha[0, 5], 0.5)
        self.assertEqual(alpha[0, 6], 0.0)

    def test_down2_averages_each_2x2_block(self):
        block = np.array([[1.0, 3.0], [5.0, 7.0]])
        self.assertEqual(float(down2(block)[0, 0]), 4.0)

    def _disc_figure(self, size=200, radius=40):
        figure = np.zeros((size, size), dtype=bool)
        yy, xx = np.mgrid[0:size, 0:size]
        cy = cx = size / 2
        figure[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = True
        return figure, cy, cx, radius

    def test_band_alphas_light_thins_the_purple_band_toward_the_light(self):
        figure, cy, cx, radius = self._disc_figure()
        [purple] = band_alphas(figure, light="ne")
        r = 2 ** -0.5
        steps = np.arange(0, 40)

        def band_width(dx, dy):
            ys = np.clip((cy + (radius + steps) * dy).round().astype(int), 0, 199)
            xs = np.clip((cx + (radius + steps) * dx).round().astype(int), 0, 199)
            return purple[ys, xs].sum()

        thin_side = band_width(r, -r)    # toward the ne light
        thick_side = band_width(-r, r)   # away from it, sw
        self.assertGreater(thick_side, thin_side * 2)

    def test_band_alphas_extrusion_widths_match_thin_and_thick_multiples(self):
        figure, cy, cx, radius = self._disc_figure(size=400, radius=80)
        purple_w = 400 * PURPLE_WIDTH_PCT / 100
        [purple] = band_alphas(figure, light="n")
        visible_purple = purple >= 0.5
        column = int(cx)
        above = visible_purple[:int(cy) - radius, column]
        below = visible_purple[int(cy) + radius:, column]

        def run_length(covered: np.ndarray) -> int:
            indices = np.nonzero(covered)[0]
            return 0 if indices.size == 0 else int(indices.max() - indices.min() + 1)

        above_width = run_length(above)   # toward the "n" light: thin
        below_width = run_length(below)   # away from it: thick
        self.assertLess(abs(above_width - purple_w * delivery_style.STROKE_LIGHT_THIN), 2)
        self.assertLess(abs(below_width - purple_w * delivery_style.STROKE_LIGHT_THICK), 2)

    def test_band_alphas_antialiases_the_purple_edge_on_a_diagonal(self):
        # The rows around a disc's own 45-degree point are the worst case for
        # staircasing. Ramping off the raw distance transform inherits the
        # boundary's steps and collapses the band edge onto a handful of
        # repeated coverage levels there; rounding the field off first spreads
        # it over many.
        figure, cy, _, radius = self._disc_figure()
        r = 2 ** -0.5
        row_center = int(cy - radius * r)
        rows = np.arange(row_center - 15, row_center + 15)

        def levels(edge_smooth):
            with mock.patch.object(delivery_style, "STROKE_EDGE_SMOOTH", edge_smooth):
                [purple] = band_alphas(figure, eps_pct=0)
            values = purple[rows].ravel()
            intermediate = values[(values > 0.05) & (values < 0.95)]
            return len(np.unique(np.round(intermediate, 3)))

        raw = levels(0.0)
        self.assertGreater(raw, 0)
        self.assertGreater(levels(delivery_style.STROKE_EDGE_SMOOTH), raw * 2)

    def test_band_alphas_without_a_white_band_starts_the_purple_at_the_figure_edge(self):
        figure = np.zeros((200, 200), dtype=bool)
        figure[60:140, 60:140] = True
        edge = figure & ~ndimage.binary_erosion(figure)
        for eps in (None, 0.0):
            with self.subTest(eps=eps):
                [purple] = band_alphas(figure, eps_pct=eps)
                self.assertGreater(purple[100, 141], 0.5)
                self.assertEqual(purple[100, 150], 0.0)
        [purple] = band_alphas(figure)
        np.testing.assert_array_equal(purple[edge], 1.0)
        self.assertEqual(purple[ndimage.binary_erosion(figure)].max(), 0.0)

    def test_band_alphas_without_outlines_is_empty(self):
        figure = np.zeros((200, 200), dtype=bool)
        figure[60:140, 60:140] = True
        for eps in (None, 0.0):
            self.assertEqual(band_alphas(figure, [], light="n", eps_pct=eps), [])

    def test_clean_background_matted_blends_a_soft_edge_with_the_stroke(self):
        size = 200
        figure_rgb = (40, 40, 40)
        pixels = np.full((size, size, 3), figure_rgb, dtype=np.uint8)
        alpha = np.zeros((size, size), dtype=np.uint8)
        alpha[60:140, 60:140] = 255
        alpha[60:140, 139] = 128
        cleaned, tag = clean_background(
            png(pixels), png(alpha), backdrop="#102030", matted=True)
        arr = np.array(Image.open(io.BytesIO(cleaned)).convert("RGB")).astype(float)
        stroke = np.array(parse_color(PURPLE), dtype=float)
        expected = (128 / 255) * np.array(figure_rgb) + (1 - 128 / 255) * stroke
        np.testing.assert_allclose(arr[100, 139], expected, atol=3)
        np.testing.assert_array_equal(arr[100, 100], figure_rgb)
        self.assertIn("-matted", tag)
        self.assertNotIn("-key", tag)
        self.assertNotIn("-outline", tag)

    def test_transparent_matted_blends_a_soft_edge_with_the_stroke(self):
        size = 200
        figure_rgb = (40, 40, 40)
        pixels = np.full((size, size, 3), figure_rgb, dtype=np.uint8)
        alpha = np.zeros((size, size), dtype=np.uint8)
        alpha[60:140, 60:140] = 255
        alpha[60:140, 139] = 128
        cut, tag = transparent(png(pixels), png(alpha), matted=True)
        rgba = np.array(Image.open(io.BytesIO(cut))).astype(float)
        stroke = np.array(parse_color(PURPLE), dtype=float)
        expected = (128 / 255) * np.array(figure_rgb) + (1 - 128 / 255) * stroke
        np.testing.assert_allclose(rgba[100, 139, :3], expected, atol=3)
        self.assertEqual(rgba[100, 139, 3], 255)
        np.testing.assert_array_equal(rgba[100, 100], (*figure_rgb, 255))
        self.assertEqual(rgba[0, 0, 3], 0)
        self.assertIn("-matted", tag)
        self.assertNotIn("-key", tag)

    def test_band_alphas_without_light_matches_omitting_the_argument(self):
        figure, *_ = self._disc_figure()
        for a, b in zip(band_alphas(figure, WHITE_PURPLE),
                        band_alphas(figure, WHITE_PURPLE, light=None)):
            np.testing.assert_array_equal(a, b)

    def test_band_alphas_even_matches_no_light(self):
        figure, *_ = self._disc_figure()
        for a, b in zip(band_alphas(figure, WHITE_PURPLE, light=None),
                        band_alphas(figure, WHITE_PURPLE, light="even")):
            np.testing.assert_array_equal(a, b)

    def test_band_alphas_none_is_no_longer_a_light(self):
        figure, *_ = self._disc_figure()
        with self.assertRaises(ValueError):
            band_alphas(figure, light="none")

    def test_band_alphas_unknown_light_key_raises(self):
        figure, *_ = self._disc_figure()
        with self.assertRaises(ValueError) as context:
            band_alphas(figure, light="north")
        message = str(context.exception)
        for key in ("n", "ne", "e", "se", "s", "sw", "w", "nw"):
            self.assertIn(repr(key), message)

    def test_the_outermost_default_outline_is_the_hand_cut_mauve(self):
        self.assertEqual(delivery_style.DELIVER_DEFAULTS["outlines"][-1]["color"], "#885b80")
        self.assertEqual(parse_color("#885b80"), (136, 91, 128))

    def _noisy_circle_figure(self, size=1000, radius=300, hole_radius=90,
                             noise_amp=4.0, freq=50):
        """A jagged ring: a noisy circular figure enclosing a hole. `freq`
        cycles fit comfortably inside `radius`'s own white/purple band
        width, so a Douglas-Peucker eps under that width can straighten
        every cycle without cutting into the figure itself."""
        yy, xx = np.mgrid[0:size, 0:size]
        cy = cx = size / 2
        theta = np.arctan2(yy - cy, xx - cx)
        r = np.hypot(yy - cy, xx - cx)
        boundary = radius + noise_amp * np.sin(theta * freq)
        return (r <= boundary) & (r > hole_radius)

    def _contour_vertex_count(self, mask: np.ndarray, resimplify_eps=4.0) -> int:
        """Vertices of `mask`'s own outline, re-simplified at a small,
        fixed tolerance so the raster staircase from rasterizing a polygon
        back to pixels doesn't itself get counted as complexity."""
        contours, _ = cv2.findContours(
            (mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return sum(len(cv2.approxPolyDP(contour, resimplify_eps, True))
                  for contour in contours)

    def test_band_alphas_cut_eps_straightens_a_noisy_outline(self):
        figure = self._noisy_circle_figure()
        wide_white = [{"color": "#ffffff", "width": 1.3}, PURPLE_ONLY[0]]
        smooth_white, _ = band_alphas(figure, wide_white, eps_pct=0)
        cut_white, _ = band_alphas(figure, wide_white, eps_pct=1.0)
        smooth_vertices = self._contour_vertex_count(smooth_white >= 0.5)
        cut_vertices = self._contour_vertex_count(cut_white >= 0.5)
        self.assertLess(cut_vertices, smooth_vertices / 3)

    def test_band_alphas_cut_eps_keeps_the_enclosed_hole(self):
        figure = self._noisy_circle_figure()
        white_a, purple_a = band_alphas(figure, WHITE_PURPLE, eps_pct=1.0)
        cy = cx = 500
        # Well inside the hole radius (90), away from either band's reach.
        self.assertLess(white_a[cy, cx], 0.5)
        self.assertLess(purple_a[cy, cx], 0.5)

    def test_transparent_light_appends_a_tag_suffix(self):
        pixels = np.full((256, 256, 3), (210, 230, 235), dtype=np.uint8)
        pixels[64:192, 64:160] = (40, 40, 40)
        _, tag = transparent(png(pixels), matte(pixels.shape[:2], (64, 192, 64, 160)),
                             light="ne")
        self.assertEqual(tag, "transparent-o3-cut0.5-light-ne")

    def test_clean_background_light_appends_a_tag_suffix(self):
        pixels = np.full((32, 32, 3), (210, 230, 235), dtype=np.uint8)
        pixels[8:24, 10:22] = (40, 40, 40)
        _, tag = clean_background(png(pixels), matte(pixels.shape[:2], (8, 24, 10, 22)),
                                  light="sw")
        self.assertRegex(tag, r"^clean-o\d+-cut0\.5-light-sw-shadow$")

    def test_clean_background_backdrop_stripes_tag_and_pattern(self):
        pixels = np.full((64, 64, 3), (210, 230, 235), dtype=np.uint8)
        pixels[16:48, 16:48] = (40, 40, 40)
        cleaned, tag = clean_background(
            png(pixels), matte(pixels.shape[:2], (16, 48, 16, 48)),
            backdrop="stripes")
        self.assertRegex(tag, r"^clean-o\d+-bg-stripes-cut0\.5$")
        arr = np.array(Image.open(io.BytesIO(cleaned)).convert("RGB"))
        corners = [tuple(arr[0, 0]), tuple(arr[0, -1]),
                  tuple(arr[-1, 0]), tuple(arr[-1, -1])]
        self.assertGreater(len(set(corners)), 1)

    def test_clean_background_backdrop_colour_tag_suffix(self):
        pixels = np.full((32, 32, 3), (210, 230, 235), dtype=np.uint8)
        pixels[8:24, 10:22] = (40, 40, 40)
        _, tag = clean_background(png(pixels), matte(pixels.shape[:2], (8, 24, 10, 22)),
                                  backdrop="#c7e5e9")
        self.assertRegex(tag, r"^clean-o\d+-bg-c7e5e9-cut0\.5$")

    def test_clean_background_backdrop_none_tag_unchanged(self):
        pixels = np.full((32, 32, 3), (210, 230, 235), dtype=np.uint8)
        pixels[8:24, 10:22] = (40, 40, 40)
        _, tag = clean_background(png(pixels), matte(pixels.shape[:2], (8, 24, 10, 22)))
        self.assertRegex(tag, r"^clean-o\d+-cut0\.5$")

    def test_clean_background_cut_tag_suffix_absent_at_eps_zero(self):
        pixels = np.full((32, 32, 3), (210, 230, 235), dtype=np.uint8)
        pixels[8:24, 10:22] = (40, 40, 40)
        with mock.patch.object(delivery_style, "STROKE_CUT_EPS_PCT", 0):
            _, tag = clean_background(png(pixels), matte(pixels.shape[:2], (8, 24, 10, 22)))
        self.assertRegex(tag, r"^clean-o\d+$")
        self.assertNotIn("-cut", tag)

    def test_keyed_coverage_ramps_the_edge_band_by_colour_distance(self):
        local = np.full((40, 40, 3), (58, 67, 81), dtype=float)
        pixels = local.copy()
        figure = np.zeros((40, 40), dtype=bool)
        figure[10:30, 10:30] = True
        pixels[10:30, 10:30] = (190, 170, 220)
        pixels[10, 20] = (78, 67, 81)  # distance 20 == tolerance, on the figure's own edge row
        coverage = keyed_coverage(pixels, figure, local, 3, 20)
        self.assertEqual(coverage[20, 20], 1.0)
        self.assertEqual(coverage[0, 0], 0.0)
        self.assertAlmostEqual(coverage[10, 20], 0.5, delta=0.02)

    def test_keyed_coverage_is_the_hard_figure_below_one_pixel_of_band(self):
        pixels = np.zeros((8, 8, 3))
        local = np.zeros((8, 8, 3))
        figure = np.zeros((8, 8), dtype=bool)
        figure[2:6, 2:6] = True
        coverage = keyed_coverage(pixels, figure, local, 0, 20)
        np.testing.assert_array_equal(coverage, figure.astype(float))

    def test_unpremultiply_solves_the_edge_blend_back_to_figure_colour(self):
        local = np.zeros((1, 1, 3))
        local[:] = (65, 198, 73)
        figure_colour = np.array((100.0, 40.0, 160.0))
        blended = (0.5 * figure_colour + 0.5 * local[0, 0]).reshape(1, 1, 3)
        coverage = np.full((1, 1), 0.5)
        solved = unpremultiply(blended, local, coverage)
        np.testing.assert_allclose(solved[0, 0], figure_colour, atol=1)

    def test_unpremultiply_leaves_fully_covered_and_fully_backdrop_pixels_alone(self):
        local = np.full((1, 2, 3), (65, 198, 73), dtype=float)
        pixels = np.array([[(100.0, 40.0, 160.0), (65.0, 198.0, 73.0)]])
        coverage = np.array([[1.0, 0.0]])
        np.testing.assert_array_equal(unpremultiply(pixels, local, coverage), pixels)

    def test_despill_green_key_removes_green_excess_from_figure_only(self):
        key = np.array((60.0, 200.0, 60.0))
        figure = np.zeros((4, 4), dtype=bool)
        figure[1:3, 1:3] = True
        pixels = np.zeros((4, 4, 3))
        pixels[..., 0] = 50
        pixels[..., 1] = 220
        pixels[..., 2] = 40
        result = despill(pixels, figure, key)
        self.assertTrue((result[..., 1][figure]
                        <= np.maximum(result[..., 0], result[..., 2])[figure]).all())
        np.testing.assert_array_equal(result[..., 1][~figure], 220)

    def test_despill_teal_key_clears_its_chroma_and_leaves_lavender_alone(self):
        key = np.array((86.0, 186.0, 155.0))
        direction = key - key.mean()
        direction = direction / np.linalg.norm(direction)
        figure = np.ones((1, 2), dtype=bool)
        lavender = np.array((220.0, 200.0, 240.0))
        pixels = np.array([[0.5 * lavender + 0.5 * key, lavender]])
        result = despill(pixels, figure, key)
        chroma = result[0, 0] - result[0, 0].mean()
        self.assertLessEqual(float(chroma @ direction), 0.5)
        np.testing.assert_array_equal(result[0, 1], lavender)

    def test_despill_grey_key_is_a_no_op(self):
        key = np.array((200.0, 200.0, 200.0))
        figure = np.ones((2, 2), dtype=bool)
        pixels = np.array([[(10.0, 250.0, 5.0), (90.0, 200.0, 30.0)],
                           [(5.0, 5.0, 5.0), (255.0, 0.0, 0.0)]])
        np.testing.assert_array_equal(despill(pixels, figure, key), pixels)

    def test_despill_excess_below_minimum_is_a_no_op(self):
        key = np.array((100.0, 89.0, 50.0))  # excess 11, one under KEY_DESPILL_MIN_EXCESS
        figure = np.ones((2, 2), dtype=bool)
        pixels = np.array([[(10.0, 250.0, 5.0), (90.0, 200.0, 30.0)],
                           [(5.0, 5.0, 5.0), (255.0, 0.0, 0.0)]])
        np.testing.assert_array_equal(despill(pixels, figure, key), pixels)

    def test_clean_background_keyed_edge_blends_and_despills_on_a_green_backdrop(self):
        # A green-screen raw: a flat green backdrop, a figure with one edge
        # row that is a 50/50 mix of the figure colour and the backdrop --
        # the soft matte artefact the keyed edge exists to blend cleanly
        # instead of cutting as a hard 0/1 pixel. The delivered backdrop is
        # set to the same colour as the raw one, isolating the blend/despill
        # arithmetic from the separate choice of coverage ramp value.
        height = width = 200
        box = (60, 140, 60, 140)
        green = np.array((60.0, 90.0, 60.0))
        figure_colour = np.array((120.0, 70.0, 60.0))
        mixed_row = 0.5 * figure_colour + 0.5 * green

        pixels = np.zeros((height, width, 3))
        pixels[:] = green
        pixels[box[0]:box[1], box[2]:box[3]] = figure_colour
        pixels[box[0], box[2]:box[3]] = mixed_row
        backdrop_hex = "#%02x%02x%02x" % tuple(int(c) for c in green)

        cleaned, tag = clean_background(
            png(pixels), matte((height, width), box), backdrop=backdrop_hex)
        self.assertTrue(re.search(r"-key(-light-|$)", tag))
        arr = np.array(Image.open(io.BytesIO(cleaned)).convert("RGB")).astype(float)

        # The mixed row is the figure's own edge: its partial coverage blends
        # into the purple stroke under it, not into the green backdrop, and
        # it carries no green fringe.
        edge = arr[box[0], 100]
        self.assertLessEqual(edge[1], max(edge[0], edge[2]))
        np.testing.assert_allclose(
            arr[box[0] - 1, 100], parse_color(PURPLE), atol=12)
        np.testing.assert_array_equal(arr[box[0] + 5, 100], figure_colour)

        fig_slice = (slice(box[0], box[1]), slice(box[2], box[3]))
        r, g, b = arr[fig_slice][..., 0], arr[fig_slice][..., 1], arr[fig_slice][..., 2]
        self.assertTrue((g <= np.maximum(r, b)).all())

        _, light_tag = clean_background(
            png(pixels), matte((height, width), box),
            backdrop=backdrop_hex, light="ne")
        self.assertRegex(light_tag, r"-key-light-ne-shadow$")

    def test_figure_rim_is_the_outermost_band_and_the_whole_figure_without_one(self):
        figure = np.zeros((64, 64), dtype=bool)
        figure[16:48, 16:48] = True
        rim = figure_rim(figure, 4)
        self.assertTrue(rim[16:20, 16:48].all())
        self.assertFalse(rim[20:44, 20:44].any())
        np.testing.assert_array_equal(figure_rim(figure, 0), figure)

    def test_clean_background_leaves_the_skin_inside_the_rim_on_a_yellow_green_key(self):
        skin = (251, 222, 206)
        pixels = np.full((1024, 1024, 3), (184, 210, 145), dtype=np.uint8)
        pixels[256:768, 256:768] = skin
        soft = np.zeros((1024, 1024), dtype=np.uint8)
        soft[256:768, 256:768] = 255
        cleaned, tag = clean_background(png(pixels), png(soft))
        arr = np.array(Image.open(io.BytesIO(cleaned)))
        self.assertTrue(tag.endswith("-key"))
        np.testing.assert_array_equal(arr[512, 512], skin)
        self.assertGreater(int(arr[512, 258, 2]), int(arr[512, 258, 1]))

    def test_clean_background_grey_backdrop_matches_the_old_arithmetic(self):
        height = width = 32
        pixels = np.full((height, width, 3), (210, 230, 235), dtype=np.uint8)
        pixels[8:24, 10:22] = (40, 40, 40)
        cleaned, tag = clean_background(png(pixels), matte(pixels.shape[:2], (8, 24, 10, 22)))
        self.assertNotIn("-key", tag)
        arr = np.array(Image.open(io.BytesIO(cleaned)).convert("RGB"))
        np.testing.assert_array_equal(arr[16, 16], (40, 40, 40))

    def _sticker_inputs(self, size=240, half=40):
        pixels = np.zeros((size, size, 3))
        figure = np.zeros((size, size), dtype=bool)
        centre = size // 2
        figure[centre - half:centre + half, centre - half:centre + half] = True
        coverage = figure.astype(float)
        backdrop_rgb = np.full((size, size, 3), 200.0)
        return pixels, figure, coverage, backdrop_rgb, centre, half

    # Just past the thick (away-from-light) band's own reach, where only the
    # shadow's shift -- not the band itself -- can still darken a pixel.
    _SHADOW_ROWS = slice(8, 12)

    def test_sticker_light_darkens_the_backdrop_away_from_the_light(self):
        pixels, figure, coverage, backdrop_rgb, centre, half = self._sticker_inputs()
        composite = sticker(pixels, figure, coverage, backdrop_rgb, light="n", shadow=True)
        below = composite[centre + half + self._SHADOW_ROWS.start:
                         centre + half + self._SHADOW_ROWS.stop,
                         centre - 10:centre + 10]
        self.assertTrue((below < 199.0).any())

    def test_sticker_light_leaves_the_backdrop_on_the_light_side_unchanged(self):
        pixels, figure, coverage, backdrop_rgb, centre, half = self._sticker_inputs()
        composite = sticker(pixels, figure, coverage, backdrop_rgb, light="n", shadow=True)
        above = composite[centre - half - 40:centre - half - 20,
                          centre - 10:centre + 10]
        np.testing.assert_allclose(above, 200.0)

    def test_sticker_without_light_leaves_the_backdrop_outside_the_bands_unchanged(self):
        pixels, figure, coverage, backdrop_rgb, centre, half = self._sticker_inputs()
        composite = sticker(pixels, figure, coverage, backdrop_rgb, light=None, shadow=True)
        below = composite[centre + half + self._SHADOW_ROWS.start:
                         centre + half + self._SHADOW_ROWS.stop,
                         centre - 10:centre + 10]
        above = composite[centre - half - 40:centre - half - 20,
                          centre - 10:centre + 10]
        np.testing.assert_allclose(below, 200.0)
        np.testing.assert_allclose(above, 200.0)

    def test_sticker_shadow_falls_straight_away_from_the_light(self):
        pixels, figure, coverage, backdrop_rgb, centre, half = self._sticker_inputs()
        bare = sticker(pixels, figure, coverage, backdrop_rgb, light="n")
        lit = sticker(pixels, figure, coverage, backdrop_rgb, light="n", shadow=True)
        cols = np.where((np.abs(lit - bare).sum(axis=2) > 1).any(axis=0))[0]
        self.assertAlmostEqual((cols.min() + cols.max()) / 2, centre, delta=1)

    def test_sticker_even_and_no_outlines_throw_no_shadow(self):
        pixels, figure, coverage, backdrop_rgb, centre, half = self._sticker_inputs()
        for light, outlines in (("even", None), ("n", [])):
            composite = sticker(pixels, figure, coverage, backdrop_rgb,
                                light=light, shadow=True, outlines=outlines)
            below = composite[centre + half + self._SHADOW_ROWS.start:
                             centre + half + self._SHADOW_ROWS.stop,
                             centre - 10:centre + 10]
            np.testing.assert_allclose(below, 200.0)

    def test_sticker_shadow_from_throws_a_shadow_with_an_even_band(self):
        pixels, figure, coverage, backdrop_rgb, centre, half = self._sticker_inputs()
        bare = sticker(pixels, figure, coverage, backdrop_rgb, light="even")
        lit = sticker(pixels, figure, coverage, backdrop_rgb, light="even",
                      shadow=True, shadow_from="n")
        rows = np.where((np.abs(lit - bare).sum(axis=2) > 1).any(axis=1))[0]
        self.assertGreater(rows.size, 0)
        self.assertGreater(rows.min(), centre - half)
        self.assertGreater(rows.max(), centre + half)

    def test_sticker_throws_no_shadow_unless_asked(self):
        pixels, figure, coverage, backdrop_rgb, centre, half = self._sticker_inputs()
        composite = sticker(pixels, figure, coverage, backdrop_rgb, light="n")
        below = composite[centre + half + self._SHADOW_ROWS.start:
                         centre + half + self._SHADOW_ROWS.stop,
                         centre - 10:centre + 10]
        np.testing.assert_allclose(below, 200.0)


def purple_only_input():
    """The synthetic figure tests/fixtures/delivery-purple-only.npz was
    rendered from, with the white band at 0, before outlines were a list."""
    height, width = 180, 140
    yy, xx = np.mgrid[0:height, 0:width]
    body = ((xx - 70) / 34.0) ** 2 + ((yy - 95) / 62.0) ** 2 <= 1.0
    arm = (np.abs(yy - 70) < 7) & (xx > 30) & (xx < 120)
    soft = ndimage.gaussian_filter((body | arm).astype(float), 1.2)
    px = np.zeros((height, width, 3), np.uint8)
    px[..., 0] = 120 + (xx % 50)
    px[..., 1] = 90 + (yy % 40)
    px[..., 2] = 160
    return px, np.clip(np.rint(soft * 255), 0, 255).astype(np.uint8)


def decode(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)))


class OutlinesTest(unittest.TestCase):
    def square(self, size=200):
        figure = np.zeros((size, size), dtype=bool)
        figure[60:140, 60:140] = True
        return figure

    def test_a_purple_only_list_reproduces_the_white_zero_delivery(self):
        golden = np.load(Path(__file__).parent / "fixtures" / "delivery-purple-only.npz")
        px, soft = purple_only_input()
        image, matte_png = png(px), png(soft)
        figure = soft >= 128
        rendered = {
            "clean_n": clean_background(image, matte_png, light="n", backdrop="dots",
                                        matted=True, outlines=PURPLE_ONLY)[0],
            "clean_even": clean_background(image, matte_png, light="even",
                                           backdrop="#c7e5e9", matted=True,
                                           outlines=PURPLE_ONLY)[0],
            "clean_moon": clean_background(image, matte_png, light="se", backdrop="dots",
                                           scene="moon", light_from="se", matted=True,
                                           outlines=PURPLE_ONLY)[0],
            "transparent_n": transparent(image, matte_png, light="n", matted=True,
                                         outlines=PURPLE_ONLY)[0],
            "transparent_even": transparent(image, matte_png, light="even", matted=True,
                                            outlines=PURPLE_ONLY)[0],
        }
        for name, data in rendered.items():
            with self.subTest(name=name):
                np.testing.assert_array_equal(decode(data), golden[name])
        for light in ("even", "n"):
            for eps in (None, 0.0):
                with self.subTest(light=light, eps=eps):
                    [purple] = band_alphas(figure, PURPLE_ONLY, light, eps)
                    np.testing.assert_allclose(purple, golden[f"bands_{light}_{eps}"], rtol=0, atol=1e-12)

    def test_two_bands_put_the_white_inside_the_purple(self):
        figure = self.square()
        wide = [{"color": "#ffffff", "width": 2.0}, {"color": PURPLE, "width": 2.0}]
        for eps in (None, 0.0):
            with self.subTest(eps=eps), mock.patch.object(
                    delivery_style, "STROKE_CUT_EPS_PCT", 0.5 if eps is None else 0):
                white, purple = band_alphas(figure, wide)
                composite = sticker(np.zeros((200, 200, 3)), figure,
                                    figure.astype(float), (0, 0, 0), "even",
                                    outlines=wide)
                self.assertGreater(white[100, 141], 0.5)
                self.assertGreater(purple[100, 146], 0.5)
                self.assertLess(white[100, 146], 0.5)
                np.testing.assert_allclose(composite[100, 141], 255, atol=1)
                np.testing.assert_allclose(composite[100, 146], parse_color(PURPLE), atol=1)
                self.assertEqual(composite[100, 160].max(), 0)

    def test_three_bands_stack_outward_in_order(self):
        figure = self.square()
        three = [{"color": "#ff0000", "width": 1.0}, {"color": "#00ff00", "width": 1.0},
                 {"color": "#0000ff", "width": 1.0}]
        alphas = band_alphas(figure, three, light="even", eps_pct=0)
        self.assertEqual(len(alphas), 3)
        composite = sticker(np.zeros((200, 200, 3)), figure, figure.astype(float),
                            (255, 255, 255), "even", outlines=three)
        for column, colour in ((141, (255, 0, 0)), (143, (0, 255, 0)),
                               (145, (0, 0, 255)), (160, (255, 255, 255))):
            with self.subTest(column=column):
                np.testing.assert_allclose(composite[100, column], colour, atol=2)

    def test_the_innermost_band_runs_under_the_figure_edge(self):
        figure = self.square()
        edge = figure & ~ndimage.binary_erosion(figure)
        white, purple = band_alphas(figure, WHITE_PURPLE)
        np.testing.assert_array_equal(white[edge], 1.0)
        self.assertEqual(purple[figure].max(), 0.0)

    def test_the_light_extrudes_only_the_outermost_band(self):
        figure = self.square(400)
        even = band_alphas(figure, WHITE_PURPLE, light="even")
        lit = band_alphas(figure, WHITE_PURPLE, light="n")
        np.testing.assert_array_equal(even[0], lit[0])
        self.assertGreater((lit[1] >= 0.5).sum(), (even[1] >= 0.5).sum())

    def test_no_bands_frame_nothing_and_throw_no_shadow(self):
        figure = self.square()
        backdrop = np.full((200, 200, 3), 200.0)
        composite = sticker(np.zeros((200, 200, 3)), figure, figure.astype(float),
                            backdrop, "n", shadow=True, outlines=[])
        np.testing.assert_array_equal(composite[~figure], 200.0)
        cut, _ = transparent(png(np.full((200, 200, 3), 40, np.uint8)),
                             png(figure.astype(np.uint8) * 255), matted=True,
                             outlines=[])
        self.assertEqual(decode(cut)[..., 3][~figure].max(), 0)

    def test_the_layers_composite_back_to_the_delivered_picture(self):
        px, soft = purple_only_input()
        for kwargs in (dict(light="n", backdrop="dots"),
                       dict(light="se", backdrop="dots", scene="moon", light_from="se")):
            with self.subTest(**kwargs):
                picture, figure, outline, backdrop, _ = deliver_pngs(
                    png(px), png(soft), matted=True, outlines=WHITE_PURPLE, **kwargs)
                fig, out = decode(figure) / 255.0, decode(outline) / 255.0
                result = decode(backdrop).astype(float)
                for layer in (out, fig):
                    result = (layer[..., :3] * 255 * layer[..., 3:]
                              + result * (1 - layer[..., 3:]))
                np.testing.assert_allclose(result, decode(picture), atol=2)

    def test_a_transparent_delivery_has_no_backdrop_layer(self):
        px, soft = purple_only_input()
        picture, figure, outline, backdrop, _ = deliver_pngs(
            png(px), png(soft), transparent=True, matted=True, light="n")
        self.assertIsNone(backdrop)
        self.assertEqual(Image.open(io.BytesIO(figure)).mode, "RGBA")
        out, delivered = decode(outline), decode(picture)
        self.assertGreater(out[..., 3].max(), 0)
        self.assertTrue((delivered[..., 3].astype(int) >= out[..., 3].astype(int) - 1).all())

    def test_the_bands_follow_the_silhouette_not_a_thinned_matte(self):
        px, soft = purple_only_input()
        thinned = soft.copy()
        inside = np.argwhere(soft == 255)
        top, left = inside.min(axis=0) + 20
        bottom, right = inside.max(axis=0) - 20
        thinned[top:bottom, left:right] = 80
        for transparent in (False, True):
            with self.subTest(transparent=transparent):
                kwargs = dict(matted=True, light="n", transparent=transparent,
                              outlines=WHITE_PURPLE)
                _, figure, outline, _, _ = deliver_pngs(
                    png(px), png(thinned), silhouette=png(soft), **kwargs)
                _, _, expected, _, _ = deliver_pngs(png(px), png(soft), **kwargs)
                np.testing.assert_array_equal(decode(outline), decode(expected))
                self.assertEqual(decode(figure)[top + 5, left + 5, 3], 80)

    def test_a_kept_scene_is_its_own_backdrop_with_an_empty_outline(self):
        px, soft = purple_only_input()
        picture, figure, outline, backdrop, tag = deliver_pngs(
            png(px), png(soft), keep_scene=True)
        self.assertEqual(tag, "scene")
        np.testing.assert_array_equal(decode(backdrop), px)
        np.testing.assert_array_equal(decode(picture), px)
        self.assertEqual(decode(outline)[..., 3].max(), 0)
        np.testing.assert_array_equal(decode(figure)[..., 3], soft)


if __name__ == "__main__":
    unittest.main()


class SkinPinTest(unittest.TestCase):
    def source(self):
        """A warm low-saturation cheek with a high-saturation lip inside it."""
        pixels = np.full((256, 256, 3), (210, 230, 235), dtype=np.uint8)
        pixels[64:192, 64:192] = (247, 226, 209)   # cheek: warm, S ~ 25
        pixels[120:136, 112:144] = (222, 40, 40)   # lip: warm, S ~ 200
        return pixels

    def test_the_cheek_is_pinned_to_the_skin_hue(self):
        source = self.source()
        redrawn = source.copy()
        redrawn[64:192, 64:192] = (226, 209, 247)  # the redraw's lavender cheek
        out, report = repin_skin_png(png(source), png(redrawn))
        hsv = np.array(Image.open(io.BytesIO(out)).convert("HSV"))
        target = delivery_style.PALETTE_WINDOWS[1]["hue_target"]
        self.assertAlmostEqual(float(hsv[96, 96, 0]), target, delta=3)
        self.assertIn("skin pinned", report[0])

    def test_the_lip_keeps_its_own_hue(self):
        source = self.source()
        out, _ = repin_skin_png(png(source), png(source))
        hsv = np.array(Image.open(io.BytesIO(out)).convert("HSV"))
        before = np.array(Image.fromarray(source).convert("HSV"))
        self.assertAlmostEqual(float(hsv[128, 128, 0]), float(before[128, 128, 0]),
                               delta=2)


    def test_a_face_the_base_drew_lavender_is_left_alone(self):
        source = self.source()
        source[64:192, 64:192] = (226, 209, 247)   # no skin drawn anywhere
        redrawn = source.copy()
        out, report = repin_skin_png(png(source), png(redrawn))
        self.assertEqual(out, png(redrawn))
        self.assertIn("not pinned", report[0])

    def test_the_hue_turns_the_short_way_round(self):
        # 191 down to 18 crosses green; the short way crosses red.
        source = self.source()
        redrawn = source.copy()
        redrawn[64:192, 64:192] = (226, 209, 247)
        out, _ = repin_skin_png(png(source), png(redrawn))
        hsv = np.array(Image.open(io.BytesIO(out)).convert("HSV"))
        cheek = hsv[64:192, 64:192, 0].astype(int)
        self.assertFalse(((cheek > 60) & (cheek < 150)).any())
