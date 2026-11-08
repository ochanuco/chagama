from __future__ import annotations

import base64
import io
import unittest

import numpy as np
from PIL import Image

from chagama.infrastructure.reference import outline


class Segmenter:
    def run(self, outputs, feeds):
        mask = np.zeros((1, 1, 1024, 1024), np.float32)
        mask[..., 200:800, 300:700] = 1.0
        return [mask]


class PhotoSegmenter:
    class Input:
        name = "input_image"

    def get_inputs(self):
        return [self.Input()]

    def run(self, outputs, feeds):
        logits = np.full((1, 1, 1024, 1024), -20.0, np.float32)
        logits[..., 256:768, 256:768] = 20.0
        return [logits]


def anime(rgb):
    return outline.figure_mask(rgb, Segmenter())


def nobody(rgb):
    return np.zeros(rgb.shape[:2], np.float32)


def photo(rgb):
    return outline.photo_mask(rgb, PhotoSegmenter())


SEGMENTERS = [("anime", anime)]


def near_left(image):
    width, height = image.size
    return np.tile(np.linspace(1.0, 0.0, width, dtype=np.float32), (height, 1))


def decode(data):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data))))


class OutlineTest(unittest.TestCase):
    def picture(self):
        rgb = np.full((400, 200, 3), 255, np.uint8)
        rgb[100:300, 50:150] = 40
        rgb[360:380, 10:30] = 0
        return Image.fromarray(rgb)

    def test_the_silhouette_is_the_figure_in_black(self):
        found = outline.outlines(self.picture(), SEGMENTERS, near_left)
        silhouette = decode(found["silhouette"])
        self.assertEqual(silhouette.shape, (400, 200))
        self.assertEqual(silhouette[200, 100], 0)
        self.assertEqual(silhouette[10, 10], 255)

    def test_the_lineart_draws_edges_inside_the_figure_only(self):
        lines = decode(outline.outlines(self.picture(), SEGMENTERS, near_left)["lineart"])
        self.assertLess(lines[200, 49:52].min(), 128)
        self.assertEqual(lines[200, 100], 255)
        self.assertEqual(lines[355:385, 5:35].min(), 255)

    def test_a_large_picture_is_shrunk(self):
        big = Image.new("RGB", (1600, 800), "white")
        self.assertEqual(decode(outline.outlines(big, SEGMENTERS, near_left)["silhouette"]).shape,
                         (384, 768))

    def test_the_depth_is_white_near_inside_the_figure_and_black_outside(self):
        depth = decode(outline.outlines(self.picture(), SEGMENTERS, near_left)["depth"])
        self.assertEqual(depth.shape, (400, 200))
        self.assertEqual(depth[10, 100], 0)
        self.assertGreater(depth[200, 30], depth[200, 160])
        self.assertGreaterEqual(depth[200, 160], round(outline.DEPTH_RANGE[0] * 255) - 1)

    def test_a_photo_anime_seg_misses_is_cut_by_the_next_segmenter(self):
        found = outline.outlines(self.picture(), [("anime", nobody), ("photo", photo)],
                                 near_left)
        self.assertEqual(found["cut"], "photo")
        self.assertEqual(decode(found["silhouette"])[200, 100], 0)
        self.assertEqual(decode(found["silhouette"])[10, 10], 255)

    def test_the_anime_cut_is_named(self):
        self.assertEqual(outline.outlines(self.picture(), SEGMENTERS, near_left)["cut"], "anime")

    def test_no_figure_cut_leaves_no_pictures(self):
        self.assertEqual(outline.outlines(self.picture(), [("anime", nobody), ("photo", nobody)],
                                          near_left), {"cut": None})


class HairTest(unittest.TestCase):
    """A body with a strand of hair hanging off its side, a lock lying across
    it, and a speck the cut kept apart."""

    def masks(self):
        inside = np.zeros((400, 400), bool)
        inside[100:300, 150:250] = True
        inside[120:380, 250:270] = True
        inside[150:160, 150:250] = True
        inside[20:26, 20:26] = True
        hair = np.zeros_like(inside)
        hair[120:380, 250:270] = True
        hair[150:160, 150:250] = True
        return inside, hair

    def test_hair_off_the_body_and_specks_are_cut_and_hair_over_it_filled(self):
        inside, hair = self.masks()
        kept, filled = outline.without_hair(inside, hair)
        self.assertTrue(kept[200, 200])
        self.assertFalse(kept[350, 260])
        self.assertFalse(kept[22, 22])
        self.assertTrue(filled[155, 200])
        self.assertFalse(filled[200, 200])

    def test_hair_in_the_head_keeps_its_depth(self):
        inside, hair = self.masks()
        head = np.zeros_like(inside)
        head[100:200, 240:280] = True
        kept, filled = outline.without_hair(inside, hair, head)
        self.assertTrue(kept[130, 260])
        self.assertFalse(filled[130, 260])

    def test_a_piece_the_head_cuts_off_goes_unless_a_body_point_holds_it(self):
        inside = np.zeros((400, 400), bool)
        inside[150:390, 100:300] = True
        inside[60:150, 150:250] = True
        inside[40:60, 120:150] = True
        inside[40:60, 250:280] = True
        inside[40:60, 280:300] = True
        head = np.zeros_like(inside)
        head[40:150, 150:250] = True
        head[60:150, 140:260] = True
        anchors = np.zeros_like(inside)
        anchors[50, 290] = True
        kept, filled = outline.without_hair(inside, np.zeros_like(inside), head, anchors)
        self.assertTrue(kept[300, 200])
        self.assertTrue(kept[100, 200])
        self.assertFalse(kept[50, 130])
        self.assertTrue(kept[50, 290])
        self.assertFalse(filled.any())

    def test_the_depth_fills_hair_over_the_body_from_the_body_around_it(self):
        inside, hair = self.masks()
        relative = np.ones(inside.shape, np.float32)
        relative[150:160] = 5.0
        picture = outline.depth(relative, inside.astype(np.float32), hair)
        self.assertEqual(picture[350, 260], 0)
        self.assertEqual(picture[22, 22], 0)
        self.assertGreater(picture[155, 200], 0)
        self.assertLessEqual(abs(int(picture[155, 200]) - int(picture[200, 200])), 2)

    def test_outlines_leave_the_hair_it_is_given_out_of_the_depth(self):
        rgb = np.full((400, 200, 3), 255, np.uint8)
        rgb[100:300, 50:150] = 40

        def hair_on_the_right(image):
            likelihood = np.zeros((image.size[1], image.size[0]), np.float32)
            likelihood[:, 140:] = 1.0
            return likelihood

        def wide(rgb):
            mask = np.zeros(rgb.shape[:2], np.float32)
            mask[100:300, 50:190] = 1.0
            return mask

        found = outline.outlines(Image.fromarray(rgb), [("anime", wide)], near_left,
                                 hair_on_the_right)
        depth = decode(found["depth"])
        self.assertEqual(depth[200, 170], 0)
        self.assertGreater(depth[200, 100], 0)
        self.assertEqual(decode(found["silhouette"])[200, 170], 0)
