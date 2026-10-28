"""Node-pack tests for comfy_nodes/yukari_finalize.

The pack lives outside src/ so ComfyUI can reach it through a junction named
after the package; sys.path is extended here the same way, by resolving this
file's own path rather than relying on the working directory.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfy_nodes.yukari_finalize import bridge, nodes  # noqa: E402


class FakeTensor:
    """Stand-in for a torch tensor: only what the bridge calls on one."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def __getitem__(self, index):
        return FakeTensor(self.array[index])


class FakeTorch:
    @staticmethod
    def from_numpy(array):
        return FakeTensor(array)


def swatch() -> np.ndarray:
    """64x64 flat #808080 backdrop with a saturated 32x32 centre block."""
    pixels = np.full((64, 64, 3), 0x80, dtype=np.uint8)
    hsv = np.zeros((32, 32, 3), dtype=np.uint8)
    hsv[..., 0], hsv[..., 1], hsv[..., 2] = (10, 180, 160)
    pixels[16:48, 16:48] = np.array(Image.fromarray(hsv, "HSV").convert("RGB"))
    return pixels


def matte_array() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    return mask


def image_tensor(pixels: np.ndarray) -> FakeTensor:
    return FakeTensor((pixels.astype(np.float32) / 255.0)[None, ...])


def mask_tensor(mask: np.ndarray) -> FakeTensor:
    return FakeTensor((mask.astype(np.float32) / 255.0)[None, ...])


class BridgeTest(unittest.TestCase):
    def test_array_png_round_trip_rgb(self):
        pixels = swatch()
        data = bridge.array_to_png(pixels, "RGB")
        back = bridge.png_to_array(data, "RGB")
        np.testing.assert_array_equal(back, pixels)

    def test_array_png_round_trip_l(self):
        mask = matte_array()
        data = bridge.array_to_png(mask, "L")
        back = bridge.png_to_array(data, "L")
        np.testing.assert_array_equal(back, mask)

    def test_image_tensor_round_trips_through_png(self):
        pixels = swatch()
        with mock.patch.dict(sys.modules, {"torch": FakeTorch()}):
            data = bridge.image_to_png(image_tensor(pixels))
            back = bridge.png_to_image(data)
        self.assertEqual(back.array.shape, (1, 64, 64, 3))
        np.testing.assert_allclose(
            (back.array[0] * 255.0).round(), pixels, atol=1)

    def test_mask_tensor_becomes_an_l_png(self):
        mask = matte_array()
        data = bridge.mask_to_png(mask_tensor(mask))
        back = bridge.png_to_array(data, "L")
        np.testing.assert_array_equal(back, mask)

    def test_png_to_image_rgba_yields_four_channels(self):
        pixels = swatch()
        alpha = matte_array()
        rgba = np.dstack([pixels, alpha])
        data = bridge.array_to_png(rgba, "RGBA")
        with mock.patch.dict(sys.modules, {"torch": FakeTorch()}):
            back = bridge.png_to_image(data, "RGBA")
        self.assertEqual(back.array.shape, (1, 64, 64, 4))

    def test_image_to_png_round_trips_a_four_channel_tensor_as_rgba(self):
        pixels = swatch()
        alpha = matte_array()
        rgba = np.dstack([pixels, alpha])
        with mock.patch.dict(sys.modules, {"torch": FakeTorch()}):
            data = bridge.image_to_png(image_tensor(rgba))
            back = bridge.png_to_image(data, "RGBA")
        self.assertEqual(Image.open(io.BytesIO(data)).mode, "RGBA")
        self.assertEqual(back.array.shape, (1, 64, 64, 4))
        np.testing.assert_allclose(
            (back.array[0] * 255.0).round(), rgba, atol=1)

    def test_png_to_mask_yields_a_batched_hw_tensor(self):
        mask = matte_array()
        data = bridge.array_to_png(mask, "L")
        with mock.patch.dict(sys.modules, {"torch": FakeTorch()}):
            back = bridge.png_to_mask(data)
        self.assertEqual(back.array.shape, (1, 64, 64))
        np.testing.assert_allclose((back.array[0] * 255.0).round(), mask, atol=1)


class NodeMappingTest(unittest.TestCase):
    def test_node_class_mappings_cover_every_node(self):
        self.assertEqual(set(nodes.NODE_CLASS_MAPPINGS), {
            "YukariMatting",
            "YukariForeground", "YukariSeeThrough", "YukariDeliver", "YukariDepthOfField",
            "YukariLight", "YukariViewfinder", "YukariLens", "YukariPhotoStage",
            "YukariStage", "YukariRelight", "YukariSaturation", "YukariPhotoDepthOfField",
            "YukariPoseSkeleton", "YukariPoseOutline",
        })
        self.assertEqual(
            set(nodes.NODE_DISPLAY_NAME_MAPPINGS),
            set(nodes.NODE_CLASS_MAPPINGS))

    def test_importing_the_package_does_not_require_comfyui(self):
        for name in list(sys.modules):
            if name == "comfy_nodes" or name.startswith("comfy_nodes."):
                del sys.modules[name]
        import comfy_nodes.yukari_finalize as reimported
        self.assertIn("YukariDeliver", reimported.NODE_CLASS_MAPPINGS)


class NodeRunTest(unittest.TestCase):
    def setUp(self):
        self._torch_patch = mock.patch.dict(
            sys.modules, {"torch": FakeTorch()})
        self._torch_patch.start()
        self.addCleanup(self._torch_patch.stop)

    def test_deliver_wiring_returns_an_image_and_a_tag(self):
        node = nodes.YukariDeliver()
        image, tag, *_ = node.run(
            image_tensor(swatch()), mask_tensor(matte_array()), keep_scene=False)
        self.assertEqual(image.array.shape[0], 1)
        self.assertEqual(image.array.shape[-1], 3)
        self.assertTrue(tag.startswith("clean-"))

    def test_matting_wiring_returns_the_alpha(self):
        seen = {}

        def predict(rgb, known):
            seen["known"] = known
            return np.full(known.shape, 0.5)

        node = nodes.YukariMatting()
        with mock.patch.object(nodes.vitmatte, "predict", predict):
            (alpha,) = node.run(image_tensor(swatch()), mask_tensor(matte_array()))
        self.assertEqual(alpha.array.shape, (1, 64, 64))
        self.assertEqual(set(np.unique(seen["known"])), {0, 128, 255})
        result = (alpha.array[0] * 255.0).round()
        self.assertEqual(result[32, 32], 255)
        self.assertEqual(result[0, 0], 0)
        self.assertTrue(((result > 0) & (result < 255)).any())

    def test_foreground_wiring_estimates_the_colour_under_the_alpha(self):
        seen = {}

        def foreground(rgb, alpha):
            seen["alpha"] = alpha
            return rgb

        node = nodes.YukariForeground()
        pixels = swatch()
        with mock.patch.object(nodes.vitmatte, "foreground", foreground):
            (image,) = node.run(image_tensor(pixels), mask_tensor(matte_array()))
        self.assertEqual(image.array.shape, (1, 64, 64, 3))
        self.assertEqual(seen["alpha"].shape, (64, 64))
        np.testing.assert_allclose(
            (image.array[0] * 255.0).round(), pixels, atol=1)

    def test_see_through_at_gain_one_returns_the_alpha_unchanged(self):
        alpha = matte_array()
        (out,) = nodes.YukariSeeThrough().run(
            image_tensor(swatch()), mask_tensor(alpha), gain=1.0)
        np.testing.assert_allclose((out.array[0] * 255.0).round(), alpha, atol=1)

    def test_deliver_matted_takes_the_alpha_as_coverage(self):
        node = nodes.YukariDeliver()
        alpha = matte_array()
        alpha[16:48, 47] = 128
        image, tag, *_ = node.run(
            image_tensor(swatch()), mask_tensor(alpha), keep_scene=False,
            transparent=True, matted=True)
        self.assertEqual(image.array.shape, (1, 64, 64, 4))
        self.assertIn("-matted", tag)
        self.assertNotIn("-key", tag)

    def test_deliver_keep_scene_returns_the_redraw_uncut(self):
        node = nodes.YukariDeliver()
        pixels = swatch()
        image, tag, *_ = node.run(
            image_tensor(pixels), mask_tensor(matte_array()), keep_scene=True)
        self.assertEqual(tag, "scene")
        np.testing.assert_allclose(
            (image.array[0] * 255.0).round(), pixels, atol=1)

    def test_deliver_transparent_returns_an_rgba_cutout(self):
        node = nodes.YukariDeliver()
        image, tag, *_ = node.run(
            image_tensor(swatch()), mask_tensor(matte_array()),
            keep_scene=False, transparent=True)
        self.assertEqual(image.array.shape, (1, 64, 64, 4))
        self.assertEqual(tag, "transparent-o1+1-cut0.5")

    def test_deliver_keep_scene_wins_over_transparent(self):
        node = nodes.YukariDeliver()
        pixels = swatch()
        image, tag, *_ = node.run(
            image_tensor(pixels), mask_tensor(matte_array()),
            keep_scene=True, transparent=True)
        self.assertEqual(image.array.shape[-1], 3)
        self.assertEqual(tag, "scene")

    def test_deliver_run_without_transparent_kwarg_still_works(self):
        node = nodes.YukariDeliver()
        image, tag, *_ = node.run(
            image_tensor(swatch()), mask_tensor(matte_array()), keep_scene=False)
        self.assertEqual(image.array.shape[-1], 3)
        self.assertTrue(tag.startswith("clean-"))

    def test_deliver_stroke_light_returns_an_image_and_a_light_tag(self):
        node = nodes.YukariDeliver()
        image, tag, *_ = node.run(
            image_tensor(swatch()), mask_tensor(matte_array()),
            keep_scene=False, transparent=True, stroke_light="ne")
        self.assertEqual(image.array.shape, (1, 64, 64, 4))
        self.assertTrue(tag.endswith("-light-ne"))

    def test_deliver_default_stroke_light_yields_the_old_tag(self):
        node = nodes.YukariDeliver()
        image, tag, *_ = node.run(
            image_tensor(swatch()), mask_tensor(matte_array()),
            keep_scene=False, transparent=True)
        self.assertEqual(tag, "transparent-o1+1-cut0.5")

    def test_deliver_returns_the_layers_beside_the_picture(self):
        node = nodes.YukariDeliver()
        image, _, figure, outline, backdrop = node.run(
            image_tensor(swatch()), mask_tensor(matte_array()), keep_scene=False,
            matted=True, outlines='[{"color": "#885b80", "width": 5}]')
        self.assertEqual(figure.array.shape, (1, 64, 64, 4))
        self.assertEqual(outline.array.shape, (1, 64, 64, 4))
        self.assertEqual(backdrop.array.shape, (1, 64, 64, 3))
        np.testing.assert_allclose(figure.array[0, ..., 3], matte_array() / 255.0)
        self.assertGreater(outline.array[0, 32, 50, 3], 0.9)

    def test_deliver_empty_outlines_leave_an_empty_outline_layer(self):
        node = nodes.YukariDeliver()
        _, tag, _, outline, _ = node.run(
            image_tensor(swatch()), mask_tensor(matte_array()), keep_scene=False,
            transparent=True, matted=True, outlines="[]")
        self.assertEqual(outline.array[0, ..., 3].max(), 0)
        self.assertIn("nooutline", tag)

    def test_depth_of_field_takes_loadimage_masks_as_inverted_alpha(self):
        node = nodes.YukariDepthOfField()
        figure_alpha = matte_array()
        outline_alpha = np.zeros((64, 64), np.uint8)
        depth = np.tile(np.linspace(0, 255, 64, dtype=np.uint8)[:, None], (1, 64))
        backdrop = np.full((64, 64, 3), 200, np.uint8)
        (image,) = node.run(
            image_tensor(swatch()), mask_tensor(255 - figure_alpha),
            image_tensor(np.zeros((64, 64, 3), np.uint8)), mask_tensor(255 - outline_alpha),
            image_tensor(np.dstack([depth] * 3)), focus_x=0.5, focus_y=0.5,
            f_number=2.8, blur_figure=False, blur_outline=False,
            blur_backdrop=False, backdrop=image_tensor(backdrop))
        result = (image.array[0] * 255.0).round()
        np.testing.assert_allclose(result[32, 32], swatch()[32, 32], atol=1)
        np.testing.assert_allclose(result[2, 2], (200, 200, 200), atol=1)
        (cut,) = node.run(
            image_tensor(swatch()), mask_tensor(255 - figure_alpha),
            image_tensor(np.zeros((64, 64, 3), np.uint8)), mask_tensor(255 - outline_alpha),
            image_tensor(np.dstack([depth] * 3)), focus_x=0.5, focus_y=0.5,
            f_number=2.8, blur_figure=True, blur_outline=True, blur_backdrop=True)
        self.assertEqual(cut.array.shape, (1, 64, 64, 4))
        self.assertEqual(cut.array[0, 0, 0, 3], 0)

    def test_lens_keeps_an_opaque_picture_rgb_and_a_cut_one_rgba(self):
        node = nodes.YukariLens()
        (image,) = node.run(image_tensor(swatch()), mask_tensor(np.zeros((64, 64), np.uint8)),
                            center_x=0.5, center_y=0.5, strength=0.7)
        self.assertEqual(image.array.shape, (1, 64, 64, 3))
        (cut,) = node.run(image_tensor(swatch()), mask_tensor(255 - matte_array()),
                          center_x=0.5, center_y=0.5, strength=-0.7)
        self.assertEqual(cut.array.shape, (1, 64, 64, 4))
        (sphere,) = node.run(image_tensor(swatch()), mask_tensor(np.zeros((64, 64), np.uint8)),
                             center_x=0.5, center_y=0.5, strength=0.7,
                             shape="sphere", radius=0.3)
        self.assertEqual(sphere.array.shape, (1, 64, 64, 3))

    def test_photo_stage_returns_the_photo_size_in_rgb(self):
        node = nodes.YukariPhotoStage()
        photo = np.full((48, 80, 3), 120, np.uint8)
        depth = np.tile(np.linspace(0, 255, 48, dtype=np.uint8)[:, None], (1, 80))
        (image,) = node.run(
            image_tensor(photo), image_tensor(np.dstack([depth] * 3)),
            image_tensor(swatch()), mask_tensor(255 - matte_array()),
            image_tensor(np.zeros((64, 64, 3), np.uint8)),
            mask_tensor(np.full((64, 64), 255, np.uint8)),
            place_x=0.5, place_y=0.95, height=0.6, stand="base", stand_x=0.5,
            stand_y=0.5, nearer=0.0, focus="subject", focus_x=0.5, focus_y=0.5,
            f_number=2.8)
        self.assertEqual(image.array.shape, (1, 48, 80, 3))

    def test_stage_sets_layers_on_a_pattern_or_a_photo(self):
        node = nodes.YukariStage()
        figure, figure_mask = image_tensor(swatch()), mask_tensor(255 - matte_array())
        outline = image_tensor(np.zeros((64, 64, 3), np.uint8))
        outline_mask = mask_tensor(np.full((64, 64), 255, np.uint8))
        image, staged_figure, staged_outline, backdrop = node.run(
            figure, figure_mask, outline, outline_mask, mode="pattern",
            backdrop="stripes", shadow_from="nw")
        self.assertEqual(image.array.shape, (1, 64, 64, 3))
        self.assertEqual(staged_figure.array.shape, (1, 64, 64, 4))
        self.assertEqual(staged_outline.array.shape, (1, 64, 64, 4))
        self.assertEqual(backdrop.array.shape, (1, 64, 64, 3))
        image, *_ = node.run(figure, figure_mask, outline, outline_mask, mode="pattern",
                             backdrop="stripes", light_scene="sunset", light_from="ne")
        self.assertEqual(image.array.shape, (1, 64, 64, 3))
        image, staged_figure, *_ = node.run(
            figure, figure_mask, outline, outline_mask, mode="pattern", backdrop="dots",
            frame=json.dumps({"box": [0.25, 0.25, 1, 1], "canvas": [0, 0, 1.5, 1],
                              "sides": {"top": "out", "right": "cut", "bottom": "cut",
                                        "left": "cut"},
                              "line": {"color": "#885b80", "width": 1.0},
                              "paper": "#ffffff", "strokes": []}))
        self.assertEqual(image.array.shape, (1, 64, 96, 3))
        self.assertEqual(staged_figure.array.shape, (1, 64, 96, 4))
        photo = image_tensor(np.full((48, 80, 3), 120, np.uint8))
        image, staged_figure, _, backdrop = node.run(
            figure, figure_mask, outline, outline_mask, mode="photo", photo=photo,
            place_x=0.5, place_y=1.0, height=0.8)
        self.assertEqual(image.array.shape, (1, 48, 80, 3))
        self.assertEqual(staged_figure.array.shape, (1, 48, 80, 4))
        self.assertEqual(backdrop.array.shape, (1, 48, 80, 3))

    def test_saturation_returns_the_picture_and_the_compressed_layers(self):
        node = nodes.YukariSaturation()
        figure, figure_mask = image_tensor(swatch()), mask_tensor(255 - matte_array())
        outline = image_tensor(np.zeros((64, 64, 3), np.uint8))
        outline_mask = mask_tensor(np.full((64, 64), 255, np.uint8))
        backdrop = image_tensor(np.full((64, 64, 3), 200, np.uint8))
        image, saturated_figure, saturated_outline = node.run(
            figure, figure_mask, outline, outline_mask, strength=1.0, backdrop=backdrop)
        self.assertEqual(image.array.shape, (1, 64, 64, 3))
        self.assertEqual(saturated_figure.array.shape, (1, 64, 64, 4))
        self.assertEqual(saturated_outline.array.shape, (1, 64, 64, 4))
        image, *_ = node.run(figure, figure_mask, outline, outline_mask, strength=0.5)
        self.assertEqual(image.array.shape, (1, 64, 64, 4))

    def test_relight_returns_the_picture_and_its_layers(self):
        node = nodes.YukariRelight()
        image, figure, outline, backdrop = node.run(
            image_tensor(swatch()), mask_tensor(255 - matte_array()),
            image_tensor(np.zeros((64, 64, 3), np.uint8)),
            mask_tensor(np.full((64, 64), 255, np.uint8)),
            image_tensor(np.full((64, 64, 3), 40, np.uint8)), direction="nw", strength=0.8)
        self.assertEqual(image.array.shape, (1, 64, 64, 3))
        self.assertEqual(figure.array.shape, (1, 64, 64, 4))
        self.assertEqual(outline.array.shape, (1, 64, 64, 4))
        self.assertEqual(backdrop.array.shape, (1, 64, 64, 3))

    def test_relight_without_a_backdrop_returns_a_transparent_picture(self):
        image, figure, outline, _ = nodes.YukariRelight().run(
            image_tensor(swatch()), mask_tensor(255 - matte_array()),
            image_tensor(np.zeros((64, 64, 3), np.uint8)),
            mask_tensor(np.full((64, 64), 255, np.uint8)), direction="nw", strength=0.8)
        self.assertEqual(image.array.shape, (1, 64, 64, 4))
        self.assertEqual(figure.array.shape, (1, 64, 64, 4))
        self.assertNotIn("backdrop", nodes.YukariRelight.INPUT_TYPES()["required"])

    def test_photo_depth_of_field_returns_the_backdrop_size_in_rgb(self):
        node = nodes.YukariPhotoDepthOfField()
        depth = np.tile(np.linspace(0, 255, 48, dtype=np.uint8)[:, None], (1, 80))
        figure = np.zeros((48, 80, 3), np.uint8)
        figure_alpha = np.full((48, 80), 255, np.uint8)
        figure_alpha[10:40, 30:50] = 0
        (image,) = node.run(
            image_tensor(np.full((48, 80, 3), 120, np.uint8)),
            image_tensor(np.dstack([depth] * 3)), image_tensor(figure),
            mask_tensor(figure_alpha), image_tensor(figure),
            mask_tensor(np.full((48, 80), 255, np.uint8)),
            place_x=0.5, place_y=40 / 48, height=30 / 48, stand="base", stand_x=0.5,
            stand_y=0.5, nearer=0.0, focus="subject", focus_x=0.5, focus_y=0.5,
            f_number=2.8, tilt=0.3, bokeh=0.5)
        self.assertEqual(image.array.shape, (1, 48, 80, 3))

    def test_deliver_backdrop_stripes_returns_rgb_and_a_bg_tag(self):
        node = nodes.YukariDeliver()
        image, tag, *_ = node.run(
            image_tensor(swatch()), mask_tensor(matte_array()),
            keep_scene=False, backdrop="stripes")
        self.assertEqual(image.array.shape[-1], 3)
        self.assertTrue(tag.endswith("-bg-stripes-cut0.5"))

    def test_deliver_default_backdrop_yields_the_old_tag(self):
        node = nodes.YukariDeliver()
        image, tag, *_ = node.run(
            image_tensor(swatch()), mask_tensor(matte_array()), keep_scene=False)
        self.assertTrue(tag.startswith("clean-"))
        self.assertNotIn("-bg-", tag)


if __name__ == "__main__":
    unittest.main()
