"""ComfyUI nodes wrapping this repo's imaging functions unchanged.

Each node's body is: tensors in, PNG bytes out, call the existing function,
PNG bytes back to tensors. The imaging logic itself lives in
``comfyui_recipes.infrastructure.imaging`` and is not duplicated here.
"""

from __future__ import annotations

import json

import numpy as np

from comfyui_recipes.infrastructure.imaging import (
    delivery, depth_blur, lens, matting, palette, photo_stage, recolor, viewfinder)
from comfyui_recipes.infrastructure.imaging import light as lighting

from . import bridge, vitmatte


class YukariRepinSkin:
    CATEGORY = "yukari"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "source": ("IMAGE",),
        }}

    def run(self, image, source):
        data, report = palette.repin_skin_png(
            bridge.image_to_png(source), bridge.image_to_png(image))
        return (bridge.png_to_image(data), "\n".join(report))


class YukariRepin:
    CATEGORY = "yukari"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "keep_legwear": ("BOOLEAN", {"default": False}),
            "keep_legwear_cut": ("FLOAT", {"default": 0.62, "min": 0.0,
                                            "max": 1.0, "step": 0.01}),
        }}

    def run(self, image, keep_legwear, keep_legwear_cut):
        data, report = palette.repin_png(
            bridge.image_to_png(image),
            keep_legwear=keep_legwear_cut if keep_legwear else None)
        return (bridge.png_to_image(data), "\n".join(report))


class YukariRecolor:
    CATEGORY = "yukari"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def run(self, image):
        data, report = recolor.recolor_png(bridge.image_to_png(image))
        return (bridge.png_to_image(data), "\n".join(report))


class YukariMatting:
    CATEGORY = "yukari"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "matte": ("MASK",)}}

    def run(self, image, matte):
        try:
            alpha = matting.alpha_png(
                bridge.image_to_png(image), bridge.mask_to_png(matte),
                vitmatte.predict)
        finally:
            vitmatte.release()
        return (bridge.png_to_mask(alpha),)


class YukariForeground:
    CATEGORY = "yukari"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "alpha": ("MASK",)}}

    def run(self, image, alpha):
        data = matting.foreground_png(
            bridge.image_to_png(image), bridge.mask_to_png(alpha),
            vitmatte.foreground)
        return (bridge.png_to_image(data),)


class YukariDeliver:
    CATEGORY = "yukari"
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "tag", "figure", "outline", "backdrop")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "matte": ("MASK",),
            "keep_scene": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "transparent": ("BOOLEAN", {"default": False}),
            "stroke_light": ("STRING", {"default": ""}),
            "backdrop": ("STRING", {"default": ""}),
            "light_scene": ("STRING", {"default": ""}),
            "light_from": ("STRING", {"default": ""}),
            "matted": ("BOOLEAN", {"default": False}),
            "outlines": ("STRING", {"default": ""}),
        }}

    def run(self, image, matte, keep_scene, transparent=False, stroke_light="",
            backdrop="", light_scene="", light_from="", matted=False, outlines=""):
        picture, figure, outline, backdrop_png, tag = delivery.deliver_pngs(
            bridge.image_to_png(image), bridge.mask_to_png(matte),
            keep_scene=keep_scene, transparent=transparent,
            light=stroke_light or None, backdrop=backdrop or None,
            scene=light_scene or None, light_from=light_from or None,
            matted=matted, outlines=json.loads(outlines) if outlines else None)
        mode = "RGBA" if (transparent and not keep_scene) else "RGB"
        # A transparent delivery has no backdrop layer; this output is never saved.
        backdrop_image = bridge.png_to_image(
            backdrop_png or bridge.array_to_png(np.zeros((1, 1, 3), np.uint8), "RGB"))
        return (bridge.png_to_image(picture, mode), tag,
                bridge.png_to_image(figure, "RGBA"),
                bridge.png_to_image(outline, "RGBA"), backdrop_image)


class YukariDepthOfField:
    CATEGORY = "yukari"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "figure": ("IMAGE",),
            "figure_mask": ("MASK",),
            "outline": ("IMAGE",),
            "outline_mask": ("MASK",),
            "depth": ("IMAGE",),
            "focus_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                  "step": 0.001}),
            "focus_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                  "step": 0.001}),
            "f_number": ("FLOAT", {"default": 2.8, "min": 1.4, "max": 22.0,
                                   "step": 0.1}),
            "blur_figure": ("BOOLEAN", {"default": True}),
            "blur_outline": ("BOOLEAN", {"default": True}),
            "blur_backdrop": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "backdrop": ("IMAGE",),
        }}

    def run(self, figure, figure_mask, outline, outline_mask, depth, focus_x,
            focus_y, f_number, blur_figure, blur_outline, blur_backdrop,
            backdrop=None):
        scope = {"figure": blur_figure, "outline": blur_outline,
                 "backdrop": blur_backdrop}
        data = depth_blur.blur_layers_png(
            bridge.loaded_rgba_png(figure, figure_mask),
            bridge.loaded_rgba_png(outline, outline_mask),
            bridge.image_to_png(backdrop) if backdrop is not None else None,
            bridge.image_to_png(depth), focus_x, focus_y, f_number, scope)
        return (bridge.png_to_image(data, "RGB" if backdrop is not None else "RGBA"),)


class YukariLight:
    CATEGORY = "yukari"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "depth": ("IMAGE",),
            "matte": ("MASK",),
            "direction": ("STRING", {"default": "nw"}),
            "scene": ("STRING", {"default": "sunset"}),
        }}

    def run(self, image, depth, matte, direction, scene):
        data = lighting.underpaint_png(
            bridge.image_to_png(image), bridge.image_to_png(depth),
            bridge.mask_to_png(matte), direction, scene)
        return (bridge.png_to_image(data),)


class YukariViewfinder:
    CATEGORY = "yukari"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "focus_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                  "step": 0.001}),
            "focus_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                  "step": 0.001}),
            "f_number": ("FLOAT", {"default": 2.8, "min": 0.7, "max": 22.0,
                                   "step": 0.1}),
        }}

    def run(self, image, focus_x, focus_y, f_number):
        data = viewfinder.viewfinder_png(
            bridge.image_to_png(image), (focus_x, focus_y), f_number)
        return (bridge.png_to_image(data, "RGBA" if image.shape[-1] == 4 else "RGB"),)


class YukariLens:
    CATEGORY = "yukari"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                   "step": 0.001}),
            "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                   "step": 0.001}),
            "strength": ("FLOAT", {"default": 0.7, "min": -1.0, "max": 1.0,
                                   "step": 0.01}),
        }, "optional": {
            "shape": (list(lens.SHAPES), {"default": "fisheye"}),
            "radius": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0,
                                 "step": 0.01}),
        }}

    def run(self, image, mask, center_x, center_y, strength, shape="fisheye",
            radius=0.3):
        data, mode = lens.lens_png(
            bridge.loaded_rgba_png(image, mask), (center_x, center_y), strength,
            shape, radius if shape == "sphere" else None)
        return (bridge.png_to_image(data, mode),)


class YukariPhotoStage:
    CATEGORY = "yukari"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        def fraction(default):
            return ("FLOAT", {"default": default, "min": 0.0, "max": 1.0, "step": 0.001})

        return {"required": {
            "photo": ("IMAGE",),
            "depth": ("IMAGE",),
            "figure": ("IMAGE",),
            "figure_mask": ("MASK",),
            "outline": ("IMAGE",),
            "outline_mask": ("MASK",),
            "place_x": fraction(0.5),
            "place_y": fraction(0.95),
            "height": ("FLOAT", {"default": 0.7, "min": 0.05, "max": 1.5, "step": 0.01}),
            "stand": (["base", "point"], {"default": "base"}),
            "stand_x": fraction(0.5),
            "stand_y": fraction(0.5),
            "nearer": fraction(0.0),
            "focus": (["subject", "point"], {"default": "subject"}),
            "focus_x": fraction(0.5),
            "focus_y": fraction(0.5),
            "f_number": ("FLOAT", {"default": 2.8, "min": 1.4, "max": 22.0, "step": 0.1}),
        }, "optional": {
            "tilt": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
        }}

    def run(self, photo, depth, figure, figure_mask, outline, outline_mask,
            place_x, place_y, height, stand, stand_x, stand_y, nearer, focus,
            focus_x, focus_y, f_number, tilt=0.0):
        data = photo_stage.stage_png(
            bridge.image_to_png(photo), bridge.image_to_png(depth),
            bridge.loaded_rgba_png(figure, figure_mask),
            bridge.loaded_rgba_png(outline, outline_mask),
            place=(place_x, place_y), height=height,
            stand_at=(stand_x, stand_y) if stand == "point" else None,
            nearer=nearer,
            focus=(focus_x, focus_y) if focus == "point" else photo_stage.SUBJECT,
            f_number=f_number, tilt=tilt)
        return (bridge.png_to_image(data),)


NODE_CLASS_MAPPINGS = {
    "YukariRepinSkin": YukariRepinSkin,
    "YukariRepin": YukariRepin,
    "YukariRecolor": YukariRecolor,
    "YukariMatting": YukariMatting,
    "YukariForeground": YukariForeground,
    "YukariDeliver": YukariDeliver,
    "YukariDepthOfField": YukariDepthOfField,
    "YukariLight": YukariLight,
    "YukariViewfinder": YukariViewfinder,
    "YukariLens": YukariLens,
    "YukariPhotoStage": YukariPhotoStage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YukariRepinSkin": "Yukari Repin Skin",
    "YukariRepin": "Yukari Repin",
    "YukariRecolor": "Yukari Recolor",
    "YukariMatting": "Yukari Matting",
    "YukariForeground": "Yukari Foreground",
    "YukariDeliver": "Yukari Deliver",
    "YukariDepthOfField": "Yukari Depth of Field",
    "YukariLight": "Yukari Light",
    "YukariViewfinder": "Yukari Viewfinder",
    "YukariLens": "Yukari Lens",
    "YukariPhotoStage": "Yukari Photo Stage",
}
