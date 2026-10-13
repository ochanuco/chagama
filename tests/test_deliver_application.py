"""Tests for the deliver graph and use case.

All collaborators are fakes: nothing here opens a socket or talks to ComfyUI.
The fake ComfyUI answers `wait_for` with the SaveImage prefixes of the graph
it was handed, so what the use case uploads follows what the graph saves.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from comfyui_recipes.application.deliver import (
    DeliverServices,
    current_cut,
    deliver,
)
from comfyui_recipes.application.ingest import classify_deliver_outputs
from comfyui_recipes.domain.yukari import delivery_style
from comfyui_recipes.domain.yukari.delivery_style import Light
from comfyui_recipes.infrastructure.comfyui.deliver_graph import deliver_graph
from comfyui_recipes.infrastructure.comfyui.refinement_graph import DEPTH_NODE

LAYERS = ["dlv-g-layer-figure", "dlv-g-layer-outline", "dlv-g-layer-backdrop"]
OUTLINES = [{"color": "#ffffff", "width": 0.8}, {"color": "#885b80", "width": 1.04}]
GRAPH_KWARGS = dict(
    skin=False, repin=True, recolor=False, keep_legwear=None, keep_scene=False,
    transparent=False, backdrop="dots", stroke_light="n", outlines=OUTLINES,
    deliver_size=None, canvas=(832, 1664), light_scene=None, light_from=None)


def classes(graph: dict) -> list[str]:
    return [node["class_type"] for node in graph.values()]


def nodes_of(graph: dict, class_type: str) -> list[tuple[str, dict]]:
    return [(key, node) for key, node in graph.items()
            if node["class_type"] == class_type]


def saved_prefixes(graph: dict) -> list[str]:
    return [node["inputs"]["filename_prefix"] for _, node in
            nodes_of(graph, "SaveImage")]


def build(**overrides) -> dict:
    kwargs = {**GRAPH_KWARGS, **overrides}
    return deliver_graph("src.png", delivery_style.MATTE_MODEL, "dlv-g", **kwargs)


class DeliverGraphTest(unittest.TestCase):
    def test_a_fresh_cut_mattes_the_unrepinned_source_and_saves_the_alpha(self):
        graph = build()
        [(load_id, _)] = nodes_of(graph, "LoadImage")
        [(birefnet_id, _)] = nodes_of(graph, "BiRefNetRMBG")
        [(matting_id, matting)] = nodes_of(graph, "YukariMatting")
        self.assertEqual(matting["inputs"],
                         {"image": [load_id, 0], "matte": [birefnet_id, 1]})
        [(to_image_id, to_image)] = nodes_of(graph, "MaskToImage")
        self.assertEqual(to_image["inputs"]["mask"], [matting_id, 0])
        self.assertIn("dlv-g-alpha", saved_prefixes(graph))
        self.assertNotIn(DEPTH_NODE, classes(graph))
        self.assertFalse(any("-depth" in prefix for prefix in saved_prefixes(graph)))
        [(repin_id, repin)] = nodes_of(graph, "YukariRepin")
        self.assertEqual(repin["inputs"]["image"], [load_id, 0])
        [(foreground_id, foreground)] = nodes_of(graph, "YukariForeground")
        self.assertEqual(foreground["inputs"],
                         {"image": [repin_id, 0], "alpha": [matting_id, 0]})
        [(_, delivered)] = nodes_of(graph, "YukariDeliver")
        self.assertEqual(delivered["inputs"]["image"], [foreground_id, 0])
        self.assertEqual(delivered["inputs"]["matte"], [matting_id, 0])
        self.assertIs(delivered["inputs"]["matted"], True)
        self.assertEqual(saved_prefixes(graph),
                         ["dlv-g-alpha", "dlv-g-delivered", *LAYERS])
        self.assertEqual(json.loads(delivered["inputs"]["outlines"]), OUTLINES)
        self.assertEqual(to_image_id, graph[next(
            key for key, node in nodes_of(graph, "SaveImage")
            if node["inputs"]["filename_prefix"] == "dlv-g-alpha")]["inputs"]["images"][0])

    def test_a_reused_alpha_is_loaded_and_nothing_is_cut(self):
        graph = build(alpha_image="alpha.png")
        self.assertFalse({"BiRefNetRMBG", "YukariMatting", DEPTH_NODE,
                          "MaskToImage"} & set(classes(graph)))
        loads = {node["inputs"]["image"]: key
                 for key, node in nodes_of(graph, "LoadImage")}
        [(to_mask_id, to_mask)] = nodes_of(graph, "ImageToMask")
        self.assertEqual(to_mask["inputs"],
                         {"image": [loads["alpha.png"], 0], "channel": "red"})
        [(_, foreground)] = nodes_of(graph, "YukariForeground")
        self.assertEqual(foreground["inputs"]["alpha"], [to_mask_id, 0])
        self.assertEqual(saved_prefixes(graph), ["dlv-g-delivered", *LAYERS])

    def test_the_layers_are_the_deliver_nodes_own_outputs(self):
        graph = build(alpha_image="alpha.png")
        [(deliver_id, _)] = nodes_of(graph, "YukariDeliver")
        saved = {node["inputs"]["filename_prefix"]: node["inputs"]["images"]
                 for _, node in nodes_of(graph, "SaveImage")}
        self.assertEqual(saved, {
            "dlv-g-delivered": [deliver_id, 0],
            "dlv-g-layer-figure": [deliver_id, 2],
            "dlv-g-layer-outline": [deliver_id, 3],
            "dlv-g-layer-backdrop": [deliver_id, 4]})

    def test_a_transparent_delivery_saves_no_backdrop_layer(self):
        graph = build(alpha_image="alpha.png", transparent=True, backdrop=None)
        self.assertEqual(saved_prefixes(graph),
                         ["dlv-g-delivered", *LAYERS[:2]])

    def test_repin_skin_recolor_follow_the_delivery_rules(self):
        graph = build(skin=True, recolor=True, repin=True)
        [(load_id, _)] = nodes_of(graph, "LoadImage")
        [(skin_id, skin)] = nodes_of(graph, "YukariRepinSkin")
        self.assertEqual(skin["inputs"], {"image": [load_id, 0], "source": [load_id, 0]})
        [(_, recolor)] = nodes_of(graph, "YukariRecolor")
        self.assertEqual(recolor["inputs"]["image"], [skin_id, 0])
        self.assertNotIn("YukariRepin", classes(graph))

    def test_see_through_scales_the_tail_alpha_not_the_saved_one(self):
        graph = build(see_through=3.0)
        [(load_id, _)] = nodes_of(graph, "LoadImage")
        [(matting_id, _)] = nodes_of(graph, "YukariMatting")
        [(_, foreground)] = nodes_of(graph, "YukariForeground")
        self.assertEqual(foreground["inputs"]["alpha"], [matting_id, 0])
        [(see_id, see)] = nodes_of(graph, "YukariSeeThrough")
        self.assertEqual(see["inputs"], {
            "image": [load_id, 0], "alpha": [matting_id, 0], "gain": 3.0})
        [(_, delivered)] = nodes_of(graph, "YukariDeliver")
        self.assertEqual(delivered["inputs"]["matte"], [see_id, 0])
        self.assertEqual(delivered["inputs"]["silhouette"], [matting_id, 0])
        [(_, to_image)] = nodes_of(graph, "MaskToImage")
        self.assertEqual(to_image["inputs"]["mask"], [matting_id, 0])

    def test_see_through_one_adds_no_node(self):
        self.assertNotIn("YukariSeeThrough", classes(build(see_through=1.0)))

    def test_keep_scene_skips_the_foreground_and_composites_with_the_alpha(self):
        graph = build(keep_scene=True)
        self.assertNotIn("YukariForeground", classes(graph))
        [(matting_id, _)] = nodes_of(graph, "YukariMatting")
        [(_, delivered)] = nodes_of(graph, "YukariDeliver")
        self.assertIs(delivered["inputs"]["keep_scene"], True)
        self.assertIs(delivered["inputs"]["matted"], False)
        self.assertEqual(delivered["inputs"]["matte"], [matting_id, 0])
        self.assertIn("dlv-g-alpha", saved_prefixes(graph))
        self.assertIn("dlv-g-layer-backdrop", saved_prefixes(graph))

    def test_deliver_size_scales_the_delivered_picture_and_every_layer(self):
        graph = build(deliver_size=832)
        scales = nodes_of(graph, "ImageScale")
        self.assertEqual(len(scales), 4)
        for _, scale in scales:
            self.assertEqual((scale["inputs"]["width"], scale["inputs"]["height"]),
                             (416, 832))
        scaled = {key for key, _ in scales}
        for _, save in nodes_of(graph, "SaveImage"):
            if save["inputs"]["filename_prefix"] != "dlv-g-alpha":
                self.assertIn(save["inputs"]["images"][0], scaled)

    def test_a_bad_backdrop_or_stroke_light_is_refused(self):
        with self.assertRaisesRegex(ValueError, "backdrop"):
            build(backdrop="plaid")
        for bad in ("north", "none"):
            with self.assertRaisesRegex(ValueError, "stroke_light"):
                build(stroke_light=bad)

    def test_outputs_are_classified_by_role(self):
        outputs = [{"filename": f"dlv-g{suffix}_00001_.png"} for suffix in
                   ("-alpha", "-delivered", "-layer-figure", "-layer-outline",
                    "-layer-backdrop")]
        roles = classify_deliver_outputs(outputs)
        self.assertEqual({role: [out["filename"] for out in found]
                          for role, found in roles.items()}, {
            "alpha": ["dlv-g-alpha_00001_.png"],
            "delivered": ["dlv-g-delivered_00001_.png"],
            "layer-figure": ["dlv-g-layer-figure_00001_.png"],
            "layer-outline": ["dlv-g-layer-outline_00001_.png"],
            "layer-backdrop": ["dlv-g-layer-backdrop_00001_.png"]})


class ManagementFake:
    def __init__(self, *, parameters=None, request_kind="generate", assets=None,
                 graph=None, records=None):
        self.records = records or {}
        self.calls = []
        self.assets = dict(assets or {})
        self.context = {"request": {"id": "source-request", "recipe": "yukari",
                                    "kind": request_kind,
                                    "parameters": parameters or {}},
                        "generations": []}
        self.graph = graph

    def request(self, method, path, payload=None, multipart=None):
        self.calls.append((method, path, payload, multipart))
        if path.endswith("/context"):
            return self.context
        if (method == "GET" and path.startswith("/api/v1/generations/")
                and path.count("/") == 4):
            if path.rsplit("/", 1)[1] in self.records:
                return self.records[path.rsplit("/", 1)[1]]
            return {"id": path.rsplit("/", 1)[1],
                    "comfy_job": {"graph": self.graph} if self.graph else None}
        if (method == "PUT" and path.endswith("/resolution")) or (
                method == "POST" and path == "/api/v1/requests"):
            return {"id": "request-id", "short_id": "req"}
        if method == "POST" and path.endswith("/jobs"):
            return {"id": "job-id"}
        if path.endswith("/generations"):
            index = sum(1 for call in self.calls if call[1].endswith("/generations"))
            return {"id": f"delivered-{index}", "short_id": "gen",
                    "canonical_url": f"https://example/g/{index}"}
        return {}

    def fetch_generation_image(self, generation_id):
        return b"picked"

    def list_assets(self, generation_id):
        return [{"role": role} for role in self.assets]

    def fetch_asset(self, generation_id, role):
        return self.assets.get(role)

    def asset_uploads(self):
        return [call for call in self.calls
                if call[0] == "POST" and call[1].endswith("/assets")]


class ComfyFake:
    def __init__(self):
        self.uploaded = []
        self.submitted = []

    def upload_image(self, name, data):
        self.uploaded.append((name, data))
        return f"uploaded-{name}"

    def submit(self, graph):
        self.submitted.append(graph)
        return "prompt-id"

    def knows(self, prompt_id):
        return True

    def wait_for(self, prompt_id):
        return [{"filename": f"{prefix}_00001_.png"}
                for prefix in saved_prefixes(self.submitted[-1])]

    def fetch(self, image):
        return b"bytes:" + image["filename"].encode()


class RecordingNotifier:
    def __init__(self):
        self.calls = []

    def send(self, *args):
        self.calls.append(args)


def services(directory, **overrides) -> DeliverServices:
    kwargs = dict(
        management=ManagementFake(), comfyui=ComfyFake(),
        graph_from_png=lambda data: None,
        image_size=lambda data: (832, 1664),
        git_metadata=lambda: {"commit": "commit", "dirty": False},
        notifier=RecordingNotifier(), output_root=Path(directory),
        emit=lambda message: None)
    kwargs.update(overrides)
    return DeliverServices(**kwargs)


def cut_asset(**overrides) -> bytes:
    return json.dumps({**current_cut(), **overrides}).encode()


def roles_uploaded(management) -> list[str]:
    return [call[3][0]["role"] for call in management.asset_uploads()]


def source_uploads(management) -> list:
    return [call for call in management.asset_uploads()
            if call[1] == "/api/v1/generations/src/assets"]


class DeliverUseCaseTest(unittest.TestCase):
    def run_deliver(self, management=None, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, **({"management": management}
                                         if management else {}))
            result = deliver("src", svc, **kwargs)
            return svc, result

    def test_a_first_delivery_cuts_and_attaches_the_alpha_to_the_source(self):
        svc, result = self.run_deliver()
        management = svc.management
        graph = svc.comfyui.submitted[0]
        self.assertIn("YukariMatting", classes(graph))
        self.assertNotIn(DEPTH_NODE, classes(graph))
        source_uploads = [call for call in management.asset_uploads()
                          if call[1] == "/api/v1/generations/src/assets"]
        self.assertEqual([call[3][0]["role"] for call in source_uploads],
                         ["alpha", "cut"])
        self.assertEqual(source_uploads[0][3][3], b"bytes:dlv-src-alpha_00001_.png")
        cut_multipart = source_uploads[1][3]
        self.assertEqual(cut_multipart[4], "application/json")
        self.assertEqual(json.loads(cut_multipart[3]), {"alpha": current_cut()["alpha"]})
        self.assertEqual(result["generation_ids"], ["delivered-1"])

    def test_the_layers_are_attached_to_the_delivered_generation(self):
        svc, _ = self.run_deliver(key_prefix="request:r1", request_id="r1")
        uploads = [call for call in svc.management.asset_uploads()
                   if call[1] == "/api/v1/generations/delivered-1/assets"]
        self.assertEqual([call[3][0]["role"] for call in uploads],
                         ["layer-figure", "layer-outline", "layer-backdrop"])
        self.assertEqual([call[3][3] for call in uploads], [
            b"bytes:dlv-src-layer-figure_00001_.png",
            b"bytes:dlv-src-layer-outline_00001_.png",
            b"bytes:dlv-src-layer-backdrop_00001_.png"])
        self.assertEqual(uploads[0][3][0]["idempotency_key"],
                         "request:r1:job:0:asset:layer-figure")
        generation_post = next(index for index, call in enumerate(svc.management.calls)
                               if call[1].endswith("/generations"))
        first_layer = svc.management.calls.index(uploads[0])
        self.assertLess(generation_post, first_layer)

    def test_a_transparent_delivery_attaches_no_backdrop_layer(self):
        svc, _ = self.run_deliver(transparent=True, backdrop=None)
        roles = [call[3][0]["role"] for call in svc.management.asset_uploads()
                 if "delivered-1" in call[1]]
        self.assertEqual(roles, ["layer-figure", "layer-outline"])

    def test_a_missing_layer_output_fails_loudly(self):
        class NoOutline(ComfyFake):
            def wait_for(self, prompt_id):
                return [out for out in super().wait_for(prompt_id)
                        if "layer-outline" not in out["filename"]]

        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, comfyui=NoOutline())
            with self.assertRaisesRegex(SystemExit, "layer-outline"):
                deliver("src", svc)

    def test_no_mask_asset_is_attached_to_the_delivered_generation(self):
        svc, _ = self.run_deliver()
        self.assertNotIn("mask", roles_uploaded(svc.management))

    def test_matching_assets_are_reused_and_no_cut_is_uploaded_back(self):
        management = ManagementFake(assets={
            "alpha": b"alpha-png", "depth": b"depth-png", "cut": cut_asset()})
        svc, _ = self.run_deliver(management)
        graph = svc.comfyui.submitted[0]
        self.assertFalse({"BiRefNetRMBG", "YukariMatting", DEPTH_NODE}
                         & set(classes(graph)))
        self.assertEqual({name.split("-")[2] for name, _ in svc.comfyui.uploaded},
                         {"source", "alpha"})
        self.assertIn(b"alpha-png", [data for _, data in svc.comfyui.uploaded])
        self.assertFalse(any("/generations/src/" in call[1]
                             for call in management.asset_uploads()))

    def test_a_changed_cut_recipe_recomputes_only_what_changed(self):
        stale_alpha = {**current_cut()["alpha"], "trimap_px": 3}
        management = ManagementFake(assets={
            "alpha": b"old-alpha", "depth": b"depth-png",
            "cut": cut_asset(alpha=stale_alpha)})
        svc, _ = self.run_deliver(management)
        graph = svc.comfyui.submitted[0]
        self.assertIn("YukariMatting", classes(graph))
        self.assertNotIn(DEPTH_NODE, classes(graph))
        uploads = source_uploads(management)
        self.assertEqual([call[3][0]["role"] for call in uploads], ["alpha", "cut"])
        self.assertEqual(json.loads(uploads[1][3][3]), current_cut())

    def test_an_alpha_cut_before_the_hole_fill_is_recomputed(self):
        before = {key: value for key, value in current_cut()["alpha"].items()
                  if key != "hole_key_share"}
        management = ManagementFake(assets={
            "alpha": b"old-alpha", "depth": b"depth-png",
            "cut": cut_asset(alpha=before)})
        svc, _ = self.run_deliver(management)
        self.assertIn("YukariMatting", classes(svc.comfyui.submitted[0]))

    def test_assets_without_a_cut_description_are_recomputed(self):
        management = ManagementFake(assets={"alpha": b"old-alpha"})
        svc, _ = self.run_deliver(management)
        self.assertIn("YukariMatting", classes(svc.comfyui.submitted[0]))
        self.assertEqual([call[3][0]["role"] for call in source_uploads(management)],
                         ["alpha", "cut"])

    def test_the_delivered_generation_refines_the_source_and_records_the_options(self):
        management = ManagementFake()
        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, management=management)
            deliver("src", svc, request_id="req-1", key_prefix="request:req-1",
                    transparent=True, backdrop=None,
                    outlines=[{"color": "#885b80", "width": 2.0}])
        job = next(call for call in management.calls
                   if call[0] == "POST" and call[1].endswith("/jobs"))
        self.assertEqual(job[2]["source_generation_id"], "src")
        resolution = next(call for call in management.calls
                          if call[1] == "/api/v1/requests/req-1/resolution")
        parameters = resolution[2]["parameters"]
        self.assertEqual(parameters["kind"], "deliver")
        self.assertEqual(parameters["base_generation"], "src")
        self.assertIs(parameters["transparent"], True)
        self.assertNotIn("backdrop", parameters)
        self.assertEqual(parameters["outlines"], [{"color": "#885b80", "width": 2.0}])
        self.assertNotIn("dof", parameters)
        self.assertEqual(resolution[2]["references"][0]["source_generation_id"], "src")

    def test_the_recipe_outlines_are_recorded_when_none_are_given(self):
        management = ManagementFake()
        with tempfile.TemporaryDirectory() as directory:
            deliver("src", services(directory, management=management))
        parameters = next(call for call in management.calls
                          if call[0] == "POST" and call[1] == "/api/v1/requests"
                          )[2]["parameters"]
        self.assertEqual(parameters["outlines"],
                         delivery_style.DELIVER_DEFAULTS["outlines"])

    def test_delivered_outputs_are_refused_as_sources(self):
        cases = [
            dict(request_kind="deliver"),
            dict(request_kind="finalize"),
            dict(parameters={"kind": "deliver"}),
            dict(request_kind="dof"),
            dict(parameters={"kind": "dof"}),
            dict(parameters={"kind": "hires-chain"}),
            dict(parameters={"kind": "repair", "deliver_only": True}),
        ]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                management = ManagementFake(**case)
                svc = services(directory, management=management)
                with self.assertRaisesRegex(SystemExit, "納品済み"):
                    deliver("src", svc)
                self.assertEqual(svc.comfyui.submitted, [])

    def test_a_layerdiffuse_source_is_refused(self):
        graph = {"1": {"class_type": "LayeredDiffusionApply", "inputs": {}}}
        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, management=ManagementFake(graph=graph))
            with self.assertRaisesRegex(SystemExit, "LayerDiffuse"):
                deliver("src", svc)

    def test_any_other_recipe_and_a_repaired_picture_are_accepted(self):
        management = ManagementFake(
            parameters={"kind": "repair", "base_generation": "base"},
            graph={"1": {"class_type": "CheckpointLoaderSimple", "inputs": {}}})
        svc, result = self.run_deliver(management)
        self.assertEqual(result["generation_ids"], ["delivered-1"])
        self.assertIn(("GET", "/api/v1/generations/base", None, None),
                      management.calls)

    def test_a_missing_delivered_output_fails_loudly(self):
        class Silent(ComfyFake):
            def wait_for(self, prompt_id):
                return []

        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, comfyui=Silent())
            with self.assertRaisesRegex(SystemExit, "delivered"):
                deliver("src", svc)


def light_redraw(**parameters):
    return {"request": {"parameters": {
        "kind": "redraw", "method": "light", "base_generation": "gen",
        **parameters}}}


def lineage(*steps):
    """Records for ancestors g1, g2, ... where each step is a request's
    parameters, the last being the oldest."""
    return {f"g{index + 1}": {"request": {"parameters": parameters}}
            for index, parameters in enumerate(steps)}


class InheritedLightTest(unittest.TestCase):
    def run_deliver(self, management, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, management=management)
            deliver("src", svc, **kwargs)
        return svc, management

    def parameters(self, management):
        return next(call for call in management.calls
                    if call[0] == "POST" and call[1] == "/api/v1/requests")[2]["parameters"]

    def test_a_light_redraw_ancestor_lights_the_delivery_and_sets_stroke_light(self):
        management = ManagementFake(
            parameters={"kind": "repair", "base_generation": "g1"},
            records=lineage(
                {"kind": "redraw", "method": "canvas", "base_generation": "g2"},
                {"kind": "redraw", "method": "light", "scene": "moon",
                 "from": "se", "base_generation": "g3"},
                {"kind": "redraw", "method": "light", "scene": "sunset",
                 "from": "n", "base_generation": "g4"}))
        svc, _ = self.run_deliver(management)
        [(_, delivered)] = nodes_of(svc.comfyui.submitted[0], "YukariDeliver")
        self.assertEqual(delivered["inputs"]["light_scene"], "moon")
        self.assertEqual(delivered["inputs"]["light_from"], "se")
        self.assertEqual(delivered["inputs"]["stroke_light"], "se")
        parameters = self.parameters(management)
        self.assertEqual(parameters["light"], {"scene": "moon", "from": "se"})
        self.assertEqual(parameters["stroke_light"], "se")

    def test_the_source_itself_may_be_the_light_redraw(self):
        management = ManagementFake(parameters={
            "kind": "redraw", "method": "light", "scene": "sunset", "from": "w",
            "base_generation": "g1"}, records=lineage({}))
        svc, _ = self.run_deliver(management)
        [(_, delivered)] = nodes_of(svc.comfyui.submitted[0], "YukariDeliver")
        self.assertEqual(delivered["inputs"]["light_scene"], "sunset")
        self.assertEqual(delivered["inputs"]["stroke_light"], "w")

    def test_an_explicit_light_wins_over_the_lineage(self):
        management = ManagementFake(parameters={
            "kind": "redraw", "method": "light", "scene": "sunset", "from": "w",
            "base_generation": "g1"}, records=lineage({}))
        svc, _ = self.run_deliver(management, light=Light("moon", "e"))
        [(_, delivered)] = nodes_of(svc.comfyui.submitted[0], "YukariDeliver")
        self.assertEqual(delivered["inputs"]["light_scene"], "moon")
        self.assertEqual(delivered["inputs"]["stroke_light"], "e")

    def test_neutral_stroke_lights_are_kept_with_an_inherited_light(self):
        for neutral in ("even", None):
            with self.subTest(stroke_light=neutral):
                management = ManagementFake(parameters={
                    "kind": "redraw", "method": "light", "scene": "moon",
                    "from": "se", "base_generation": "g1"}, records=lineage({}))
                svc, _ = self.run_deliver(management, stroke_light=neutral)
                [(_, delivered)] = nodes_of(svc.comfyui.submitted[0], "YukariDeliver")
                self.assertEqual(delivered["inputs"]["light_scene"], "moon")
                self.assertEqual(delivered["inputs"]["stroke_light"], neutral or "")

    def test_a_conflicting_explicit_stroke_light_is_refused(self):
        management = ManagementFake(parameters={
            "kind": "redraw", "method": "light", "scene": "moon", "from": "se",
            "base_generation": "g1"}, records=lineage({}))
        with tempfile.TemporaryDirectory() as directory:
            svc = services(directory, management=management)
            with self.assertRaisesRegex(SystemExit, "light の from"):
                deliver("src", svc, stroke_light="n")
        self.assertEqual(svc.comfyui.submitted, [])

    def test_without_a_light_redraw_the_recipe_defaults_apply(self):
        management = ManagementFake(
            parameters={"kind": "redraw", "method": "canvas", "base_generation": "g1"},
            records=lineage({}))
        svc, _ = self.run_deliver(management)
        [(_, delivered)] = nodes_of(svc.comfyui.submitted[0], "YukariDeliver")
        self.assertNotIn("light_scene", delivered["inputs"])
        self.assertEqual(delivered["inputs"]["stroke_light"], "n")
        self.assertNotIn("light", self.parameters(management))

    def test_the_walk_stops_after_ten_hops(self):
        steps = [{"kind": "redraw", "method": "canvas",
                  "base_generation": f"g{index + 2}"} for index in range(11)]
        steps[10] = {"kind": "redraw", "method": "light", "scene": "moon",
                     "from": "se"}
        management = ManagementFake(
            parameters={"kind": "redraw", "method": "canvas", "base_generation": "g1"},
            records=lineage(*steps))
        svc, _ = self.run_deliver(management)
        [(_, delivered)] = nodes_of(svc.comfyui.submitted[0], "YukariDeliver")
        self.assertNotIn("light_scene", delivered["inputs"])

    def test_the_refines_link_is_followed_when_there_is_no_base_generation(self):
        management = ManagementFake(
            records={"g1": {"request": {"parameters": {
                "kind": "redraw", "method": "light", "scene": "moon", "from": "se"}}}})
        management.context["refines_generation"] = {"id": "g1"}
        svc, _ = self.run_deliver(management)
        [(_, delivered)] = nodes_of(svc.comfyui.submitted[0], "YukariDeliver")
        self.assertEqual(delivered["inputs"]["light_scene"], "moon")


if __name__ == "__main__":
    unittest.main()
