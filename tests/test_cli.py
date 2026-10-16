from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from chagama.application.deliver import RECIPE_DEFAULT
from chagama.domain.yukari.delivery_style import Light
from chagama.interfaces import cli


class CliTest(unittest.TestCase):
    @patch.object(cli, "generate")
    @patch.object(cli, "ChimeraClient")
    def test_generate_dispatches_without_network(self, chimera_class, run_generate):
        cli.main(["generate", "--request", "request.json", "--dry-run", "--force"])
        services = run_generate.call_args.args[1]
        self.assertIs(services.management, chimera_class.return_value)
        self.assertEqual(str(run_generate.call_args.args[0]), "request.json")
        self.assertEqual(run_generate.call_args.kwargs,
                         {"dry_run": True, "force": True})

    @patch.object(cli, "git_metadata", return_value={"commit": "c", "dirty": False})
    @patch.object(cli, "import_images")
    @patch.object(cli, "ChimeraClient")
    def test_import_registers_images_with_a_content_derived_key(
            self, chimera_class, run_import, git_metadata):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "a.png"
            image.write_bytes(b"png")
            with redirect_stdout(io.StringIO()):
                run_import.return_value = {"generation_ids": ["g1"]}
                cli.main(["import", str(image), "--recipe", "yukari",
                          "--parameters", '{"pose": "bust"}',
                          "--references",
                          '[{"source_generation_id": "src", "purpose": "rebuild"}]'])
                cli.main(["import", str(image), "--recipe", "yukari"])
        first, second = run_import.call_args_list
        self.assertIs(first.args[0], chimera_class.return_value)
        self.assertEqual(first.kwargs["images"], [image])
        self.assertEqual(first.kwargs["resolution"], {
            "recipe": "yukari", "raw_instruction": "",
            "parameters": {"pose": "bust"},
            "git_commit": "c", "git_dirty": False,
            "references": [{"source_generation_id": "src", "purpose": "rebuild"}]})
        self.assertTrue(first.kwargs["idempotency_key"].startswith("import:"))
        self.assertEqual(first.kwargs["idempotency_key"],
                         second.kwargs["idempotency_key"])

    @patch.object(cli, "watch")
    @patch.object(cli, "ChimeraClient")
    def test_watch_dispatches_without_network(self, chimera_class, run_watch):
        cli.main(["watch", "--interval", "5", "--once", "--dry-run"])
        watch_services = run_watch.call_args.args[0]
        self.assertIs(watch_services.management, chimera_class.return_value)
        self.assertIs(
            watch_services.generate_services.management, chimera_class.return_value)
        self.assertEqual(run_watch.call_args.kwargs,
                         {"interval": 5.0, "once": True, "dry_run": True})

    @patch.object(cli, "work")
    @patch.object(cli, "ChimeraClient")
    def test_work_dispatches_without_network(self, chimera_class, run_work):
        cli.main(["work", "--interval", "5", "--once", "--dry-run",
                  "--worker-id", "worker-1", "--kinds", "generate"])
        work_services = run_work.call_args.args[0]
        self.assertIs(work_services.management, chimera_class.return_value)
        self.assertIs(
            work_services.generate_services.management, chimera_class.return_value)
        self.assertEqual(work_services.worker_id, "worker-1")
        self.assertEqual(work_services.kinds, ("generate",))
        self.assertEqual(run_work.call_args.kwargs,
                         {"interval": 5.0, "once": True, "dry_run": True,
                          "publish_catalog": True})

    @patch.object(cli, "repair")
    @patch.object(cli, "ChimeraClient")
    def test_repair_dispatches_without_network(self, chimera_class, run_repair):
        cli.main(["repair", "gen-1", "--parts", "hands, feet",
                  "--region", "0.1,0.2,0.3,0.4", "--region", "0.5,0.5,0.9,0.9",
                  "--denoise", "0.7", "--seeds", "1,2,3", "--size", "768",
                  "--pad", "1.5"])
        args, kwargs = run_repair.call_args
        self.assertEqual(args[0], "gen-1")
        self.assertIs(args[1].management, chimera_class.return_value)
        self.assertEqual(kwargs["parts"], ["hands", "feet"])
        self.assertEqual(kwargs["regions"], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.9, 0.9]])
        self.assertEqual(kwargs["denoise"], 0.7)
        self.assertEqual(kwargs["seeds"], [1, 2, 3])
        self.assertEqual(kwargs["size"], 768)
        self.assertEqual(kwargs["pad"], 1.5)

    @patch.object(cli, "repair")
    @patch.object(cli, "ChimeraClient")
    def test_repair_defaults_need_no_flags(self, chimera_class, run_repair):
        cli.main(["repair", "gen-1"])
        args, kwargs = run_repair.call_args
        self.assertEqual(kwargs["parts"], ["hands", "feet"])
        self.assertEqual(kwargs["regions"], [])
        self.assertEqual(kwargs["denoise"], 0.6)
        self.assertEqual(kwargs["seeds"], [1, 2, 3, 4])
        self.assertEqual(kwargs["size"], 1024)
        self.assertEqual(kwargs["pad"], 1.0)
        self.assertIsNone(kwargs["model"])

    @patch.object(cli, "repair")
    @patch.object(cli, "ChimeraClient")
    def test_repair_model_flag_dispatches_without_network(
            self, chimera_class, run_repair):
        cli.main(["repair", "gen-1", "--model", "anima"])
        kwargs = run_repair.call_args.kwargs
        self.assertEqual(kwargs["model"], "anima")

    def test_repair_unknown_model_is_rejected_by_argparse(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["repair", "gen-1", "--model", "nope"])

    @patch.object(cli, "deliver")
    @patch.object(cli, "ChimeraClient")
    def test_deliver_without_flags_runs_the_recipe_defaults(
            self, chimera_class, run_deliver):
        cli.main(["deliver", "gen-1"])
        args, kwargs = run_deliver.call_args
        self.assertEqual(args[0], "gen-1")
        self.assertEqual(kwargs, {
            "skin": False, "recolor": False, "keep_legwear": None,
            "keep_scene": False, "transparent": False, "backdrop": "dots",
            "stroke_light": RECIPE_DEFAULT, "deliver_size": None,
            "outlines": [{"color": "#ffffff", "width": 0.8},
                         {"color": "#885b80", "width": 1.04}],
            "light": None, "context": None})

    @patch.object(cli, "deliver")
    @patch.object(cli, "ChimeraClient")
    def test_deliver_flags_dispatch_without_network(
            self, chimera_class, run_deliver):
        cli.main(["deliver", "gen-1", "--skin", "--transparent",
                  "--deliver-size", "1536", "--light", "moon,se",
                  "--outline", "#ffffff,0.8", "--outline", "#885b80,2",
                  "--keep-legwear"])
        kwargs = run_deliver.call_args.kwargs
        self.assertIs(kwargs["skin"], True)
        self.assertIs(kwargs["transparent"], True)
        self.assertIsNone(kwargs["backdrop"])
        self.assertEqual(kwargs["deliver_size"], 1536)
        self.assertEqual(kwargs["keep_legwear"], 0.62)
        self.assertEqual(kwargs["outlines"], [{"color": "#ffffff", "width": 0.8},
                                              {"color": "#885b80", "width": 2.0}])
        self.assertEqual(kwargs["light"], Light("moon", "se"))
        self.assertEqual(kwargs["stroke_light"], "se")

    @patch.object(cli, "deliver")
    @patch.object(cli, "ChimeraClient")
    def test_deliver_rejects_a_stroke_light_that_disagrees_with_light(
            self, chimera_class, run_deliver):
        with self.assertRaisesRegex(SystemExit, "light の from"):
            cli.main(["deliver", "gen-1", "--light", "moon,se",
                      "--stroke-light", "n"])
        run_deliver.assert_not_called()

    @patch.object(cli, "deliver")
    @patch.object(cli, "ChimeraClient")
    def test_deliver_no_outlines_and_no_dof(self, chimera_class, run_deliver):
        cli.main(["deliver", "gen-1", "--no-outlines"])
        self.assertEqual(run_deliver.call_args.kwargs["outlines"], [])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["deliver", "gen-1", "--dof", "0.5,0.5,2.8"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["deliver", "gen-1", "--stroke-light", "none"])

    @patch.object(cli, "dof")
    @patch.object(cli, "ChimeraClient")
    def test_dof_flags_dispatch_without_network(self, chimera_class, run_dof):
        cli.main(["dof", "gen-1", "--focus", "0.5,0.4", "--f-number", "4",
                  "--scope", "figure,outline", "--viewfinder", "both"])
        args, kwargs = run_dof.call_args
        self.assertEqual(args[0], "gen-1")
        self.assertEqual(kwargs, {
            "focus": (0.5, 0.4), "f_number": 4.0, "viewfinder": "both",
            "scope": {"figure": True, "outline": True, "backdrop": False},
            "photo_blur": None})
        cli.main(["dof", "gen-1", "--focus", "0.5,0.4"])
        self.assertEqual(run_dof.call_args.kwargs["scope"],
                         {"figure": True, "outline": True, "backdrop": True})
        with self.assertRaisesRegex(SystemExit, "scope"):
            cli.main(["dof", "gen-1", "--focus", "0.5,0.4", "--scope", "rim"])

    @patch.object(cli, "saturation")
    @patch.object(cli, "ChimeraClient")
    def test_saturation_dispatches_without_network(self, chimera_class, run_saturation):
        cli.main(["saturation", "gen-1"])
        args, kwargs = run_saturation.call_args
        self.assertEqual((args[0], kwargs), ("gen-1", {"strength": 1.0}))
        cli.main(["saturation", "gen-1", "--strength", "0.4"])
        self.assertEqual(run_saturation.call_args.kwargs, {"strength": 0.4})
        with self.assertRaises(SystemExit):
            cli.main(["saturation", "gen-1", "--strength", "2"])

    @patch.object(cli, "lens")
    @patch.object(cli, "ChimeraClient")
    def test_lens_flags_dispatch_without_network(self, chimera_class, run_lens):
        cli.main(["lens", "gen-1", "--center", "0.5,0.8", "--strength", "-0.6"])
        args, kwargs = run_lens.call_args
        self.assertEqual(args[0], "gen-1")
        self.assertEqual(kwargs, {"center": (0.5, 0.8), "strength": -0.6,
                                  "shape": "fisheye", "radius": None})
        cli.main(["lens", "gen-1"])
        self.assertEqual(run_lens.call_args.kwargs,
                         {"center": (0.5, 0.5), "strength": 0.7,
                          "shape": "fisheye", "radius": None})
        cli.main(["lens", "gen-1", "--shape", "sphere", "--radius", "0.4"])
        self.assertEqual((run_lens.call_args.kwargs["shape"],
                          run_lens.call_args.kwargs["radius"]), ("sphere", 0.4))
        with self.assertRaisesRegex(SystemExit, "center"):
            cli.main(["lens", "gen-1", "--center", "middle"])
        with self.assertRaisesRegex(SystemExit, "strength"):
            cli.main(["lens", "gen-1", "--strength", "0"])

    @patch.object(cli, "photo")
    @patch.object(cli, "ChimeraClient")
    def test_photo_flags_dispatch_without_network(self, chimera_class, run_photo):
        cli.main(["photo", "gen-1", "--photo", "pic-1", "--place", "0.3,0.9",
                  "--height", "0.5", "--stand-at", "0.8,0.4", "--nearer", "0.1",
                  "--focus", "0.8,0.4", "--f-number", "2", "--tilt", "0.3", "--bokeh", "0.6"])
        args, kwargs = run_photo.call_args
        self.assertEqual(args[0], "gen-1")
        self.assertEqual(kwargs, {
            "photo_id": "pic-1", "place": (0.3, 0.9), "height": 0.5,
            "stand_at": (0.8, 0.4), "nearer": 0.1, "focus": (0.8, 0.4),
            "f_number": 2.0, "tilt": 0.3, "bokeh": 0.6, "focal_length": "85mm"})
        cli.main(["photo", "gen-1", "--photo", "pic-1"])
        self.assertEqual(run_photo.call_args.kwargs, {
            "photo_id": "pic-1", "place": (0.5, 0.95), "height": 0.7,
            "stand_at": None, "nearer": 0.0, "focus": "subject", "f_number": 2.8,
            "tilt": 0.0, "bokeh": 0.0, "focal_length": "85mm"})
        with self.assertRaisesRegex(SystemExit, "place"):
            cli.main(["photo", "gen-1", "--photo", "pic-1", "--place", "left"])

    @patch.object(cli, "redraw")
    @patch.object(cli, "ChimeraClient")
    def test_redraw_methods_dispatch_without_network(
            self, chimera_class, run_redraw):
        cli.main(["redraw", "gen-1", "--method", "canvas", "--denoise", "0.5",
                  "--size", "2048", "--latent-route", "--upscale", "lanczos",
                  "--keep-region", "0,0,0.5,0.5"])
        args, kwargs = run_redraw.call_args
        self.assertEqual(args[0], "gen-1")
        self.assertEqual(kwargs["method"], "canvas")
        self.assertEqual(kwargs["denoise"], 0.5)
        self.assertEqual(kwargs["size"], 2048)
        self.assertIs(kwargs["latent_route"], True)
        self.assertEqual(kwargs["keep_regions"], [[0.0, 0.0, 0.5, 0.5]])
        cli.main(["redraw", "gen-1", "--method", "hires", "--hires", "2048"])
        kwargs = run_redraw.call_args.kwargs
        self.assertEqual((kwargs["method"], kwargs["hires"], kwargs["denoise"]),
                         ("hires", 2048, 0.45))
        cli.main(["redraw", "gen-1", "--method", "light", "--light", "moon,se"])
        kwargs = run_redraw.call_args.kwargs
        self.assertEqual((kwargs["method"], kwargs["light"]),
                         ("light", Light("moon", "se")))

    @patch.object(cli, "redraw")
    @patch.object(cli, "ChimeraClient")
    def test_redraw_refuses_flags_of_another_method(
            self, chimera_class, run_redraw):
        with self.assertRaises(SystemExit):
            cli.main(["redraw", "gen-1", "--method", "light", "--light", "moon",
                      "--size", "2048"])
        with self.assertRaises(SystemExit):
            cli.main(["redraw", "gen-1"])
        run_redraw.assert_not_called()

    @patch.object(cli, "dials_scope")
    @patch.object(cli, "fetch_source")
    @patch.object(cli, "redraw")
    @patch.object(cli, "ChimeraClient")
    def test_redraw_denoise_word_resolves_through_the_source_recipe(
            self, chimera_class, run_redraw, fetch_source, dials_scope):
        context = object()
        fetch_source.return_value = (context, "yukari")
        dials_scope.return_value = {"denoise": {"tidy": 0.65}}
        cli.main(["redraw", "gen-1", "--method", "canvas", "--denoise", "tidy"])
        fetch_source.assert_called_once_with(chimera_class.return_value, "gen-1")
        dials_scope.assert_called_once_with("yukari", "redraw")
        args, kwargs = run_redraw.call_args
        self.assertEqual(kwargs["denoise"], 0.65)
        # The context fetch_source already made is passed through so
        # redraw() does not fetch it again.
        self.assertIs(kwargs["context"], context)

    @patch.object(cli, "dials_scope")
    @patch.object(cli, "fetch_source")
    @patch.object(cli, "redraw")
    @patch.object(cli, "ChimeraClient")
    def test_redraw_unknown_word_exits_before_redrawing(
            self, chimera_class, run_redraw, fetch_source, dials_scope):
        fetch_source.return_value = ({}, "yukari")
        dials_scope.return_value = {"denoise": {"keep": 0.4}}
        with self.assertRaises(SystemExit):
            cli.main(["redraw", "gen-1", "--method", "canvas", "--denoise", "blurry"])
        run_redraw.assert_not_called()

    @patch.object(cli, "fetch_source")
    @patch.object(cli, "redraw")
    @patch.object(cli, "ChimeraClient")
    def test_redraw_numeric_denoise_never_looks_up_the_recipe(
            self, chimera_class, run_redraw, fetch_source):
        cli.main(["redraw", "gen-1", "--method", "canvas", "--denoise", "0.7"])
        fetch_source.assert_not_called()
        kwargs = run_redraw.call_args.kwargs
        self.assertEqual(kwargs["denoise"], 0.7)
        self.assertIsNone(kwargs["context"])

    @patch.object(cli, "dials_scope")
    @patch.object(cli, "fetch_source")
    @patch.object(cli, "repair")
    @patch.object(cli, "ChimeraClient")
    def test_repair_denoise_and_lora_words_resolve_through_the_source_recipe(
            self, chimera_class, run_repair, fetch_source, dials_scope):
        context = object()
        fetch_source.return_value = (context, "yukari")
        dials_scope.return_value = {"denoise": {"keep": 0.6}, "lora": {"on": 0.8}}
        cli.main(["repair", "gen-1", "--denoise", "keep", "--lora", "on"])
        fetch_source.assert_called_once_with(chimera_class.return_value, "gen-1")
        dials_scope.assert_called_once_with("yukari", "repair")
        args, kwargs = run_repair.call_args
        self.assertEqual(kwargs["denoise"], 0.6)
        self.assertEqual(kwargs["lora"], 0.8)
        # The context fetch_source already made is passed through so
        # repair() does not fetch it again.
        self.assertIs(kwargs["context"], context)

    @patch.object(cli, "fetch_source")
    @patch.object(cli, "repair")
    @patch.object(cli, "ChimeraClient")
    def test_repair_numeric_args_never_look_up_the_recipe(
            self, chimera_class, run_repair, fetch_source):
        cli.main(["repair", "gen-1", "--denoise", "0.7"])
        fetch_source.assert_not_called()
        kwargs = run_repair.call_args.kwargs
        self.assertEqual(kwargs["denoise"], 0.7)
        self.assertIsNone(kwargs["context"])

    @patch.object(cli, "work")
    @patch.object(cli, "ChimeraClient")
    def test_work_default_kinds_include_every_request_kind(
            self, chimera_class, run_work):
        cli.main(["work", "--once"])
        work_services = run_work.call_args.args[0]
        self.assertEqual(
            work_services.kinds,
            ("generate", "redraw", "repair", "masked_redraw", "deliver", "dof", "lens", "photo",
             "stage", "lighting", "saturation"))
        self.assertIs(
            work_services.stage_services.management, chimera_class.return_value)
        self.assertIs(
            work_services.dof_services.management, chimera_class.return_value)
        self.assertIs(
            work_services.lens_services.management, chimera_class.return_value)
        self.assertIs(
            work_services.photo_services.management, chimera_class.return_value)
        self.assertIs(
            work_services.redraw_services.management, chimera_class.return_value)
        self.assertIs(
            work_services.deliver_services.management, chimera_class.return_value)
        self.assertIs(
            work_services.repair_services.management, chimera_class.return_value)
        self.assertIs(
            work_services.masked_redraw_services.management, chimera_class.return_value)

    @patch.object(cli, "work")
    @patch.object(cli, "ChimeraClient")
    def test_work_no_catalog_flag_disables_publish(self, chimera_class, run_work):
        cli.main(["work", "--once", "--no-catalog"])
        self.assertFalse(run_work.call_args.kwargs["publish_catalog"])

    @patch.object(cli, "ChimeraClient")
    def test_catalog_prints_the_document_and_does_not_publish(self, chimera_class):
        output = io.StringIO()
        with redirect_stdout(output):
            cli.main(["catalog"])
        chimera_class.return_value.put_catalog.assert_not_called()
        document = cli.json.loads(output.getvalue())
        self.assertEqual(document["schema_version"], 3)
        self.assertEqual(
            {recipe["name"] for recipe in document["recipes"]},
            {"yukari"})

    @patch.object(cli, "publish_catalog_document")
    @patch.object(cli, "ChimeraClient")
    def test_catalog_publish_flag_puts_and_prints_the_response(
            self, chimera_class, publish):
        publish.return_value = {"ok": True}
        output = io.StringIO()
        with redirect_stdout(output):
            cli.main(["catalog", "--publish"])
        args, kwargs = publish.call_args
        self.assertIs(args[0], chimera_class.return_value)
        self.assertIn('"ok": true', output.getvalue())

    @patch.object(cli.metadata, "add_tag")
    @patch.object(cli, "ChimeraClient")
    def test_metadata_tag_dispatches_without_network(self, chimera_class, add_tag):
        with redirect_stdout(io.StringIO()):
            cli.main(["metadata", "tag", "generation", "approved"])
        add_tag.assert_called_once_with(
            chimera_class.return_value, "generation", "approved")

    @patch.object(cli.metadata, "record_publication")
    @patch.object(cli, "ChimeraClient")
    def test_metadata_publish_dispatches_without_network(
            self, chimera_class, record_publication):
        output = io.StringIO()
        with redirect_stdout(output):
            cli.main(["metadata", "publish", "generation",
                      "--url", "https://x.com/post/1", "--idempotency-key", "pub-key"])
        record_publication.assert_called_once_with(
            chimera_class.return_value, "generation", url="https://x.com/post/1",
            idempotency_key="pub-key")
        self.assertIn("publish -> generation (https://x.com/post/1)", output.getvalue())

    @patch.object(cli.metadata, "record_publication")
    @patch.object(cli, "ChimeraClient")
    def test_metadata_publish_without_url_needs_no_flag(
            self, chimera_class, record_publication):
        with redirect_stdout(io.StringIO()):
            cli.main(["metadata", "publish", "generation"])
        record_publication.assert_called_once_with(
            chimera_class.return_value, "generation", url=None, idempotency_key=None)

    def test_yukari_prompt_json_needs_no_clients(self):
        output = io.StringIO()
        with patch.object(cli, "ChimeraClient") as chimera_class, \
                redirect_stdout(output):
            cli.main(["yukari", "prompt", "--pose", "bust", "--json"])
        chimera_class.assert_not_called()
        self.assertIn('"positive"', output.getvalue())
        self.assertIn('"negative"', output.getvalue())


class WatchIntervalTest(unittest.TestCase):
    def _parse(self, interval: str):
        with redirect_stderr(io.StringIO()):
            return cli.parser().parse_args(["watch", "--interval", interval])

    def test_zero_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("0")

    def test_negative_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("-1")

    def test_nan_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("nan")

    def test_infinity_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("inf")

    def test_non_numeric_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("soon")

    def test_valid_value_is_accepted(self):
        args = self._parse("2.5")
        self.assertEqual(args.interval, 2.5)

    @patch.object(cli, "watch")
    @patch.object(cli, "ChimeraClient")
    def test_valid_value_reaches_watch(self, chimera_class, run_watch):
        cli.main(["watch", "--interval", "2.5", "--once"])
        self.assertEqual(run_watch.call_args.kwargs["interval"], 2.5)


if __name__ == "__main__":
    unittest.main()
