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

    @patch.object(cli, "deliver")
    @patch.object(cli, "ChimeraClient")
    def test_deliver_without_flags_runs_the_recipe_defaults(
            self, chimera_class, run_deliver):
        cli.main(["deliver", "gen-1"])
        args, kwargs = run_deliver.call_args
        self.assertEqual(args[0], "gen-1")
        self.assertEqual(kwargs, {
            "keep_scene": False, "transparent": False, "backdrop": "dots",
            "stroke_light": RECIPE_DEFAULT,
            "outlines": [{"color": "#ffffff", "width": 0.8},
                         {"color": "#885b80", "width": 1.04}],
            "light": None})

    @patch.object(cli, "deliver")
    @patch.object(cli, "ChimeraClient")
    def test_deliver_flags_dispatch_without_network(
            self, chimera_class, run_deliver):
        cli.main(["deliver", "gen-1", "--transparent", "--light", "moon,se",
                  "--outline", "#ffffff,0.8", "--outline", "#885b80,2"])
        kwargs = run_deliver.call_args.kwargs
        self.assertIs(kwargs["transparent"], True)
        self.assertIsNone(kwargs["backdrop"])
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
    def test_redraw_hires_dispatches_without_network(
            self, chimera_class, run_redraw):
        cli.main(["redraw", "gen-1", "--method", "hires", "--hires", "2048"])
        args, kwargs = run_redraw.call_args
        self.assertEqual(args[0], "gen-1")
        self.assertEqual((kwargs["method"], kwargs["hires"], kwargs["denoise"]),
                         ("hires", 2048, 0.45))
        cli.main(["redraw", "gen-1", "--method", "hires", "--hires", "2048",
                  "--denoise", "0.3", "--cut"])
        kwargs = run_redraw.call_args.kwargs
        self.assertEqual((kwargs["denoise"], kwargs["cut"]), (0.3, True))

    @patch.object(cli, "redraw")
    @patch.object(cli, "ChimeraClient")
    def test_redraw_refuses_a_retired_method_and_a_missing_one(
            self, chimera_class, run_redraw):
        for argv in (["redraw", "gen-1", "--method", "canvas"],
                     ["redraw", "gen-1", "--method", "light"],
                     ["redraw", "gen-1"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                cli.main(argv)
        run_redraw.assert_not_called()

    @patch.object(cli, "work")
    @patch.object(cli, "ChimeraClient")
    def test_work_default_kinds_include_every_request_kind(
            self, chimera_class, run_work):
        cli.main(["work", "--once"])
        work_services = run_work.call_args.args[0]
        self.assertEqual(
            work_services.kinds,
            ("generate", "redraw", "deliver", "dof", "lens", "photo", "stage",
             "lighting", "saturation", "reference", "reference_render"))
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
