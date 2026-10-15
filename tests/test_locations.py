"""Tests for data/config locations, release.json and the worker status file."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chagama.infrastructure.locations import config_dir, data_dir
from chagama.infrastructure.repository import git_metadata, release_info
from chagama.infrastructure.worker_status import status_path, status_writer

RELEASE = {"version": "r412-25b6fd4", "commit": "a" * 40, "ref": "production"}


class LocationsTest(unittest.TestCase):
    def test_defaults_sit_under_the_repository(self):
        with patch.dict(os.environ, clear=False):
            os.environ.pop("CHAGAMA_DATA", None)
            os.environ.pop("CHAGAMA_CONFIG", None)
            repository = Path("/repo")
            self.assertEqual(data_dir(repository), repository / ".local/_nogit")
            self.assertEqual(config_dir(repository), repository / ".local")

    def test_environment_overrides_both(self):
        with patch.dict(os.environ, {"CHAGAMA_DATA": "/apps/data",
                                     "CHAGAMA_CONFIG": "/secrets"}):
            self.assertEqual(data_dir(Path("/repo")), Path("/apps/data"))
            self.assertEqual(config_dir(Path("/repo")), Path("/secrets"))

    def test_blank_override_falls_back_to_default(self):
        with patch.dict(os.environ, {"CHAGAMA_DATA": " ", "CHAGAMA_CONFIG": ""}):
            self.assertEqual(data_dir(Path("/repo")), Path("/repo/.local/_nogit"))
            self.assertEqual(config_dir(Path("/repo")), Path("/repo/.local"))


class ReleaseMetadataTest(unittest.TestCase):
    def test_release_json_replaces_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp)
            (repository / "release.json").write_text(json.dumps(RELEASE))
            with patch("chagama.infrastructure.repository.subprocess.run") as run:
                metadata = git_metadata(repository)
            run.assert_not_called()
            self.assertEqual(metadata, {
                "commit": "a" * 40, "branch": "production", "dirty": False})
            self.assertEqual(release_info(repository)["version"], "r412-25b6fd4")

    def test_without_release_json_git_is_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(release_info(Path(tmp)))
            with patch("chagama.infrastructure.repository.subprocess.run") as run:
                run.return_value.stdout = "x\n"
                metadata = git_metadata(Path(tmp))
            self.assertEqual(run.call_count, 3)
            self.assertEqual(metadata["commit"], "x")


class StatusFileTest(unittest.TestCase):
    def test_status_records_release_and_process(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"CHAGAMA_DATA": str(Path(tmp) / "data")}):
            repository = Path(tmp) / "release"
            repository.mkdir()
            (repository / "release.json").write_text(json.dumps(RELEASE))
            write = status_writer(repository, lambda: git_metadata(repository))
            write()
            first = json.loads(status_path(repository).read_text())
            write()
            second = json.loads(status_path(repository).read_text())

            self.assertEqual(status_path(repository),
                             Path(tmp) / "data/worker/status.json")
            self.assertEqual(first["version"], "r412-25b6fd4")
            self.assertEqual(first["commit"], "a" * 40)
            self.assertEqual(first["ref"], "production")
            self.assertEqual(first["pid"], os.getpid())
            self.assertEqual(first["started_at"], second["started_at"])
            self.assertTrue(second["updated_at"].endswith("+00:00"))
            self.assertEqual(
                [p.name for p in status_path(repository).parent.iterdir()],
                ["status.json"])

    def test_version_is_null_outside_a_release(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"CHAGAMA_DATA": tmp}):
            write = status_writer(
                Path(tmp), lambda: {"commit": "c", "branch": "dev/x"})
            write()
            record = json.loads(status_path(Path(tmp)).read_text())
            self.assertIsNone(record["version"])
            self.assertEqual(record["ref"], "dev/x")
