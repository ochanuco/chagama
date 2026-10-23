from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from chagama.infrastructure.reference import env
from chagama.infrastructure.reference.models import MODELS_ENV, ReferenceUnavailable

ROOT = Path(__file__).resolve().parents[1]


def stub_src(directory: Path, body: str) -> Path:
    """A src/ whose analyze module is `body`, shadowing the real one."""
    src = directory / "src"
    package = src / "chagama" / "infrastructure" / "reference"
    package.mkdir(parents=True)
    for level in (src / "chagama", src / "chagama" / "infrastructure", package):
        (level / "__init__.py").write_text("", encoding="utf-8")
    (package / "analyze.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return src


class EnvLocationTest(unittest.TestCase):
    def test_the_key_hashes_both_requirements_files(self):
        key = env.env_key(ROOT)
        self.assertRegex(key, r"^[0-9a-f]{12}$")
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory)
            for name in env.REQUIREMENTS:
                (copy / name).parent.mkdir(parents=True, exist_ok=True)
                (copy / name).write_bytes((ROOT / name).read_bytes())
            self.assertEqual(env.env_key(copy), key)
            (copy / env.REQUIREMENTS[1]).write_text("torch==0\n", encoding="utf-8")
            self.assertNotEqual(env.env_key(copy), key)

    def test_an_env_not_marked_ready_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {"CHAGAMA_DATA": directory, env.PYTHON_ENV: ""}):
            with self.assertRaisesRegex(ReferenceUnavailable, "not built"):
                env.env_python(ROOT)
            built = Path(directory) / "reference-env" / env.env_key(ROOT)
            built.mkdir(parents=True)
            (built / "ready").write_text("over-comfyui", encoding="ascii")
            self.assertTrue(str(env.env_python(ROOT)).startswith(str(built)))

    def test_the_override_names_any_python(self):
        with patch.dict(os.environ, {env.PYTHON_ENV: "/opt/py/bin/python"}):
            self.assertEqual(env.env_python(ROOT), Path("/opt/py/bin/python"))


class RunAnalysisTest(unittest.TestCase):
    def test_the_subprocess_reads_the_picture_and_answers_in_json(self):
        with tempfile.TemporaryDirectory() as directory:
            src = stub_src(Path(directory), """
                import json, os
                from pathlib import Path

                def main(argv):
                    picture, answer = argv
                    Path(answer).write_text(json.dumps({
                        "size": len(Path(picture).read_bytes()),
                        "models": os.environ.get("CHAGAMA_REFERENCE_MODELS")}))
            """)
            found = env.run_analysis(Path(sys.executable), src, b"12345",
                                     models_dir=Path("/models"))
        self.assertEqual(found, {"size": 5, "models": str(Path("/models"))})

    def test_a_failing_subprocess_raises_with_its_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            src = stub_src(Path(directory), """
                def main(argv):
                    raise SystemExit("d1 needs transformers>=5.14")
            """)
            with self.assertRaisesRegex(RuntimeError, "transformers>=5.14"):
                env.run_analysis(Path(sys.executable), src, b"x")

    def test_a_missing_python_is_unavailable(self):
        with self.assertRaisesRegex(ReferenceUnavailable, "not found"):
            env.run_analysis(Path("/nonexistent/python"), ROOT / "src", b"x")

    def test_a_timeout_is_reported(self):
        def slow(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            env.run_analysis(Path("python"), ROOT / "src", b"x", timeout=1, run=slow)

    def test_the_models_dir_is_passed_through_the_environment(self):
        seen = {}

        def capture(command, **kwargs):
            seen.update(kwargs["env"])
            Path(command[-1]).write_text(json.dumps({}), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        env.run_analysis(Path("python"), ROOT / "src", b"x",
                         models_dir=Path("/m"), run=capture)
        self.assertEqual(seen[MODELS_ENV], str(Path("/m")))


if __name__ == "__main__":
    unittest.main()
