from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "src/chagama/domain"
FORBIDDEN_DOMAIN_IMPORTS = {
    "argparse",
    "numpy",
    "os",
    "pathlib",
    "PIL",
    "requests",
    "scipy",
    "urllib",
}


class ArchitectureTest(unittest.TestCase):
    def test_domain_has_no_io_or_framework_imports(self) -> None:
        problems = []
        for path in DOMAIN.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    root = name.split(".", 1)[0]
                    if root in FORBIDDEN_DOMAIN_IMPORTS:
                        problems.append(f"{path.relative_to(ROOT)} imports {name}")
        self.assertEqual(problems, [])

    def test_comfyui_node_schema_stays_outside_domain(self) -> None:
        offenders = [
            str(path.relative_to(ROOT))
            for path in DOMAIN.rglob("*.py")
            if "class_type" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
