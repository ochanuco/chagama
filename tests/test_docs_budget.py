"""docs/ stays small enough to open: size caps, a complete index, a frozen archive.

Findings are rewritten in place when knowledge changes; history lives in
chimera and git. The archive is the pre-split measurement log and
takes no further writes.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import unittest

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

MAX_LINES = 300
MAX_FINDING_LINES = 150

# Files over MAX_LINES when the cap arrived. Values may only go DOWN; remove
# an entry once the file fits. Never add one.
OVERSIZE: dict[str, int] = {}

ARCHIVE_SHA256 = {
    "render-notes/1-early.md":
        "39625ce8ca12456ace5b35d33154bf6921097f38e5cad3d1b5c14283f29b859d",
    "render-notes/2-2026-08-16_17.md":
        "7dc26fa4ae0b9fd33903b4665fe04ec83bd590eb5325f5f302aa2b96fbb26cc3",
    "render-notes/3-2026-08-18_19.md":
        "778736c05ff640df3d9b04a67ed53141a8e57d203d2d2a7180acd2185eebbec6",
    "render-notes/4-2026-08-19_25.md":
        "8b00bc7c195c1a5683cbf516df41313792f7207ca3f0b94f5bcaa0d7eb0f14d2",
    "render-notes/5-2026-08-26_31.md":
        "80aeb08d1ec62b3216303be886542d99bf0217be931adf1de328fa53a7d6bf91",
    "render-notes/6-2026-09.md":
        "67c1dfe6d00722e574403908d366220abefbc8911af56d165c3eeb3ebc55bbc8",
}


def live_docs() -> list[pathlib.Path]:
    return [p for p in sorted(DOCS.rglob("*.md"))
            if p.relative_to(DOCS).parts[0] != "archive"]


class DocsBudgetTest(unittest.TestCase):
    def test_line_caps(self) -> None:
        for path in live_docs():
            rel = path.relative_to(DOCS).as_posix()
            count = len(path.read_text(encoding="utf-8").splitlines())
            cap = MAX_FINDING_LINES if rel.startswith("findings/") else MAX_LINES
            cap = OVERSIZE.get(rel, cap)
            with self.subTest(doc=rel):
                self.assertLessEqual(
                    count, cap,
                    f"docs/{rel} is {count} lines (cap {cap}). Rewrite it shorter "
                    "or split it by topic; do not append history -- that goes "
                    "to chimera or git.")

    def test_oversize_entries_still_needed(self) -> None:
        for rel, cap in OVERSIZE.items():
            path = DOCS / rel
            with self.subTest(doc=rel):
                self.assertTrue(path.exists(), f"remove docs/{rel} from OVERSIZE")
                count = len(path.read_text(encoding="utf-8").splitlines())
                self.assertGreater(count, MAX_LINES,
                                   f"docs/{rel} fits now; remove it from OVERSIZE")

    def test_index_lists_every_doc(self) -> None:
        index = (DOCS / "README.md").read_text(encoding="utf-8")
        linked = set(re.findall(r"\]\(([^)#]+\.md)", index))
        for path in live_docs():
            rel = path.relative_to(DOCS).as_posix()
            if rel == "README.md":
                continue
            with self.subTest(doc=rel):
                self.assertIn(rel, linked, f"docs/README.md does not list {rel}")

    def test_archive_is_frozen(self) -> None:
        archive = DOCS / "archive"
        found = {p.relative_to(archive).as_posix()
                 for p in archive.rglob("*.md") if p.name != "README.md"}
        self.assertEqual(found, set(ARCHIVE_SHA256),
                         "docs/archive/ takes no new files")
        for rel, digest in ARCHIVE_SHA256.items():
            with self.subTest(doc=rel):
                actual = hashlib.sha256((archive / rel).read_bytes()).hexdigest()
                self.assertEqual(actual, digest,
                                 f"docs/archive/{rel} changed; write the new "
                                 "conclusion into docs/findings/ instead")


if __name__ == "__main__":
    unittest.main()
