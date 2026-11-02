"""Code comments are current-state-only; research logs are append-only.

Enforces the split AGENTS.md declares: experiment history (weights tried,
seeds, render IDs, walk-downs, user quotes, dates) belongs in the render's
semantic on chimera, the commit message or ``docs/``, never in ``src/``
comments. A failure here means content that should be a render record, a
commit message or a doc was written as a comment.
"""

from __future__ import annotations

import io
import pathlib
import re
import tokenize
import unittest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

# Files written before the discipline existed. This list may only SHRINK:
# clean a file (history to chimera/git/docs, comments to current-state),
# then remove it here. Never add an entry.
PENDING_CLEANUP: set[str] = set()

FORBIDDEN = {
    "date (history belongs in chimera or git)":
        re.compile(r"\b20\d\d-\d\d-\d\d\b"),
    "render/prompt id (belongs in chimera or the commit message)":
        re.compile(r"\b[0-9a-f]{8}\b"),
    "numeric walk-down / sweep trail (belongs in chimera or the commit message)":
        re.compile(r"\d(?:\.\d+)?\s*->\s*\d"),
    "user-feedback quote (belongs in chimera semantic or docs/)":
        re.compile(r"[「」]"),
    "pick/ reference (belongs in chimera or docs/)":
        re.compile(r"\bpick/"),
}

# A current-state constraint fits in a few lines. Anything longer is telling
# a story: the record goes to chimera, the conclusion to docs/findings/.
MAX_COMMENT_BLOCK_LINES = 8


def _comments(path: pathlib.Path) -> list[tuple[int, int, str]]:
    """(line, column, text) for every comment token."""
    with path.open(encoding="utf-8") as handle:
        tokens = tokenize.generate_tokens(io.StringIO(handle.read()).readline)
        return [(tok.start[0], tok.start[1], tok.string)
                for tok in tokens if tok.type == tokenize.COMMENT]


class CommentDisciplineTest(unittest.TestCase):
    def _checked_files(self) -> list[pathlib.Path]:
        files = [path for path in sorted(SRC.rglob("*.py"))
                 if str(path.relative_to(SRC)) not in PENDING_CLEANUP]
        self.assertTrue(files, "no source files found under src/")
        return files

    def test_pending_cleanup_entries_still_exist(self) -> None:
        for rel in sorted(PENDING_CLEANUP):
            self.assertTrue((SRC / rel).is_file(),
                            f"stale PENDING_CLEANUP entry: {rel}")

    def test_no_experiment_history_in_comments(self) -> None:
        violations = []
        for path in self._checked_files():
            rel = path.relative_to(SRC)
            for line, _col, comment in _comments(path):
                for label, pattern in FORBIDDEN.items():
                    if pattern.search(comment):
                        violations.append(
                            f"{rel}:{line}: {label}: {comment.strip()[:80]}")
        self.assertEqual(violations, [], "\n" + "\n".join(violations))

    def test_comment_blocks_stay_short(self) -> None:
        violations = []
        for path in self._checked_files():
            rel = path.relative_to(SRC)
            source = path.read_text(encoding="utf-8").splitlines()
            block_start, block_len, prev = None, 0, None
            # Full-line comments only: a column of trailing comments beside
            # consecutive code lines (e.g. stage constants) is not a block.
            rows = [(line, text) for line, col, text in _comments(path)
                    if source[line - 1][:col].strip() == ""] + [(-2, "")]
            for line, _ in rows:
                if prev is not None and line == prev + 1:
                    block_len += 1
                else:
                    if block_start is not None and block_len > MAX_COMMENT_BLOCK_LINES:
                        violations.append(
                            f"{rel}:{block_start}: comment block of "
                            f"{block_len} lines (max {MAX_COMMENT_BLOCK_LINES})"
                            " -- record it on chimera instead")
                    block_start, block_len = line, 1
                prev = line
        self.assertEqual(violations, [], "\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
