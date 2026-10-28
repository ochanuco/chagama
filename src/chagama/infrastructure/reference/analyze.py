"""One picture through the Anima Tagger, then d1 on the tags it left
undecided, then DWPose for the figure's pose and anime-seg for its
silhouette and lineart.

Runs inside the reference env (`env.run_analysis`):
`main([picture, answer])` writes the analysis as JSON to `answer`. The
selection itself stays with the worker; this returns only what it reads.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image

from ...domain.yukari import reference as rules
from . import dwpose, outline


def analyze(image: Image.Image, load_tagger: Callable, load_verifier: Callable,
            load_estimator: Callable, load_segmenter: Callable) -> dict:
    """`{scores, kept, groups, verdicts, figure}`: candidate scores, every kept
    tag, the candidate groups, d1's P(yes) per asked tag and the figure's pose
    (`dwpose.detect`, with `outline.outlines` added when a figure is found)."""
    tagger = load_tagger()
    try:
        tagging = tagger.predict(image)
    finally:
        tagger.close()
    sources = {source for names in rules.PART_SOURCES.values() for source in names}
    groups = {group.name: list(group.tags) for group in tagging.groups
              if group.name in sources}
    asked = rules.questions(tagging.scores, tagging.kept, groups)
    verdicts: dict[str, float] = {}
    if asked:
        verifier = load_verifier()
        try:
            verdicts = verifier.verify(image, asked)
        finally:
            verifier.close()
    candidates = rules.candidates(groups)
    figure = dwpose.detect(image, load_estimator())
    if figure is not None:
        figure.update(outline.outlines(image, load_segmenter()))
    return {
        "scores": {tag: score for tag, score in tagging.scores.items()
                   if tag in candidates},
        "kept": dict(tagging.kept),
        "groups": groups,
        "verdicts": verdicts,
        "figure": figure,
    }


def main(argv: Sequence[str] | None = None) -> None:
    picture, answer = argv if argv is not None else sys.argv[1:]
    from .anima_tagger import load_tagger
    from .d1 import load_verifier

    with Image.open(picture) as opened:
        image = opened.convert("RGB")
    result = analyze(image, load_tagger, load_verifier, dwpose.load_estimator,
                     outline.load_segmenter)
    Path(answer).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
