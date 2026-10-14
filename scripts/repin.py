#!/usr/bin/env python3
"""Pin a render's colour to the palette, leaving the brushwork alone.

Thin CLI over `chagama.infrastructure.imaging.palette.repin`; the
windows and targets it applies live in the Yukari domain's `delivery_style`.

    uv run scripts/repin.py <in.png> <out.png>
"""
import sys

import numpy as np
from PIL import Image

from chagama.infrastructure.imaging.palette import repin


def main() -> None:
    im = np.array(Image.open(sys.argv[1]).convert("RGB"))
    rgb, report = repin(im)
    for line in report:
        print(line)
    Image.fromarray(rgb).save(sys.argv[2])
    print(sys.argv[2])


if __name__ == "__main__":
    main()
