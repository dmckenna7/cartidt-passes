"""Render the qualitative seg + uncertainty panel for Fig. 3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render Fig. 3 qualitative panels")
    parser.add_argument("--input-volume", type=Path, required=True)
    parser.add_argument("--input-seg", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.parse_args(argv if argv is not None else sys.argv[1:])
    raise NotImplementedError(
        "Figure rendering lives in `scripts/render_fig3.py`; this CLI exists only to keep the "
        "import surface aligned with `docs/implementation-map.md`."
    )


if __name__ == "__main__":
    raise SystemExit(main())
