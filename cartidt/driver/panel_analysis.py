"""Render the chord + boxplot + Sankey + heatmap analysis panel for Fig. 4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render Fig. 4 analytic panels")
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.parse_args(argv if argv is not None else sys.argv[1:])
    raise NotImplementedError(
        "Implemented in `scripts/render_fig4.py`; this entry only preserves the public import surface."
    )


if __name__ == "__main__":
    raise SystemExit(main())
