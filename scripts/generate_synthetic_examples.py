#!/usr/bin/env python
"""Generate the bounded synthetic STED MVP example set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.synthetic.schema import load_yaml
from fibras.synthetic.storage import save_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    save_dataset(load_yaml(args.config), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

