"""CLI wrapper for synthetic-depth dataset generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.synthetic_depth_task import main


if __name__ == "__main__":
    raise SystemExit(main())
