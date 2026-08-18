"""Freeze the Stage 2B full-sequence corpus and loss-calibration panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.paper2_stage2b_data import write_prelock_packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-data", type=Path, required=True)
    parser.add_argument("--new-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    packet = write_prelock_packet(
        old_data=args.old_data, new_data=args.new_data, output_dir=args.output_dir
    )
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

