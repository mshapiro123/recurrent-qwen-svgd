"""Emit the D0 build-only receipt without models, labeling, or training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.speculative_depth_d0_spec import (
    D0ExecutionPolicy,
    build_only_contract,
    calibrated_depth_targets,
    d0_draft,
    depth_recoverable_fraction,
    dynamic_depth_target,
    unresolved_paths,
)


OUTPUT = ROOT / "outputs/stage5/stage5_paper2_d0_build_only_20260725/summary.json"


def main() -> int:
    D0ExecutionPolicy().assert_allowed(labeling=False, training=False)
    draft = d0_draft()
    fixtures = {
        "dynamic_first_match": dynamic_depth_target([False, True, True, True]),
        "dynamic_never_match": dynamic_depth_target([False, False, False, False]),
        "graded_mapping": calibrated_depth_targets(
            {
                "q1": [0.40, 0.48, 0.50, 0.50],
                "q2": [0.30, 0.31, 0.31, 0.31],
                "q3": [0.20, 0.25, 0.29, 0.30],
                "q4": [0.10, 0.10, 0.10, 0.10],
            }
        ),
        "recoverable_fraction": depth_recoverable_fraction(
            loop1_matches=20,
            self_halted_matches=35,
            rejected_positions=100,
        ),
    }
    summary = {
        "kind": "stage5_paper2_d0_build_only",
        "status": "build_complete_no_labeling_no_training",
        "contract": build_only_contract(),
        "draft_unresolved_paths": unresolved_paths(draft),
        "synthetic_fixture_receipts": fixtures,
        "models_loaded": 0,
        "teacher_forwards": 0,
        "optimizer_steps": 0,
        "checkpoints_written": 0,
        "labeling_authorized": False,
        "training_authorized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
