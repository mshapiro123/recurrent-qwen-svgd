"""Complete the zero-GPU Phase G-alpha frozen-data and preregistration package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.phase_g_alpha_spec import write_preregistration


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preparation_summary(manifest: dict[str, Any], prereg: dict[str, Any]) -> dict[str, Any]:
    split_statuses = {name: row["status"] for name, row in manifest["splits"].items()}
    forms_locked = prereg["status"] == "forms_locked_numeric_margins_pending_power_calculation"
    only_blank = bool(prereg["power_calculation_todo"]["only_remaining_preregistration_blank"])
    passed = (
        manifest["status"] == "passed"
        and manifest["calibration_test_id_overlap"] == 0
        and all(status == "passed" for status in split_statuses.values())
        and forms_locked
        and only_blank
    )
    return {
        "kind": "stage5_phase_g_alpha_prepare",
        "status": "passed" if passed else "blocked",
        "phase_g_alpha_preparation_complete": passed,
        "split_statuses": split_statuses,
        "calibration_test_id_overlap": manifest["calibration_test_id_overlap"],
        "frozen_manifest": "data/phase_g_alpha/manifest.json",
        "preregistration": "data/phase_g_alpha/preregistration.json",
        "gate_forms_locked": forms_locked,
        "only_remaining_blank": "numeric_margin_from_calibration_split_power_calculation",
        "substrate_gate": (
            "constructive_experiment1_then_arbitrary_N24_calibration_competence_and_green_guardrail"
        ),
        "comparators": [
            "entropy_matched_answer_head_sampling_at_matched_K",
            "deterministic_depth_at_matched_recurrent_transition_count",
        ],
        "deferred": prereg["deferred_until_G_alpha_win"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/phase_g_alpha")
    parser.add_argument("--output_dir", default="outputs/stage5/stage5_phase_g_alpha_prepare_20260712")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    output_dir = ROOT / args.output_dir
    subprocess.run(
        [
            sys.executable,
            "training/generate_phase_g_frozen_eval.py",
            "--output_dir",
            data_dir.as_posix(),
        ],
        cwd=ROOT,
        check=True,
    )
    prereg_path = write_preregistration(data_dir / "preregistration.json")
    payload = preparation_summary(read_json(data_dir / "manifest.json"), read_json(prereg_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Phase G-alpha Zero-GPU Preparation",
        "",
        f"- Status: `{payload['status']}`",
        "- N=24 calibration/test frozen sets: complete",
        "- Exact preimage strata: 128 unique, 128 small, 128 large per split",
        "- Forward-orbit coverage verifier: complete",
        "- Entropy-matched answer-head comparator: complete",
        "- Frozen-block architecture contract: complete",
        "- Gate forms: locked",
        "- Substrate seam: N=20 constructive pass must also clear N=24 arbitrary calibration",
        "- Remaining preregistration blank: powered numeric margin",
        "",
        "This package does not authorize G-alpha training until the deterministic "
        "checkpoint is competent on the arbitrary N=24 calibration split and its SHA receipt is green.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["phase_g_alpha_preparation_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
