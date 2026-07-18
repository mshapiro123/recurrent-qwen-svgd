"""Score the preregistered repeated-prompt Phase G posterior-control gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score_posterior_control(
    audit: dict[str, Any],
    *,
    min_multi_target_groups: int,
    min_teacher_target_rate: float,
    min_teacher_prior_target_lift: float,
    min_teacher_prior_distinct_prediction_lift: float,
) -> dict[str, Any]:
    """Apply thresholds that were supplied before the guidance training run.

    The audit is deliberately evaluated at K=1. Oracle coverage is not a
    substitute for target conditioning: every selected target is valid for the
    prompt, so coverage alone cannot reveal whether the posterior responds to
    the selected chain.
    """

    conditioning = audit["posterior_target_conditioning"]
    fidelity = audit["posterior_target_fidelity"]["by_k"]["1"]
    teacher_target_rate = float(fidelity["posterior_teacher"]["target_in_k_rate"])
    prior_target_rate = float(fidelity["prior"]["target_in_k_rate"])
    teacher_prior_target_lift = teacher_target_rate - prior_target_rate
    teacher_distinct = float(
        conditioning["posterior_teacher"]["mean_distinct_first_predictions"]
    )
    prior_distinct = float(conditioning["prior"]["mean_distinct_first_predictions"])
    teacher_prior_distinct_lift = teacher_distinct - prior_distinct
    multi_target_groups = int(conditioning["multi_target_groups"])

    checks = {
        "repeated_prompt_groups": {
            "observed": multi_target_groups,
            "minimum": min_multi_target_groups,
            "passed": multi_target_groups >= min_multi_target_groups,
        },
        "teacher_selected_target_rate": {
            "observed": teacher_target_rate,
            "minimum": min_teacher_target_rate,
            "passed": teacher_target_rate >= min_teacher_target_rate,
        },
        "teacher_minus_prior_selected_target_rate": {
            "observed": teacher_prior_target_lift,
            "minimum": min_teacher_prior_target_lift,
            "passed": teacher_prior_target_lift >= min_teacher_prior_target_lift,
        },
        "teacher_minus_prior_distinct_first_predictions": {
            "observed": teacher_prior_distinct_lift,
            "minimum": min_teacher_prior_distinct_prediction_lift,
            "passed": (
                teacher_prior_distinct_lift
                >= min_teacher_prior_distinct_prediction_lift
            ),
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "kind": "phase_g_multitarget_posterior_control_gate",
        "status": "passed" if passed else "blocked",
        "gate_type": "posterior_control_before_coverage",
        "K": 1,
        "checks": checks,
        "interpretation": (
            "posterior_target_control_present"
            if passed
            else "posterior_target_control_not_established"
        ),
        "next_step": (
            "authorize_one_matched_coverage_rerun"
            if passed
            else "close_guided_width_correction_without_coverage_rerun"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--min_multi_target_groups", type=int, required=True)
    parser.add_argument("--min_teacher_target_rate", type=float, required=True)
    parser.add_argument("--min_teacher_prior_target_lift", type=float, required=True)
    parser.add_argument(
        "--min_teacher_prior_distinct_prediction_lift",
        type=float,
        required=True,
    )
    args = parser.parse_args()
    if args.min_multi_target_groups < 1:
        raise ValueError("min_multi_target_groups must be positive")
    for name in (
        "min_teacher_target_rate",
        "min_teacher_prior_target_lift",
        "min_teacher_prior_distinct_prediction_lift",
    ):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")

    result = score_posterior_control(
        read_json(args.audit_json),
        min_multi_target_groups=args.min_multi_target_groups,
        min_teacher_target_rate=args.min_teacher_target_rate,
        min_teacher_prior_target_lift=args.min_teacher_prior_target_lift,
        min_teacher_prior_distinct_prediction_lift=(
            args.min_teacher_prior_distinct_prediction_lift
        ),
    )
    write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
