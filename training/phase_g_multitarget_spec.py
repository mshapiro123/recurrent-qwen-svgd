"""Pre-launch contracts for the repeated-prompt Phase G correction."""

from __future__ import annotations

from typing import Any, Mapping

from training.phase_g_multitarget_task import validate_multitarget_rows


def required_posterior_control_thresholds(
    environment: Mapping[str, str],
) -> dict[str, float | int]:
    """Require posterior-control thresholds before a GPU training launch."""

    declarations: tuple[tuple[str, type[int] | type[float]], ...] = (
        ("STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS", int),
        ("STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_TARGET_RATE", float),
        ("STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_PRIOR_TARGET_LIFT", float),
        ("STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_PRIOR_DISTINCT_LIFT", float),
    )
    values: dict[str, float | int] = {}
    missing: list[str] = []
    for name, parser in declarations:
        raw = environment.get(name, "").strip()
        if not raw:
            missing.append(name)
            continue
        values[name] = parser(raw)
    if missing:
        raise RuntimeError(
            "Phase G multi-target posterior-control thresholds must be locked "
            "before training. Missing: " + ", ".join(missing)
        )
    if int(values["STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS"]) < 1:
        raise ValueError("STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS must be positive")
    for name, value in values.items():
        if name != "STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS" and float(value) < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    return values


def assert_multitarget_curriculum(
    rows: list[dict[str, Any]],
    *,
    require_all_reachable_targets: bool,
) -> dict[str, Any]:
    """Require actual same-prompt, different-target exposure before training."""

    report = validate_multitarget_rows(rows)
    if report["status"] != "passed":
        raise AssertionError(
            "Invalid Phase G multi-target curriculum: "
            + "; ".join(report["errors"][:10])
        )
    if report["base_problem_groups"] < 1:
        raise AssertionError("Phase G multi-target curriculum has no base problems")
    if report["groups_with_multiple_targets"] != report["base_problem_groups"]:
        raise AssertionError(
            "Every base prompt must expose at least two distinct valid target chains"
        )
    if require_all_reachable_targets and not report["all_reachable_targets_covered"]:
        raise AssertionError(
            "The locked curriculum requires every reachable terminal target per prompt"
        )
    return report


def preregistration_payload() -> dict[str, Any]:
    """Machine-readable gate forms for the curriculum-identifiability repair."""

    return {
        "kind": "phase_g_multitarget_correction_preregistration",
        "status": "forms_locked_before_multitarget_training",
        "keeper": "same_frozen_phase_g_alpha_keeper_with_exact_sha256_assertion",
        "trainable": [
            "phase_g_prior_head.*",
            "phase_g_posterior_head.*",
            "phase_g_injection_scale",
        ],
        "curriculum": {
            "same_prompt_multiple_targets": "required",
            "target_chain": "one valid exact-depth chain per terminal target variant",
            "sampling_policy": "base_problem_uniform_then_target_variant_uniform",
            "full_reachable_support": "required for the initial correction run",
        },
        "gate_order": [
            "posterior_exact_target_control_on_repeated_prompt_holdout",
            "K1_deterministic_preservation",
            "prior_coverage_vs_entropy_matched_temperature",
            "prior_coverage_vs_iso_compute_depth",
        ],
        "posterior_control_form": {
            "metric": "K1 exact selected-target rate and group-level target switching",
            "comparison": "posterior_teacher versus prior on identical repeated-prompt rows",
            "numeric_margin": "lock_from_holdout_power_calculation_before_training",
        },
        "coverage_form": {
            "metric": "paired exact oracle coverage at K",
            "comparators": ["entropy_matched_temperature", "iso_compute_depth"],
            "reuse": "original_frozen_coverage_rows_and_reader",
        },
        "deferred": ["G_beta", "selector", "per_trajectory_halting", "SVGD"],
    }
