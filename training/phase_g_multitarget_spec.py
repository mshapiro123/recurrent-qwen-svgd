"""Pre-launch contracts for the repeated-prompt Phase G correction."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from training.branching_relations_task import row_manifest
from training.phase_g_multitarget_task import validate_multitarget_rows


def required_posterior_control_thresholds(
    environment: Mapping[str, str],
) -> dict[str, float | int]:
    """Require posterior-control thresholds before a GPU training launch."""

    declarations: tuple[tuple[str, type[int] | type[float]], ...] = (
        ("STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS", int),
        ("STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_TARGET_RATE", float),
        ("STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_PRIOR_TARGET_LIFT", float),
        ("STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_SWITCHING_GROUPS", int),
        ("STAGE5_PHASE_G_MULTITARGET_MAX_TEACHER_PRIOR_TARGET_LIFT_PVALUE", float),
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
    for name in (
        "STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS",
        "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_SWITCHING_GROUPS",
    ):
        if int(values[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if int(values["STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_SWITCHING_GROUPS"]) > int(
        values["STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS"]
    ):
        raise ValueError("Teacher switching-group minimum cannot exceed total group minimum")
    for name, value in values.items():
        if name not in {
            "STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS",
            "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_SWITCHING_GROUPS",
        } and float(value) < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    p_value = float(values["STAGE5_PHASE_G_MULTITARGET_MAX_TEACHER_PRIOR_TARGET_LIFT_PVALUE"])
    if not 0.0 < p_value <= 1.0:
        raise ValueError(
            "STAGE5_PHASE_G_MULTITARGET_MAX_TEACHER_PRIOR_TARGET_LIFT_PVALUE "
            "must be in (0, 1]"
        )
    return values


def frozen_gradient_assertion_count(training_summary: Mapping[str, Any]) -> int:
    """Read the training-owned frozen-gradient receipt from its canonical field."""

    config = training_summary.get("config")
    if not isinstance(config, Mapping):
        raise AssertionError("Phase G training summary lacks its config receipt")
    value = config.get("frozen_gradient_assertions")
    if not isinstance(value, int) or value < 0:
        raise AssertionError(
            "Phase G training config lacks a nonnegative frozen_gradient_assertions receipt"
        )
    return value


def resolve_posterior_control_gate_lock_path(
    root: str | Path,
    raw_path: str | Path | None,
) -> Path:
    """Resolve the required gate-lock receipt before any GPU setup begins."""

    if raw_path is None or not str(raw_path).strip():
        raise RuntimeError(
            "STAGE5_PHASE_G_MULTITARGET_GATE_LOCK must identify a committed "
            "pre-training posterior-control gate-lock JSON"
        )
    root = Path(root)
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase G posterior-control gate lock: {path}")
    return path


def posterior_control_surface_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe the exact held-out repeated-prompt surface used by the gate."""

    validation = assert_multitarget_curriculum(
        rows,
        require_all_reachable_targets=True,
    )
    target_counts = Counter(int(row["target_variant_count"]) for row in rows)
    by_problem: dict[str, int] = {}
    for row in rows:
        base_problem_id = str(row["base_problem_id"])
        by_problem[base_problem_id] = by_problem.get(base_problem_id, 0) + 1
    return {
        "row_manifest": row_manifest(rows),
        "base_problem_groups": validation["base_problem_groups"],
        "groups_with_multiple_targets": validation["groups_with_multiple_targets"],
        "target_variant_count_histogram": {
            str(count): frequency
            for count, frequency in sorted(Counter(by_problem.values()).items())
        },
        "row_target_variant_count_histogram": {
            str(count): frequency for count, frequency in sorted(target_counts.items())
        },
    }


def build_posterior_control_gate_lock(
    rows: list[dict[str, Any]],
    thresholds: Mapping[str, float | int],
) -> dict[str, Any]:
    """Create a reviewable threshold lock bound to exact frozen control rows."""

    locked_thresholds = required_posterior_control_thresholds(
        {name: str(value) for name, value in thresholds.items()}
    )
    return {
        "kind": "phase_g_multitarget_posterior_control_gate_lock",
        "status": "locked_before_multitarget_training",
        "surface": posterior_control_surface_manifest(rows),
        "thresholds": locked_thresholds,
        "gate_order": [
            "posterior_control",
            "K1_deterministic_preservation",
            "coverage_vs_temperature",
            "coverage_vs_iso_compute_depth",
        ],
    }


def assert_posterior_control_gate_lock(
    gate_lock: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Reject a gate lock that was made for a different or invalid control set."""

    if gate_lock.get("kind") != "phase_g_multitarget_posterior_control_gate_lock":
        raise AssertionError("Unexpected Phase G posterior-control gate-lock kind")
    if gate_lock.get("status") != "locked_before_multitarget_training":
        raise AssertionError("Phase G posterior-control gate lock is not marked pre-training")
    actual_surface = posterior_control_surface_manifest(rows)
    if gate_lock.get("surface") != actual_surface:
        raise AssertionError("Phase G posterior-control rows do not match the locked surface")
    thresholds = gate_lock.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise AssertionError("Phase G posterior-control gate lock lacks thresholds")
    return required_posterior_control_thresholds(
        {name: str(value) for name, value in thresholds.items()}
    )


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
