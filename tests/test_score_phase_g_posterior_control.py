from __future__ import annotations

from eval.score_phase_g_posterior_control import score_posterior_control


def audit(
    *,
    teacher_rate: float,
    prior_rate: float,
    teacher_distinct: float,
    prior_distinct: float,
    helped: int = 32,
    hurt: int = 4,
    switching_groups: int = 48,
) -> dict:
    fidelity = {
        "prior": {
            "target_in_k_rate": prior_rate,
            "validity_rate": 0.80,
        },
        "posterior_teacher": {
            "target_in_k_rate": teacher_rate,
            "validity_rate": 0.78,
        },
        "paired_target_in_k": {
            "helped": helped,
            "hurt": hurt,
            "tied": 64 - helped - hurt,
        },
    }
    return {
        "posterior_target_fidelity": {"by_k": {"1": fidelity}},
        "posterior_target_conditioning": {
            "multi_target_groups": 64,
            "posterior_teacher": {
                "mean_distinct_first_predictions": teacher_distinct,
                "mean_group_selected_target_rate": teacher_rate,
                "groups_with_at_least_two_first_predictions": switching_groups,
            },
            "prior": {
                "mean_distinct_first_predictions": prior_distinct,
                "mean_group_selected_target_rate": prior_rate,
                "groups_with_at_least_two_first_predictions": 0,
            },
            "paired_group_selected_target_rate": {
                "helped": helped,
                "hurt": hurt,
                "tied": 64 - helped - hurt,
                "posterior_minus_prior_mean": teacher_rate - prior_rate,
            },
            "K1_validity_sanity": {
                "prior": 0.8,
                "posterior_teacher": 0.78,
                "posterior_minus_prior": -0.02,
            },
        },
    }


def score(payload: dict) -> dict:
    return score_posterior_control(
        payload,
        min_multi_target_groups=32,
        min_teacher_target_rate=0.60,
        min_teacher_prior_target_lift=0.15,
        min_teacher_switching_groups=24,
        max_teacher_prior_target_lift_p_value=0.05,
    )


def test_posterior_control_gate_passes_only_with_conditioned_target_signal() -> None:
    result = score(audit(teacher_rate=0.8, prior_rate=0.5, teacher_distinct=2.0, prior_distinct=1.0))

    assert result["status"] == "passed"
    assert result["next_step"] == "authorize_one_matched_coverage_rerun"


def test_posterior_control_gate_blocks_when_teacher_is_target_invariant() -> None:
    result = score(
        audit(
            teacher_rate=0.8,
            prior_rate=0.5,
            teacher_distinct=1.0,
            prior_distinct=1.0,
            switching_groups=0,
        )
    )

    assert result["status"] == "blocked"
    assert result["checks"]["teacher_switching_groups"]["passed"] is False


def test_posterior_control_gate_blocks_when_lift_lacks_paired_support() -> None:
    result = score(
        audit(
            teacher_rate=0.8,
            prior_rate=0.5,
            teacher_distinct=2.0,
            prior_distinct=1.0,
            helped=3,
            hurt=2,
        )
    )

    assert result["status"] == "blocked"
    assert result["checks"]["teacher_minus_prior_selected_target_sign_test"]["passed"] is False
