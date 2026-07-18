from __future__ import annotations

from eval.score_phase_g_posterior_control import score_posterior_control


def audit(*, teacher_rate: float, prior_rate: float, teacher_distinct: float, prior_distinct: float) -> dict:
    return {
        "posterior_target_fidelity": {
            "by_k": {
                "1": {
                    "posterior_teacher": {"target_in_k_rate": teacher_rate},
                    "prior": {"target_in_k_rate": prior_rate},
                }
            }
        },
        "posterior_target_conditioning": {
            "multi_target_groups": 64,
            "posterior_teacher": {
                "mean_distinct_first_predictions": teacher_distinct,
            },
            "prior": {"mean_distinct_first_predictions": prior_distinct},
        },
    }


def score(payload: dict) -> dict:
    return score_posterior_control(
        payload,
        min_multi_target_groups=64,
        min_teacher_target_rate=0.7,
        min_teacher_prior_target_lift=0.2,
        min_teacher_prior_distinct_prediction_lift=0.4,
    )


def test_posterior_control_gate_passes_only_with_conditioned_target_signal() -> None:
    result = score(audit(teacher_rate=0.8, prior_rate=0.5, teacher_distinct=2.0, prior_distinct=1.0))

    assert result["status"] == "passed"
    assert result["next_step"] == "authorize_one_matched_coverage_rerun"


def test_posterior_control_gate_blocks_when_teacher_is_target_invariant() -> None:
    result = score(audit(teacher_rate=0.8, prior_rate=0.5, teacher_distinct=1.0, prior_distinct=1.0))

    assert result["status"] == "blocked"
    assert result["checks"]["teacher_minus_prior_distinct_first_predictions"]["passed"] is False
