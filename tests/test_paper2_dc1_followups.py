import torch

from training.paper2_dc1_followups import (
    STAGE_A_RESOURCE_PROPOSAL,
    floor_payload_has_all_positions,
    scale_response_reading,
    score_stage_a_verdict,
    transition_ledger,
)


def test_transition_ledger_reports_full_and_depth1_splits() -> None:
    predictions = torch.tensor([[1, 1, 2], [2, 1, 1], [1, 2, 1], [2, 2, 2]])
    teacher = torch.tensor([1, 1, 1, 1])
    result = transition_ledger(predictions, teacher, before_depth=1, after_depth=2)

    assert result["all_positions"]["helps"] == 1
    assert result["all_positions"]["hurts"] == 1
    assert result["depth1_accepted"]["hurts"] == 1
    assert result["depth1_rejected"]["helps"] == 1


def test_floor_coverage_requires_more_than_rejected_only_rows() -> None:
    rejected = {"all_position_rows": [{"predictions": [1, 2]}] * 4}
    full = {"all_position_rows": [{"predictions": [1, 2]}] * 5}
    assert not floor_payload_has_all_positions(rejected, rejected_positions=4)
    assert floor_payload_has_all_positions(full, rejected_positions=4)


def test_scale_response_reading_compares_trough_with_cosine_crossover() -> None:
    rows = [
        {
            "label": "small",
            "transition": {"after_accuracy": 0.3},
            "cosine_to_fed": {"mean": 0.1},
            "cosine_to_k0": {"mean": 0.8},
        },
        {
            "label": "middle",
            "transition": {"after_accuracy": 0.1},
            "cosine_to_fed": {"mean": 0.45},
            "cosine_to_k0": {"mean": 0.50},
        },
        {
            "label": "raw",
            "transition": {"after_accuracy": 0.6},
            "cosine_to_fed": {"mean": 0.9},
            "cosine_to_k0": {"mean": 0.4},
        },
    ]
    result = scale_response_reading(rows)
    assert result["trough_coincides_with_nearest_measured_crossover"] is True
    assert result["fed_cosine_non_decreasing_pairs"] == 2


def test_stage_a_verdict_bands_are_exact() -> None:
    assert score_stage_a_verdict(
        trained_helps=20,
        trained_hurts=19,
        untrained_helps=10,
        untrained_hurts=100,
        positions=1000,
        row_cluster_bootstrap_net_ci95_lower=-2.0,
    )["verdict"] == "qualifies"
    assert score_stage_a_verdict(
        trained_helps=10,
        trained_hurts=40,
        untrained_helps=10,
        untrained_hurts=100,
        positions=1000,
        row_cluster_bootstrap_net_ci95_lower=-40.0,
    )["verdict"] == "partial_domestication"
    assert score_stage_a_verdict(
        trained_helps=10,
        trained_hurts=60,
        untrained_helps=10,
        untrained_hurts=100,
        positions=1000,
        row_cluster_bootstrap_net_ci95_lower=-60.0,
    )["verdict"] == "no_material_improvement"


def test_stage_a_resource_proposal_is_bounded_and_not_authority() -> None:
    assert STAGE_A_RESOURCE_PROPOSAL["step_ceiling"] <= 2000
    assert STAGE_A_RESOURCE_PROPOSAL["microbatch_rows"] == 1
    assert STAGE_A_RESOURCE_PROPOSAL["gradient_accumulation_steps"] == 1
    assert STAGE_A_RESOURCE_PROPOSAL["learning_rate"] == 1e-4
    assert STAGE_A_RESOURCE_PROPOSAL["status"] == "proposal_for_preregistration_not_training_authority"
