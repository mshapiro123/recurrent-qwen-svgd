from __future__ import annotations

import inspect

import torch

from training import run_paper2_phase3_p33 as runner


def test_runner_binds_exact_e2_receipts_and_checkpoint_lineage() -> None:
    assert runner.EXPECTED_PREFLIGHT_SHA256 == (
        "9a71e3e59526383b3dd830a320a0e18ad3778571f67dac1e262ee2713ea0ffd0"
    )
    assert runner.EXPECTED_CALIBRATION_SHA256 == (
        "e46198291bdea16f3561b44eaa1a77764aa7a0fcc49a60c4c58802491aef985c"
    )
    assert set(runner.EXPECTED_MIGRATED_SHA256) == {0, 1}
    assert runner.EXPECTED_PHASE3_CONFIGURATION == 1_185_973
    assert runner.EXPECTED_OPTIMIZER_MARKED == 280_880
    assert runner.P33_TOTAL_STEPS == 1000
    assert runner.P33_LOOK_INTERVAL == 50
    assert runner.P33_LOOKS == 20


def test_guardrail_is_init_relative_token_retention_only() -> None:
    calibration = {
        "tier_s": {"one_sided_alpha": 0.05, "decision_margin_relative_to_init": -0.006},
        "tier_w": {"one_sided_alpha": 0.10, "decision_margin_relative_to_init": -0.001},
    }
    result = runner.guardrail_read(retained=[True] * 1024, calibration=calibration)
    assert result["mean_difference_from_init"] == 0.0
    assert not result["tier_s"]["condition_met"]
    assert not result["tier_w"]["condition_met"]


def test_runner_includes_required_measurement_surfaces() -> None:
    source = inspect.getsource(runner.run)
    assert "instrumentation_nonperturbation" in source
    assert "tier1_observatory_read" in source
    assert "a_state_intervention_battery" in source
    assert '"task_level_capability_scoring": False' in source
    assert '"confirm_scored": False' in source
    assert '"eval_e_scored": False' in source


def test_observatory_defines_gradient_dot_write_without_task_generation() -> None:
    source = inspect.getsource(runner.tier1_observatory_read)
    assert "loss_gradient=loss_gradient" in source
    assert "teacher_14b_top1" in source
    assert "student_top1" in source
    assert "generate(" not in source


def test_p33_population_cache_excludes_unused_teacher_state_surface() -> None:
    source = inspect.getsource(runner.build_p33_population_cache)
    assert '_parallel_receipts(summary, "student_0p5b")' in source
    assert '_parallel_receipts(summary, "teacher_14b")' not in source
    assert '"teacher_state_materialized": False' in source


def test_tensor_digest_is_stable() -> None:
    values = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    assert runner.tensor_digest(values) == runner.tensor_digest(dict(reversed(list(values.items()))))


def test_batch_maps_source_local_anchors_into_concatenated_cache() -> None:
    cache = {
        "documents": ["old-0", "old-1", "new-0", "new-1", "new-2"],
        "source_anchor_offsets": {"old": 0, "new": 2},
        "positions": torch.tensor([1, 2, 3, 4, 5]),
        "student_hidden": torch.arange(5 * 4 * 896).reshape(5, 4, 896),
        "candidate_ids": torch.zeros((5, 4, 1), dtype=torch.long),
        "candidate_mask": torch.ones((5, 4, 1), dtype=torch.bool),
        "base_log_probs": torch.zeros((5, 4, 1)),
        "base_tail": torch.zeros((5, 4)),
    }
    records = [
        {"source": "old", "anchor_index": 1, "horizon": 1, "gate_label": 0},
        {"source": "new", "anchor_index": 1, "horizon": 1, "gate_label": 0},
    ]
    batch = runner._batch(
        cache=cache,
        records=records,
        direction_index={},
        directions=torch.empty((0, 896)),
        device="cpu",
    )
    assert torch.equal(batch["hidden4"][0], cache["student_hidden"][1])
    assert torch.equal(batch["hidden4"][1], cache["student_hidden"][3])
