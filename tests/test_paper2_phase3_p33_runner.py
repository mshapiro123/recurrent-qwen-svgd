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


def test_tensor_digest_is_stable() -> None:
    values = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    assert runner.tensor_digest(values) == runner.tensor_digest(dict(reversed(list(values.items()))))
