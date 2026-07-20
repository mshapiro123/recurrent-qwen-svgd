from __future__ import annotations

import pytest

from training.oracle_intrablock_control_spec import (
    preregistration_payload,
    score_oracle_intrablock_control,
)


def _arm(*, passed: bool) -> dict:
    return {
        "kind": "phase_g_oracle_interface_arm",
        "route": "layerwise_film",
        "passed": passed,
        "status": "passed" if passed else "blocked",
        "checks": {},
    }


def test_preregistration_changes_only_command_access_location() -> None:
    payload = preregistration_payload()

    assert payload["terminal_probe"]
    assert payload["arms"] == ["single_entry_film_control", "layerwise_film"]
    assert payload["only_variable"] == "command_access_location"
    assert payload["keeper"]["frozen"]
    assert payload["conditioner"]["shared_across_recurrent_layers"]
    assert payload["conditioner"]["parameter_matched_to_single_entry_film"]
    assert "coverage" in payload["deferred"]


def test_layerwise_pass_reopens_variational_design_without_running_it() -> None:
    result = score_oracle_intrablock_control(_arm(passed=True))

    assert result["measured_reading"] == "DISTRIBUTED_INTERFACE_CONTROLS"
    assert result["interpretation"] == "single_entry_access_was_the_binding_constraint"
    assert not result["automatic_successor_authorized"]


def test_layerwise_failure_closes_frozen_conditioning_more_strongly() -> None:
    result = score_oracle_intrablock_control(_arm(passed=False))

    assert result["measured_reading"] == "DISTRIBUTED_INTERFACE_FAILS"
    assert result["interpretation"] == "frozen_substrate_not_oracle_controllable_under_tested_small_interfaces"
    assert not result["automatic_successor_authorized"]


def test_unknown_route_is_rejected() -> None:
    with pytest.raises(AssertionError, match="layerwise_film"):
        score_oracle_intrablock_control(
            {
                **_arm(passed=True),
                "route": "additive",
            }
        )
