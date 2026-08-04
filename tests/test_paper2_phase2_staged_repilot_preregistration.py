from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _registration() -> dict:
    return json.loads(
        (ROOT / "training/paper2_phase2_staged_repilot_preregistration.json").read_text(
            encoding="utf-8"
        )
    )


def test_staged_repilot_is_locked_and_a1_precedes_a2() -> None:
    registration = _registration()
    assert registration["status"] == "locked_before_training"
    assert registration["alpha"] == 0.5
    assert registration["seeds"] == [0, 1]
    assert registration["calibration"]["optimizer_updates"] == 0
    assert registration["a1"]["execution_gates_closed"] is True
    assert registration["a2"]["authorized_after_a1_strategy_gate_only"] is True


def test_staged_repilot_gradient_shares_and_budgets_are_fully_specified() -> None:
    registration = _registration()
    for stage in ("a1", "a2"):
        shares = registration[stage]["target_gradient_shares"]
        assert abs(sum(shares.values()) - 1.0) < 1e-12
        assert registration[stage]["nominal_steps"] == 1000
        assert registration[stage]["maximum_steps"] == 2000
    assert registration["a1"]["target_gradient_shares"]["flow"] == 0.60
    assert registration["a1"]["target_gradient_shares"]["functional_probe_kl"] <= 0.25
    assert registration["a2"]["oracle_headroom_minimum_relative"] == 0.02


def test_staged_repilot_shapers_are_observation_mode() -> None:
    registration = _registration()
    assert registration["trust"]["penalty_weight"] == 0.0
    assert registration["trust"]["endpoint_ratio_tripwire"] == 5.0
    assert registration["clip"]["ceiling_multiplier_over_calibration_p99"] == 10.0
    assert registration["clip"]["alarm_only"] is True


def test_staged_protocol_has_no_open_lock_blockers() -> None:
    protocol = (
        ROOT / "docs/PAPER2_PHASE2_STAGED_REPILOT_PROTOCOL_DRAFT_20260805.md"
    ).read_text(encoding="utf-8")
    assert "Status: `locked_before_training`" in protocol
    assert "[LOCK-BLOCKER]" not in protocol
    assert "A1 launcher contains no path that can enter A2" in protocol
