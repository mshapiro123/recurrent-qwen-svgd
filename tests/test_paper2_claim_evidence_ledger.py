from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "paper2_claim_evidence_ledger.json"


def load_ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def test_paper2_ledger_closes_phase_g_and_records_t1_replication_state() -> None:
    payload = load_ledger()
    claims = {claim["id"]: claim for claim in payload["claims"]}

    assert payload["program"] == "paper2_causal_control_of_recurrent_computation"
    assert claims["guided_width_initial_exploratory"]["status"] == (
        "exploratory_non_identifying"
    )
    assert claims["posterior_target_control_negative"]["status"] == (
        "registered_negative"
    )
    assert claims["forced_injection_no_channel"]["metrics"]["registered_verdict"] == (
        "NO-CHANNEL"
    )
    assert claims["terminal_oracle_reentry_both_fail"]["metrics"][
        "registered_verdict"
    ] == "BOTH_FAIL"
    assert claims["terminal_oracle_train_fit_diagnostic"]["metrics"][
        "film_full_nondefault_control"
    ] < 0.25
    assert claims["internal_token_halting"]["status"] == (
        "two_registered_negatives_replicated_exact_raw_control_bounded_preservation_cost"
    )
    assert claims["internal_token_halting"]["metrics"][
        "all_five_t0_contracts_passed"
    ] is True
    assert claims["internal_token_halting"]["metrics"]["t0_training_performed"] is False
    assert claims["internal_token_halting"]["metrics"]["t1_seed0_training_performed"] is True
    assert claims["internal_token_halting"]["metrics"]["seed1_training_performed"] is True
    assert claims["internal_token_halting"]["metrics"]["seed1_raw_forced_correct"] == 971
    assert claims["internal_token_halting"]["metrics"][
        "seed1_raw_exact_selection_correct"
    ] == 1024
    assert claims["internal_token_halting"]["metrics"][
        "p0_lineage_matches_registered_t1"
    ] is False
    assert claims["speculative_depth_d0"]["status"] == "not_recoverable_at_pilot_scale"
    assert claims["speculative_depth_d0"]["metrics"]["accepted_position_net_loss"] == 4928
    assert claims["speculative_depth_d0"]["metrics"]["t1_retention_correct"] == 1005
    assert payload["active_queue"]["t1"].startswith("two_registered_negatives")
    assert claims["coconut_composite_integrity"]["status"] == (
        "engineering_preflight_bounded_fail"
    )


def test_paper2_claim_evidence_paths_exist() -> None:
    missing = []
    for claim in load_ledger()["claims"]:
        for evidence in claim.get("evidence", []):
            if not (ROOT / evidence["path"]).exists():
                missing.append(f"{claim['id']}: {evidence['path']}")
    assert missing == []


def test_status_and_log_no_longer_leave_phase_g_live() -> None:
    status = (ROOT / "docs" / "PROJECT_STATUS_PAPER.md").read_text(encoding="utf-8")
    log = (ROOT / "docs" / "EXPERIMENT_LOG.md").read_text(encoding="utf-8")

    assert "Arm G is closed on the tested frozen re-entry interface" in status
    assert "NO-CHANNEL" in status
    assert "BOTH_FAIL" in status
    assert "Train only the Phase G prior head" not in status
    assert "Paper Two record reconciliation" in log
