from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "paper2_claim_evidence_ledger.json"


def load_ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def test_paper2_ledger_closes_tested_phase_g_route_and_records_t0_without_t1() -> None:
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
    assert claims["internal_token_halting"]["status"] == "t0_passed_t1_not_run"
    assert claims["internal_token_halting"]["metrics"][
        "all_five_t0_contracts_passed"
    ] is True
    assert claims["internal_token_halting"]["metrics"]["training_performed"] is False


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

    assert "Arm G closed on the tested frozen re-entry interface" in status
    assert "NO-CHANNEL" in status
    assert "BOTH_FAIL" in status
    assert "Train only the Phase G prior head" not in status
    assert "Paper Two record reconciliation" in log
