from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "part1_claim_evidence_ledger.json"


def load_ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def test_part1_ledger_has_closed_scope_and_open_width_claim() -> None:
    payload = load_ledger()

    assert payload["program"] == "part1_deterministic_recurrent_qwen"
    assert payload["status"] == "closed"
    claims = {claim["id"]: claim for claim in payload["claims"]}
    assert claims["phase_a_synthetic_surpass"]["status"] == "supported_bounded"
    assert claims["cross_support_frontier_law"]["status"] == "supported_bounded"
    assert claims["natural_surface_tail_inversion"]["status"] == "supported_bounded"
    assert claims["branching_width_substrate"]["status"] == "supported_gate_pass"
    assert claims["guided_latent_width"]["status"] == "open"
    assert claims["general_natural_reasoning_superiority"]["status"] == "not_supported"


def test_every_claim_evidence_path_exists() -> None:
    payload = load_ledger()

    missing: list[str] = []
    for claim in payload["claims"]:
        for evidence in claim.get("evidence", []):
            path = ROOT / evidence["path"]
            if not path.exists():
                missing.append(f"{claim['id']}: {evidence['path']}")
    assert missing == []


def test_phase_a_receipt_matches_ledger_arithmetic() -> None:
    payload = load_ledger()
    claims = {claim["id"]: claim for claim in payload["claims"]}
    claim = claims["phase_a_synthetic_surpass"]
    receipt_path = ROOT / claim["evidence"][0]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    counts = receipt["scoring"]["counts"]
    expected = claim["metrics"]
    assert sum(counts["A"].values()) == expected["recurrent_correct"] == 1506
    assert sum(counts["B_step4000"].values()) == expected["dense_direct_correct"] == 470
    assert sum(counts["C_step4000"].values()) == expected["dense_scratchpad_correct"] == 952
    assert sum(counts["D_step4000"].values()) == expected["dense_1_5b_correct"] == 322
    assert receipt["scoring"]["rows"] == expected["rows"] == 1792
    assert receipt["scoring"]["row_ids_match"] is True


def test_branching_gate_receipt_matches_ledger() -> None:
    payload = load_ledger()
    claims = {claim["id"]: claim for claim in payload["claims"]}
    claim = claims["branching_width_substrate"]
    receipt = json.loads((ROOT / claim["evidence"][0]["path"]).read_text(encoding="utf-8"))

    natural = receipt["branching_screens"]["natural_step2000_N20_verbal"]["gate"]
    symbolic = receipt["branching_screens"]["n24_step6000_N24_symbolic"]["gate"]
    assert natural["passed"] is True
    assert natural["pooled_correct"] == claim["metrics"]["natural_n20_correct"] == 389
    assert symbolic["passed"] is False
    assert symbolic["pooled_correct"] == claim["metrics"]["symbolic_n24_correct"] == 355
    assert receipt["phase_g_alpha_status"] == "ready_for_powered_margin_lock_then_launch"


def test_manuscript_rejects_prohibited_generalizations() -> None:
    payload = load_ledger()
    manuscript = (ROOT / payload["canonical_manuscript"]).read_text(encoding="utf-8")

    for phrase in payload["global_prohibited_phrases"]:
        assert phrase.lower() not in manuscript.lower()
    assert "synthetic-family" in manuscript
    assert "Guided stochastic width remains open" in manuscript


def test_manuscript_v2_artifact_map_paths_exist() -> None:
    payload = load_ledger()
    missing = [
        path
        for path in payload["manuscript_v2_artifact_map"].values()
        if not (ROOT / path).exists()
    ]
    assert missing == []
