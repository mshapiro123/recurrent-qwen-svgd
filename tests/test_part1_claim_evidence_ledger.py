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
    assert claims["peft_installation_measured"]["status"] == "supported_bounded"
    assert claims["depth_selection_bounded_negative"]["status"] == "registered_negative"
    assert claims["general_capability_preservation"]["status"] == "supported_bounded"
    assert claims["adapter_persistence"]["status"] == "supported"
    assert claims["adapter_zero_shot_transfer_minimal"]["status"] == "supported_bounded"
    assert claims["adapter_retention_joint_pass"]["status"] == "not_supported"
    assert claims["adapter_verbal_transference"]["status"] == "supported_bounded"


def test_peft_and_selector_closure_use_strategy_locked_accounting() -> None:
    claims = {claim["id"]: claim for claim in load_ledger()["claims"]}
    peft = claims["peft_installation_measured"]
    selector = claims["depth_selection_bounded_negative"]

    assert peft["metrics"]["optimizer_marked_parameters"] == 7_613_953
    assert peft["metrics"]["forward_active_parameters"] == 6_007_425
    assert peft["metrics"]["bridge_legacy_concat_parameters_bypassed"] == 1_606_528
    assert "underpowered to claim parity" in peft["metrics"]["parity_inference"]
    assert selector["metrics"]["forced_diagonal_correct"] == 759
    assert selector["metrics"]["s2_selection_histogram"] == {"12": 768}
    assert "starved information path" in selector["interpretation"]


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
    receipt = json.loads((ROOT / claim["evidence"][0]["path"]).read_text(encoding="utf-8"))
    audit = json.loads((ROOT / claim["evidence"][1]["path"]).read_text(encoding="utf-8"))

    counts = receipt["scoring"]["counts"]
    expected = claim["metrics"]
    assert sum(counts["A"].values()) == expected["recurrent_correct"] == 1506
    assert audit["arms"]["B_step4000"]["corrected_correct"] == expected["dense_direct_correct"] == 496
    assert audit["arms"]["C_step4000"]["corrected_correct"] == expected["dense_scratchpad_correct"] == 1292
    assert audit["arms"]["D_step4000"]["corrected_correct"] == expected["dense_1_5b_correct"] == 656
    paired = audit["arms"]["C_step4000"]["paired_against_full_block_recurrent"]
    assert paired["recurrent_helped"] == expected["recurrent_vs_scratchpad_helped"] == 262
    assert paired["recurrent_hurt"] == expected["recurrent_vs_scratchpad_hurt"] == 48
    primary = audit["paper_one_audit"]["a_vs_b_corrected_paired"]
    gate = audit["paper_one_audit"]["a_vs_b_preregistered_count_gate"]
    assert primary["left_only"] == expected["recurrent_vs_direct_helped"] == 1048
    assert primary["right_only"] == expected["recurrent_vs_direct_hurt"] == 38
    assert gate["pass"] is expected["preregistered_primary_gate_pass"] is True
    assert gate["passing_consecutive_depths"] == list(range(2, 15))
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


def test_adapter_closure_claims_match_canonical_summaries() -> None:
    claims = {claim["id"]: claim for claim in load_ledger()["claims"]}
    persistence = claims["adapter_persistence"]
    transfer = claims["adapter_zero_shot_transfer_minimal"]
    retention = claims["adapter_retention_joint_pass"]

    assert persistence["metrics"]["active_correct"] == 636
    assert persistence["metrics"]["above_diagonal_continue"] == 380
    assert transfer["metrics"]["relay_correct"] == 249
    assert transfer["metrics"]["pointer_correct"] == 264
    assert retention["metrics"]["inverse_correct"] == 2
    assert retention["metrics"]["synthetic_min_accuracy"] == 0.09375
    assert retention["metrics"]["natural_baseline_correct"] == 60
    assert retention["metrics"]["natural_step100_correct"] == 49
    assert retention["metrics"]["tier1_step100_correct"] == 59


def test_adapter_verbal_transference_claim_matches_truncated_receipt() -> None:
    claims = {claim["id"]: claim for claim in load_ledger()["claims"]}
    claim = claims["adapter_verbal_transference"]
    receipt = json.loads((ROOT / claim["evidence"][0]["path"]).read_text(encoding="utf-8"))

    decision = receipt["decision"]
    matched = decision["truncated_transference"]
    near_miss = decision["arm_s_guardrail_near_miss"]
    assert decision["planned_endpoint_available"] is False
    assert decision["last_matched_step"] == claim["metrics"]["last_matched_step"] == 3000
    assert matched["arm_t"]["correct"] == claim["metrics"]["arm_t_correct"] == 1852
    assert matched["arm_s"]["correct"] == claim["metrics"]["arm_s_correct"] == 1282
    assert matched["paired"]["t_only"] == claim["metrics"]["paired_t_only"] == 763
    assert matched["paired"]["s_only"] == claim["metrics"]["paired_s_only"] == 193
    assert matched["paired"]["two_sided_p"] == claim["metrics"]["paired_two_sided_p"]
    assert decision["arm_t_synthetic_regression"]["verdict"] == "retained"
    assert near_miss["paired"]["two_sided_p"] == 0.5


def test_figure4_contains_five_series_and_support_annotations() -> None:
    figure = (
        ROOT / load_ledger()["manuscript_v2_artifact_map"]["figure_4_phase_a_depth_profile"]
    ).read_text(encoding="utf-8")

    assert figure.count("<polyline") == 5
    assert "Arm A: full block, 180.6M trainable" in figure
    assert "Arm E: R16 + bridge, 6.0M trainable" in figure
    assert "First-completed-response accuracy" in figure
    assert "trained support (depths 1-8)" in figure
    assert "Arm A/E crossover: d11.54" in figure
