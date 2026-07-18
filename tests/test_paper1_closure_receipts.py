from __future__ import annotations

import math
from pathlib import Path

from colab.build_paper1_closure_receipts import (
    PRIMARY_MULTIPLICITY_FAMILY,
    build_receipts,
    render_markdown,
)


def test_receipt_compiler_locks_guardrail_multiplicity_and_dead_bridge() -> None:
    payload = build_receipts()
    battery = payload["guardrails"]["lineage_regression_battery"]
    adverse = battery["most_adverse_primary_result"]

    assert battery["primary_family"]["family_size"] == PRIMARY_MULTIPLICITY_FAMILY == 8
    assert adverse["benchmark"] == "arc_easy"
    assert adverse["score_target"] == "cyclic_label_aggregated"
    assert adverse["correct_delta"] == -14
    assert math.isclose(adverse["raw_sign_test_p"], 0.03847730828420026)
    assert math.isclose(adverse["bonferroni_p"], 0.3078184662736021)
    assert battery["all_noninferior_at_minus_3pp"] is True

    bridge = payload["early_stochastic_era"]["dead_bridge"]
    assert bridge["bridge"]["bridge_gate"] == 0.0
    assert bridge["bridge"]["sample_bridge_delta_rms"] == 0.0
    assert bridge["gradient_liveness"]["weight_grad_rms"] == 0.0


def test_receipt_compiler_resolves_r16_active_budget_and_claim_boundaries() -> None:
    payload = build_receipts()
    accounting = payload["parameter_accounting"]["r16_split_bridge"]

    assert accounting["optimizer_marked"] == 7_613_953
    assert accounting["forward_active"] == 6_007_425
    assert accounting["bridge_legacy_bypassed"] == 1_606_528
    assert "bridge.proj.weight" in accounting["legacy_tensor_names"]
    assert any("McLeish" in claim for claim in payload["do_not_claim"])

    markdown = render_markdown(payload)
    assert "corrected `p=0.307818`" in markdown
    assert "manuscript prose was not edited" in markdown


def test_early_telemetry_is_deduplicated_to_expected_cells() -> None:
    telemetry = build_receipts()["early_stochastic_era"]["telemetry"]
    by_label = {row["label"]: row for row in telemetry}

    assert by_label["extended_fold0_random32_rep05"]["unique_task_seed_cells"] == 35
    assert by_label["extended_fold0_random32_rep05"]["config"]["num_trajectories"] == 4
    assert (
        by_label["recreated_current_within_group_dim8_rep2"]["unique_task_seed_cells"]
        == 70
    )
    assert (
        by_label["original_stage4_exact_phase1_vs_phase2"]["unique_task_seed_cells"]
        == 39
    )


def test_cpu_receipt_target_is_wired_without_a_gpu_requirement() -> None:
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = Path("colab/STAGE5_PAPER1_CLOSURE_RECEIPTS_CELL.py").read_text(
        encoding="utf-8"
    )

    assert '"paper1_closure_receipts"' in bootstrap
    assert "tests/test_paper1_closure_receipts.py" in cell
    assert "nvidia-smi" not in cell
    assert "manuscript prose was not edited" in cell
