from __future__ import annotations

from pathlib import Path

from colab.run_stage5_depth_selector_bounded import (
    HELDOUT64,
    N24_KEEPER_SHA256,
    N24_KEEPER_DRIVE,
    TRAIN_ROWS,
    locked_spec,
)


ROOT = Path(__file__).resolve().parents[1]


def test_locked_spec_matches_bounded_handoff() -> None:
    spec = locked_spec()
    assert spec["max_loops"] == 12
    assert spec["heldout_rows_per_depth"] == 64
    assert spec["steps_per_arm"] == 2000
    assert spec["batch_size"] == 8
    assert spec["s1_min_correct_per_depth"] == 46
    assert spec["s1_answer_delta_floor"] == -0.03
    assert spec["s2_geometric_prior_mean"] == 6.0
    assert spec["s2_beta"] == 0.02
    assert spec["s2_strong_spearman_floor"] == 0.8
    assert spec["s2_partial_spearman_floor"] == 0.3


def test_source_lineage_is_explicit_and_frozen() -> None:
    assert N24_KEEPER_SHA256 == "898a259db2ab344ece4545e2910b051840e8408dbe4927f799e7cdb3cdd8c7dc"
    assert "stage5_n24_support12_rung_20260707_140139" in str(N24_KEEPER_DRIVE)
    assert TRAIN_ROWS == ROOT / "outputs/stage5/stage5_n24_support12_rung_20260707_140139/data/train_chain_mcq.jsonl"
    assert HELDOUT64 == ROOT / "outputs/stage5/stage5_n24_support12_rung_20260707_140139/data/test_chain_mcq_heldout64.jsonl"


def test_bootstrap_exposes_bounded_selector_target() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    assert '"depth_selector_bounded_assessment"' in bootstrap
    assert "STAGE5_DEPTH_SELECTOR_CELL.py" in bootstrap
    assert "depth_selector_bounded_v1" in bootstrap


def test_launcher_carries_frozen_contract_markers() -> None:
    launcher = (ROOT / "colab/STAGE5_DEPTH_SELECTOR_CELL.py").read_text(encoding="utf-8")
    for marker in (
        "N24_KEEPER_SHA256",
        "frozen_parameter_hash",
        "S1_supervised_depth_reading",
        "S2_ponder_outcome",
        "canary_exemption",
        "tests/test_depth_selector_bounded.py",
    ):
        assert marker in launcher


def test_runner_supports_s2_only_resume_and_late_saturation() -> None:
    runner = (ROOT / "colab/run_stage5_depth_selector_bounded.py").read_text(encoding="utf-8")
    for marker in (
        "STAGE5_DEPTH_SELECTOR_RESUME_S2",
        "resuming_depth_selector_S2",
        "reusing_published_S1_gate",
        "selector_gradient_saturated",
        "if step == 1:",
    ):
        assert marker in runner
