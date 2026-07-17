from __future__ import annotations

from pathlib import Path

import colab.run_stage5_peft_ponder_closure as runner


ROOT = Path(__file__).resolve().parents[1]


def test_runner_is_locked_to_corrected_reference_and_adamw() -> None:
    assert runner.REFERENCE_SUMMARY.as_posix().endswith(
        "stage5_chain_scaled_corrected_20260702_182827/summary.json"
    )
    config_source = (ROOT / "colab/run_stage5_peft_ponder_closure.py").read_text(encoding="utf-8")
    assert '"training_mode": "controller_only" if controller_only else "frozen_lora"' in config_source
    assert '"optimizer": "adamw"' in config_source
    assert '"reject_muon": True' in config_source
    assert '"bridge_prelude_lr_multiplier": 1.0 if controller_only else 10.0' in config_source
    assert '"bridge_prelude_grad_multiplier": 1.0' in config_source
    assert '"require_frozen_base_hash": True' in config_source


def test_runner_wires_identity_interval_canaries_and_p2() -> None:
    source = (ROOT / "colab/run_stage5_peft_ponder_closure.py").read_text(encoding="utf-8")
    for marker in (
        "eval/eval_peft_identity.py",
        "eval/eval_synthetic_depth_active_labels.py",
        "tier1_canary_verdict",
        "eval/eval_ponder_depth.py",
        "checkpoint_include_frozen_lora",
        "base_hash_unchanged",
        "ponder_final_natural_loop1",
    ):
        assert marker in source


def test_bootstrap_exposes_guarded_peft_ponder_target() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_PEFT_PONDER_CLOSURE_CELL.py").read_text(encoding="utf-8")
    assert '"peft_ponder_closure": {' in bootstrap
    for marker in (
        "peft_ponder_closure_v1",
        "frozen_lora",
        "controller_only",
        "reject_muon",
        "require_frozen_base_hash",
        "tests/test_peft_ponder_closure.py",
        "pinned_checkout",
    ):
        assert marker in bootstrap
        assert marker in cell


def test_balanced_prefix_is_exact() -> None:
    rows = [
        {"id": f"d{depth}_{index}", "depth": depth}
        for depth in range(1, 4)
        for index in range(4)
    ]
    selected = runner.balanced_prefix(rows, rows_per_depth=2, max_depth=3)
    assert len(selected) == 6
    assert [row["id"] for row in selected] == [
        "d1_0",
        "d1_1",
        "d2_0",
        "d2_1",
        "d3_0",
        "d3_1",
    ]
