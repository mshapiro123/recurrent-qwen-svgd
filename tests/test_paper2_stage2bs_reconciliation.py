import json
from pathlib import Path

import torch

from eval.eval_paper2_stage2bs_reconciliation import (
    _first_divergence,
    tensor_comparison,
)


ROOT = Path(__file__).resolve().parents[1]


def test_reconciliation_lock_is_authorized_before_trace() -> None:
    lock = json.loads(
        (ROOT / "training/paper2_stage2bs_reconciliation_lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert lock["locked_before_trace"] is True
    assert lock["constraints"]["training_authorized"] is False
    assert lock["constraints"]["optimizer_constructed"] is False
    assert lock["authority"]["sha256"] == (
        "4cb2f9b2e05da7bbe1400a43e64412df59743ecf921aaca1463605460537153f"
    )


def test_tensor_comparison_reports_identity_and_rank() -> None:
    value = torch.tensor([[1.0, -2.0, 3.0]])
    exact = tensor_comparison(value, value.clone())
    assert exact["bit_exact"] is True
    assert exact["max_abs_delta"] == 0.0
    assert exact["top_abs_coordinate_agreement_at_k"] == 1.0
    changed = tensor_comparison(value, value + 1.0)
    assert changed["bit_exact"] is False
    assert changed["max_abs_delta"] == 1.0


def test_first_divergence_uses_registered_order() -> None:
    rows = [
        {"stage": "tokenized_inputs", "comparable": True, "bit_exact": True},
        {"stage": "prefix_output", "comparable": True, "bit_exact": True},
        {"stage": "loop_1_post_state", "comparable": True, "bit_exact": False},
        {"stage": "loop_1_capped_write", "comparable": False},
    ]
    result = _first_divergence(rows)
    assert result["stage"] == "loop_1_post_state"
    assert result["classification"] == "effective_k_and_one_shot_vs_full_recurrent_iteration"


def test_source_contract_names_both_distinct_graphs() -> None:
    source = (ROOT / "eval/eval_paper2_stage2bs_reconciliation.py").read_text(
        encoding="utf-8"
    )
    assert "P34TaskInferenceGraph" in source
    assert "Stage2BTaskInferenceGraph" in source
    assert "stage2b_depth_enabled=True" in source
    assert "training_performed" in source


def test_colab_target_is_score_only_and_sealed() -> None:
    cell = (ROOT / "colab/STAGE5_PAPER2_STAGE2BS_RECONCILIATION_CELL.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "colab/run_stage5_paper2_stage2bs_reconciliation.py").read_text(
        encoding="utf-8"
    )
    assert "paper2_stage2bs_reconciliation_v1" in cell
    assert "no optimizer no training" in cell
    assert "CONFIRM and EVAL-E remain sealed" in cell
    assert "optimizer_steps" in runner
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(
        encoding="utf-8"
    )
    assert '"paper2_stage2bs_reconciliation"' in bootstrap
    assert "STAGE5_PAPER2_STAGE2BS_RECONCILIATION_CELL.py" in bootstrap
    assert 'assert GH_TOKEN, "Missing GH_TOKEN' not in bootstrap
    assert 'if GH_TOKEN:' in bootstrap
