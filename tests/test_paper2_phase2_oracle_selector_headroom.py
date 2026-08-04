from __future__ import annotations

import torch

from eval.eval_paper2_phase2_oracle_selector_headroom import selector_headroom


def test_selector_headroom_prices_positive_mass_and_quality_safe_ceiling() -> None:
    rows = {
        "accepted_length": torch.tensor([2.0, 0.5, 1.5]),
        "base_accepted_length": torch.tensor([1.0, 1.0, 1.0]),
        "acceptance_delta": torch.tensor([1.0, -0.5, 0.5]),
        "base_correct_by_horizon": torch.tensor(
            [[True, True], [True, True], [True, True]]
        ),
        "bridge_correct_by_horizon": torch.tensor(
            [[True, True], [True, True], [True, False]]
        ),
    }
    result = selector_headroom(rows)
    assert result["oracle_selected_rows"] == 2
    assert abs(result["oracle_acceptance_delta"] - 0.5) < 1e-7
    assert result["quality_safe_selected_rows"] == 1
    assert abs(result["quality_safe_oracle_acceptance_delta"] - 1.0 / 3.0) < 1e-7
    assert result["quality_loss_rows_among_acceptance_oracle"] == 1
    assert result["quality_safe_oracle_retention"] == 1.0


def test_selector_headroom_rejects_inconsistent_banked_deltas() -> None:
    rows = {
        "accepted_length": torch.tensor([2.0]),
        "base_accepted_length": torch.tensor([1.0]),
        "acceptance_delta": torch.tensor([0.0]),
        "base_correct_by_horizon": torch.tensor([[True]]),
        "bridge_correct_by_horizon": torch.tensor([[True]]),
    }
    try:
        selector_headroom(rows)
    except RuntimeError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("inconsistent banked deltas must fail closed")


def test_headroom_sources_prohibit_model_compute() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    evaluator = (root / "eval/eval_paper2_phase2_oracle_selector_headroom.py").read_text(
        encoding="utf-8"
    )
    runner = (root / "colab/run_stage5_paper2_phase2_oracle_selector_headroom.py").read_text(
        encoding="utf-8"
    )
    combined = evaluator + runner
    assert "torch.optim" not in combined
    assert "transformers" not in combined
    assert '"model_inference_runs": 0' in evaluator
    assert '"optimizer_steps": 0' in evaluator
    assert "stage5_paper2_phase2_oracle_selector_headroom_20260805" in runner
