from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from eval.rescore_d0_expert_choice import (
    binary_auc,
    causal_window_expert_choice,
    curve_replay_diagnostics,
    floor_transition_archaeology,
    score_selected_second_loop,
)


def test_causal_window_choice_never_uses_future_scores() -> None:
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.7])
    row_indices = torch.tensor([0, 0, 0, 0, 0])
    selected = causal_window_expert_choice(
        scores,
        row_indices=row_indices,
        budget_fraction=0.5,
        window=2,
    )
    # The first token is selected before the future 0.9 exists. At index 2,
    # the prior 0.9 occupies the one-token capacity and the current 0.2 stops.
    assert selected.tolist() == [True, True, False, True, False]


def test_causal_window_resets_at_source_row_boundaries() -> None:
    scores = torch.tensor([0.9, 0.1, 0.2])
    rows = torch.tensor([0, 0, 1])
    selected = causal_window_expert_choice(
        scores,
        row_indices=rows,
        budget_fraction=0.5,
        window=256,
    )
    assert selected.tolist() == [True, False, True]


def test_selected_second_loop_reports_help_minus_harm() -> None:
    matches = torch.tensor([[False, True], [True, False], [True, True], [False, False]])
    selected = torch.tensor([True, True, True, False])
    result = score_selected_second_loop(matches, selected)
    assert result["helps"] == 1
    assert result["hurts"] == 1
    assert result["neutral"] == 1
    assert result["net_correct_delta"] == 0
    assert result["correct"] == 2


def test_binary_auc_excludes_neutral_positions() -> None:
    scores = torch.tensor([0.9, 0.1, 100.0])
    helps = torch.tensor([True, False, False])
    hurts = torch.tensor([False, True, False])
    assert binary_auc(scores, helps=helps, hurts=hurts) == 1.0


def test_floor_archaeology_reconstructs_matches_from_predictions() -> None:
    result = floor_transition_archaeology(
        [
            {"teacher_7b": 7, "predictions": [2, 7]},
            {"teacher_7b": 7, "predictions": [7, 2]},
            {"teacher_7b": 7, "predictions": [7, 7]},
        ]
    )
    assert result["helps"] == 1
    assert result["hurts"] == 1


def test_curve_replay_accepts_only_tiny_cross_runtime_decision_drift() -> None:
    banked = [
        {
            "penalty": 0.01,
            "correct": 10,
            "total": 16,
            "accuracy": 0.625,
            "mean_loops": 1.125,
            "net_utility": 0.62375,
        }
    ]
    total = 200_000
    banked[0].update(
        {
            "correct": 125_000,
            "total": total,
            "accuracy": 0.625,
            "mean_loops": 1.125,
            "net_utility": 0.62375,
        }
    )
    reconstructed = [
        {
            **banked[0],
            "correct": 125_001,
            "accuracy": 125_001 / total,
            "mean_loops": 1.125 + 4 / total,
            "net_utility": 125_001 / total - 0.01 * (0.125 + 4 / total),
        }
    ]
    diagnostics = curve_replay_diagnostics(reconstructed, banked)
    assert diagnostics["pass"] is True
    assert diagnostics["status"] == "numerically_equivalent_not_bit_exact"
    assert diagnostics["decision_fields_bit_exact"] is False
    assert diagnostics["maximum_correct_count_difference"] == 1
    assert diagnostics["maximum_derived_absolute_difference"] > 0

    wrong_counts = [
        {
            **reconstructed[0],
            "correct": 125_003,
            "accuracy": 125_003 / total,
        }
    ]
    diagnostics = curve_replay_diagnostics(wrong_counts, banked)
    assert diagnostics["pass"] is False
    assert diagnostics["within_numerical_equivalence_envelope"] is False


def test_rescore_cli_resolves_repo_imports_outside_repo_cwd(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "eval/rescore_d0_expert_choice.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
