from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from eval.eval_mcq import aggregate_loop_diagnostics, extract_loop_diagnostics


def test_extract_loop_diagnostics_scalarizes_recurrent_output() -> None:
    output = SimpleNamespace(
        metrics={
            "mean_expected_loops": torch.tensor(2.5),
            "mean_halt_entropy": torch.tensor(1.2),
        },
        expected_loops=torch.tensor([[2.0, 3.0]]),
        halting_weights=torch.tensor([[[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]]]),
    )

    diagnostics = extract_loop_diagnostics(output)

    assert diagnostics["mean_expected_loops"] == pytest.approx(2.5)
    assert diagnostics["mean_halt_entropy"] == pytest.approx(1.2)
    assert diagnostics["expected_loops"] == [2.0, 3.0]
    assert diagnostics["mean_halting_weights"] == pytest.approx([0.15, 0.35, 0.5])


def test_aggregate_loop_diagnostics_reports_answer_and_prediction_depth() -> None:
    diagnostics = aggregate_loop_diagnostics(
        {
            "A": {"mean_expected_loops": 1.5, "mean_halt_entropy": 0.6},
            "B": {"mean_expected_loops": 3.0, "mean_halt_entropy": 1.1},
        },
        answer="A",
        prediction="B",
    )

    assert diagnostics["mean_expected_loops"] == pytest.approx(2.25)
    assert diagnostics["mean_halt_entropy"] == pytest.approx(0.85)
    assert diagnostics["answer_expected_loops"] == pytest.approx(1.5)
    assert diagnostics["prediction_expected_loops"] == pytest.approx(3.0)
