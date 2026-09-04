from __future__ import annotations

import pytest

from analysis.weft1_epiplexity import (
    EvalLossInterval,
    build_prequential_area_receipt,
)


def point(delta: int, loss: float, k: int = 4) -> EvalLossInterval:
    return EvalLossInterval(
        delta_prediction_tokens=delta,
        eval_bits_per_prediction_token=loss,
        executed_k=k,
        scored_k=k,
    )


def test_dep1_preq_area_uses_explicit_token_intervals_and_terminal_loss() -> None:
    receipt = build_prequential_area_receipt(
        (point(100, 2.0, 1), point(200, 1.5, 2), point(300, 1.0, 4))
    )

    assert receipt.preq_area == pytest.approx(200.0)
    assert receipt.terminal_bits_per_prediction_token == 1.0
    assert receipt.preq_area_units == "bits"
    assert receipt.loss_units == "bits_per_prediction_token"
    assert receipt.curriculum_scoring == "each_eval_at_executed_k_t"
    assert receipt.as_dict()["preq_area"] == pytest.approx(200.0)


def test_dep1_preq_area_follows_unclamped_ratified_formula() -> None:
    receipt = build_prequential_area_receipt((point(10, 0.5), point(20, 1.0)))
    assert receipt.preq_area == pytest.approx(-5.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {
                "delta_prediction_tokens": 0,
                "eval_bits_per_prediction_token": 1.0,
                "executed_k": 1,
                "scored_k": 1,
            },
            "delta_prediction_tokens",
        ),
        (
            {
                "delta_prediction_tokens": 10,
                "eval_bits_per_prediction_token": 1.0,
                "executed_k": 2,
                "scored_k": 4,
            },
            "executed K_t",
        ),
    ),
)
def test_dep1_preq_area_fails_closed_on_unreplayable_intervals(
    kwargs: dict[str, int | float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EvalLossInterval(**kwargs)


def test_dep1_preq_area_requires_a_terminal_eval_point() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_prequential_area_receipt(())
