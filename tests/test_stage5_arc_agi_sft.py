from __future__ import annotations

from pathlib import Path

from colab.run_stage5_arc_agi_sft import best_ladder_row, checkpoint_delta, checkpoint_step


def _summary(selected: int, best: int, first: int = 0, valid_rate: float = 1.0) -> dict[str, object]:
    return {
        "selected_exact": selected,
        "best_of_k_exact": best,
        "first_exact": first,
        "valid_candidate_rate": valid_rate,
    }


def test_checkpoint_step_parses_phase1_checkpoint_name() -> None:
    assert checkpoint_step(Path("phase1_step_150.pt")) == 150
    assert checkpoint_step(Path("not_a_checkpoint.pt")) == -1


def test_checkpoint_delta_tracks_recovery_against_reference() -> None:
    delta = checkpoint_delta(_summary(3, 4, first=2, valid_rate=0.75), _summary(1, 4, first=3, valid_rate=0.25))
    assert delta == {
        "first_exact_delta": -1,
        "selected_exact_delta": 2,
        "best_of_k_exact_delta": 0,
        "valid_candidate_rate_delta": 0.5,
    }


def test_best_ladder_row_prefers_best_then_selected_then_valid_rate_then_later_step() -> None:
    rows = [
        {"step": 100, "summary": _summary(4, 5, valid_rate=0.9)},
        {"step": 200, "summary": _summary(3, 6, valid_rate=0.8)},
        {"step": 300, "summary": _summary(3, 6, valid_rate=0.7)},
    ]
    assert best_ladder_row(rows) == rows[1]
