from __future__ import annotations

from colab.run_stage5_arc_agi_distill_sft_gate import compact as distill_compact
from colab.run_stage5_arc_agi_trace_sft_gate import available_arms, compact as trace_compact


def _summary(selected: int, best: int, valid_rate: float = 1.0) -> dict[str, object]:
    return {
        "selected_exact": selected,
        "best_of_k_exact": best,
        "first_exact": selected,
        "valid_candidate_rate": valid_rate,
        "examples_with_targets": 10,
        "tasks_solved_best_of_k": best,
        "tasks_with_targets": 10,
    }


def _run_summary() -> dict[str, object]:
    return {
        "base": _summary(6, 7),
        "phase1_start": _summary(2, 3),
        "phase1_arc_agi_tuned": _summary(4, 4, valid_rate=0.4),
        "eval_diagnostics": {
            "phase1_arc_agi_tuned": {
                "program_verifier_summary": {
                    "candidates_with_program": 2,
                    "candidates_program_fits_train": 1,
                    "program_fit_selected_exact": 1,
                }
            }
        },
        "best_checkpoint": {
            "step": 150,
            "summary": _summary(5, 6, valid_rate=0.8),
            "eval_diagnostics": {
                "program_verifier_summary": {
                    "candidates_with_program": 4,
                    "candidates_program_fits_train": 3,
                    "program_fit_selected_exact": 2,
                }
            },
        },
    }


def test_distill_gate_compact_uses_best_checkpoint_when_available() -> None:
    row = distill_compact(_run_summary())
    assert row["tuned_best"] == 4
    assert row["best_step"] == 150
    assert row["best_best"] == 6
    assert row["best_valid_rate"] == 0.8
    assert row["best_program_fits"] == 3


def test_trace_gate_compact_uses_best_checkpoint_when_available() -> None:
    row = trace_compact(_run_summary())
    assert row["tuned_selected"] == 4
    assert row["best_selected"] == 5
    assert row["tasks_solved_best"] == 6
    assert row["tuned_program_fits"] == 1
    assert row["best_program_fit_selected_exact"] == 2


def test_trace_gate_supports_symbolic_program_arms() -> None:
    arms = available_arms()
    assert arms["symbolic_program_trace_covered"] == ("symbolic_program", "covered")
    assert arms["symbolic_program_trace_all"] == ("symbolic_program", "all")
