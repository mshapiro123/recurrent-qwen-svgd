from __future__ import annotations

from colab.run_stage5_arc_agi_autopilot import decide_distill_gate, decide_trace_gate


def test_decide_trace_gate_runs_when_symbolic_and_hybrid_clear_thresholds() -> None:
    payload = {
        "symbolic_coverage": {"exact_symbolic": 1},
        "rows": [
            {"variant": "phase1_model_only", "best": 2},
            {"variant": "phase1_hybrid_symbolic_first", "best": 2},
            {"variant": "base_model_only", "best": 1},
            {"variant": "base_hybrid_symbolic_first", "best": 1},
        ],
    }
    decision, evidence = decide_trace_gate(payload)
    assert decision is True
    assert evidence["best_hybrid_delta"] == 0


def test_decide_trace_gate_skips_without_symbolic_coverage() -> None:
    payload = {
        "symbolic_coverage": {"exact_symbolic": 0},
        "rows": [
            {"variant": "phase1_model_only", "best": 2},
            {"variant": "phase1_hybrid_symbolic_first", "best": 3},
        ],
    }
    decision, evidence = decide_trace_gate(payload)
    assert decision is False
    assert evidence["symbolic_exact"] == 0


def test_decide_distill_gate_runs_when_trace_matches_grid() -> None:
    payload = {
        "comparison": {
            "grid_only": {"tuned_best": 3, "tuned_selected": 2},
            "symbolic_trace_covered": {"tuned_best": 3, "tuned_selected": 4},
        }
    }
    decision, evidence = decide_distill_gate(payload)
    assert decision is True
    assert evidence["best_trace_arm"] == "symbolic_trace_covered"
    assert evidence["best_trace_mode"] == "symbolic"
    assert evidence["best_trace_filter"] == "covered"
    assert evidence["trace_best_delta"] == 0
    assert evidence["trace_selected_delta"] == 2


def test_decide_distill_gate_selects_best_state_trace_arm() -> None:
    payload = {
        "comparison": {
            "grid_only": {"tuned_best": 2, "tuned_selected": 2, "best_best": 2},
            "symbolic_program_trace_covered": {"tuned_best": 2, "tuned_selected": 3, "best_best": 3},
            "symbolic_state_trace_covered": {"tuned_best": 4, "tuned_selected": 3, "best_best": 4},
        }
    }

    decision, evidence = decide_distill_gate(payload)

    assert decision is True
    assert evidence["best_trace_arm"] == "symbolic_state_trace_covered"
    assert evidence["best_trace_mode"] == "symbolic_state_trace"
    assert evidence["best_trace_filter"] == "covered"
    assert evidence["trace_best_delta"] == 2
    assert evidence["trace_selected_delta"] == 1
