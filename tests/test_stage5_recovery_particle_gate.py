from __future__ import annotations

from colab.run_stage5_arc_agi_recovery_particle_gate import (
    compare_summaries,
    compare_task_family_summaries,
    decide_particle_value,
    decide_recovery,
    parse_particle_variants,
    recovered_task_family_summary,
    select_recovered_checkpoint,
    summarize_particle_ladder,
    summarize_holdout_recovery,
    summary_from_payload,
    task_family_summary_from_payload,
)


def _summary(selected: int, best: int, first: int = 0, valid_rate: float = 1.0) -> dict[str, object]:
    return {
        "selected_exact": selected,
        "best_of_k_exact": best,
        "first_exact": first,
        "valid_candidate_rate": valid_rate,
    }


def test_parse_particle_variants() -> None:
    variants = parse_particle_variants("control:0:0,svgd:0.01:0.5")
    assert [variant.name for variant in variants] == ["control", "svgd"]
    assert variants[1].noise == 0.01
    assert variants[1].repulsion == 0.5


def test_compare_summaries_tracks_selected_best_and_valid_rate() -> None:
    delta = compare_summaries(_summary(3, 4, first=2, valid_rate=0.8), _summary(1, 4, first=3, valid_rate=0.5))
    assert delta == {
        "selected_delta": 2,
        "best_of_k_delta": 0,
        "first_delta": -1,
        "valid_rate_delta": 0.30000000000000004,
    }


def test_payload_helpers_accept_full_eval_payloads_and_compact_diagnostics() -> None:
    payload = {
        "summary": _summary(2, 3),
        "task_family_summary": {"move_recolor": {"selected_exact": 2}},
    }
    compact = {"eval_diagnostics": {"task_family_summary": {"frame_object": {"selected_exact": 1}}}}

    assert summary_from_payload(payload)["best_of_k_exact"] == 3
    assert summary_from_payload(_summary(1, 1))["selected_exact"] == 1
    assert task_family_summary_from_payload(payload)["move_recolor"]["selected_exact"] == 2
    assert task_family_summary_from_payload(compact)["frame_object"]["selected_exact"] == 1


def test_compare_task_family_summaries_tracks_family_deltas() -> None:
    delta = compare_task_family_summaries(
        {"move_recolor": _summary(3, 4), "frame_object": _summary(1, 1)},
        {"move_recolor": _summary(1, 4), "crop": _summary(2, 2)},
    )

    assert delta["move_recolor"]["selected_delta"] == 2
    assert delta["move_recolor"]["best_of_k_delta"] == 0
    assert delta["frame_object"]["selected_delta"] == 1
    assert delta["crop"]["selected_delta"] == -2


def test_decide_recovery_requires_non_negative_selected_and_best() -> None:
    payload = {
        "base": _summary(4, 4),
        "phase1_start": _summary(1, 2),
        "phase1_arc_agi_tuned": _summary(2, 2),
    }
    decision, evidence = decide_recovery(payload)
    assert decision is True
    assert evidence["phase1_tuned_vs_start"]["selected_delta"] == 1
    assert evidence["phase1_tuned_vs_start"]["best_of_k_delta"] == 0


def test_decide_recovery_rejects_selected_regression_even_if_best_matches() -> None:
    payload = {
        "base": _summary(4, 4),
        "phase1_start": _summary(2, 2),
        "phase1_arc_agi_tuned": _summary(1, 2),
    }
    decision, evidence = decide_recovery(payload)
    assert decision is False
    assert evidence["phase1_tuned_vs_start"]["selected_delta"] == -1


def test_select_recovered_checkpoint_prefers_best_checkpoint() -> None:
    payload = {
        "tuned_checkpoint": "final.pt",
        "phase1_arc_agi_tuned": _summary(1, 1),
        "best_checkpoint": {
            "step": 150,
            "checkpoint": "best.pt",
            "summary": _summary(3, 4),
        },
    }
    recovered = select_recovered_checkpoint(payload)
    assert recovered["source"] == "best_checkpoint"
    assert recovered["checkpoint"] == "best.pt"
    assert recovered["summary"]["best_of_k_exact"] == 4


def test_decide_recovery_uses_best_checkpoint_when_available() -> None:
    payload = {
        "base": _summary(4, 4),
        "phase1_start": _summary(1, 1),
        "phase1_arc_agi_tuned": _summary(0, 0),
        "tuned_checkpoint": "final.pt",
        "best_checkpoint": {
            "step": 150,
            "checkpoint": "best.pt",
            "summary": _summary(2, 2),
        },
    }
    decision, evidence = decide_recovery(payload)
    assert decision is True
    assert evidence["phase1_recovered"]["checkpoint"] == "best.pt"
    assert evidence["phase1_tuned_vs_start"]["selected_delta"] == 1


def test_recovered_task_family_summary_prefers_best_checkpoint() -> None:
    payload = {
        "eval_diagnostics": {
            "phase1_arc_agi_tuned": {
                "task_family_summary": {"move_recolor": {"selected_exact": 1}},
            }
        },
        "best_checkpoint": {
            "eval_diagnostics": {
                "task_family_summary": {"frame_object": {"selected_exact": 2}},
            }
        },
    }
    recovered = {"source": "best_checkpoint"}

    assert recovered_task_family_summary(payload, recovered)["frame_object"]["selected_exact"] == 2


def test_decide_particle_value_requires_non_negative_over_tuned() -> None:
    tuned = _summary(2, 3)
    particle_summaries = {
        "bad": _summary(3, 2),
        "good": _summary(2, 4),
    }
    decision, evidence = decide_particle_value(particle_summaries, tuned)
    assert decision is True
    assert evidence["best_non_negative_variant"] == "good"


def test_decide_particle_value_reports_task_family_deltas_from_full_payloads() -> None:
    tuned = _summary(2, 3)
    tuned_families = {"move_recolor": _summary(1, 1)}
    particle_summaries = {
        "svgd": {
            "summary": _summary(2, 4),
            "task_family_summary": {"move_recolor": _summary(2, 3)},
        }
    }

    decision, evidence = decide_particle_value(particle_summaries, tuned, tuned_families)

    assert decision is True
    assert evidence["variants"]["svgd"]["task_family_delta_vs_tuned"]["move_recolor"]["selected_delta"] == 1
    assert evidence["variants"]["svgd"]["task_family_delta_vs_tuned"]["move_recolor"]["best_of_k_delta"] == 2


def test_decide_particle_value_rejects_all_negative() -> None:
    decision, evidence = decide_particle_value({"bad": _summary(2, 2)}, _summary(2, 3))
    assert decision is False
    assert evidence["best_non_negative_variant"] is None


def test_summarize_particle_ladder_tracks_variant_windows() -> None:
    rows = [
        {
            "step": 100,
            "variant": "noise",
            "delta_vs_recurrent": {
                "selected_delta": 0,
                "best_of_k_delta": 1,
                "first_delta": 0,
                "valid_rate_delta": 0.0,
            },
        },
        {
            "step": 200,
            "variant": "noise",
            "delta_vs_recurrent": {
                "selected_delta": -1,
                "best_of_k_delta": 2,
                "first_delta": 0,
                "valid_rate_delta": 0.0,
            },
        },
        {
            "step": 200,
            "variant": "svgd",
            "delta_vs_recurrent": {
                "selected_delta": 1,
                "best_of_k_delta": 1,
                "first_delta": 0,
                "valid_rate_delta": 0.0,
            },
        },
    ]
    summary = summarize_particle_ladder(rows)
    assert summary["evaluated_rows"] == 3
    assert summary["by_variant"]["noise"]["evaluated_checkpoints"] == 2
    assert summary["by_variant"]["noise"]["non_negative_checkpoints"] == 1
    assert summary["by_variant"]["svgd"]["non_negative_checkpoints"] == 1
    assert summary["best_row"]["step"] == 200
    assert summary["best_row"]["variant"] == "noise"


def test_summarize_holdout_recovery_tracks_parse_modes_and_deltas() -> None:
    holdout = {
        "prefer": {
            "base": {"summary": _summary(0, 1), "parse_method_summary": {"grid": {"count": 1}}},
            "phase1_start": {"summary": _summary(1, 1), "parse_method_summary": {"grid": {"count": 1}}},
            "phase1_tuned": {"summary": _summary(2, 3), "parse_method_summary": {"program": {"count": 3}}},
        }
    }
    summary = summarize_holdout_recovery(holdout)
    assert summary["prefer"]["phase1_tuned_vs_start"]["selected_delta"] == 1
    assert summary["prefer"]["phase1_tuned_vs_start"]["best_of_k_delta"] == 2
    assert summary["prefer"]["parse_methods"]["phase1_tuned"] == {"program": {"count": 3}}
