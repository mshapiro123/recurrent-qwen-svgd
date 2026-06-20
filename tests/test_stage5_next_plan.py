from __future__ import annotations

import json

from colab.plan_stage5_next_run import (
    best_recovered_tta_row,
    plan_next_actions,
    source_kind,
)


def _summary(selected: int, best: int, examples: int = 50) -> dict[str, object]:
    return {
        "selected_exact": selected,
        "best_of_k_exact": best,
        "examples_with_targets": examples,
        "valid_candidate_rate": 1.0,
    }


def test_failed_candidate_distillation_recommends_baseline_curriculum(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "compact": {
            "candidate_distillation_passed": False,
            "final_checkpoint": None,
            "particle_passed": False,
        }
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run baseline curriculum without candidate distillation"
    assert "RUN_CANDIDATE_DISTILL_GATE=0" in actions[0]["command"]


def test_smoke_win_recommends_confirmation_and_export(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        },
        "recovered_benchmark": {
            "base": {"summary": _summary(2, 3)},
            "phase1_start": {"summary": _summary(1, 1)},
            "recovered": {"summary": _summary(3, 4)},
            "deltas": {
                "recovered_vs_base": {
                    "selected_exact_delta": 1,
                    "best_of_k_exact_delta": 1,
                },
                "recovered_vs_start": {
                    "selected_exact_delta": 2,
                    "best_of_k_exact_delta": 3,
                },
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)
    names = [action["name"] for action in actions]

    assert any(name.startswith("Confirm recovered-vs-base") for name in names)
    assert "Export recovered adapter to Hugging Face" in names


def test_partial_recovery_recommends_scaled_curriculum(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        },
        "recovered_benchmark": {
            "base": {"summary": _summary(8, 9)},
            "phase1_start": {"summary": _summary(2, 2)},
            "recovered": {"summary": _summary(4, 5)},
            "deltas": {
                "recovered_vs_base": {
                    "selected_exact_delta": -4,
                    "best_of_k_exact_delta": -4,
                },
                "recovered_vs_start": {
                    "selected_exact_delta": 2,
                    "best_of_k_exact_delta": 3,
                },
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Scale deterministic curriculum"
    assert "STAGE5_ARC_AGI_TRAIN_TASK_LIMIT=160" in actions[0]["command"]


def test_no_recovery_recommends_training_target_diagnostics(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        },
        "recovered_benchmark": {
            "base": {"summary": _summary(8, 9)},
            "phase1_start": {"summary": _summary(4, 5)},
            "recovered": {"summary": _summary(3, 4)},
            "deltas": {
                "recovered_vs_base": {
                    "selected_exact_delta": -5,
                    "best_of_k_exact_delta": -5,
                },
                "recovered_vs_start": {
                    "selected_exact_delta": -1,
                    "best_of_k_exact_delta": -1,
                },
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run trace/candidate-distillation diagnostics before more SFT"
    assert "run_stage5_arc_agi_candidate_distill_gate.py" in actions[0]["command"]


def test_best_recovered_tta_row_prefers_best_then_selected() -> None:
    tta = {
        "rows": [
            {"arm": "recovered", "tta_variant": "none", "best_of_k_exact": 2, "selected_exact": 2},
            {"arm": "recovered", "tta_variant": "all", "best_of_k_exact": 4, "selected_exact": 1},
            {"arm": "recovered", "tta_variant": "rotations", "best_of_k_exact": 4, "selected_exact": 3},
        ]
    }

    assert best_recovered_tta_row(tta)["tta_variant"] == "rotations"


def test_source_kind_classifies_followup_and_autopilot() -> None:
    assert source_kind({"recovered_benchmark": {}}) == "followup"
    assert source_kind({"compact": {}}) == "autopilot"
