from __future__ import annotations

import json

from colab.assess_stage5_balanced_mcq import build_assessment, checkpoint_label


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _easy_sweep(path) -> None:
    _write(
        path,
        {
            "kind": "stage5_arceasy_checkpoint_sweep",
            "base": {"summary": {"correct": 421, "total": 570, "accuracy": 421 / 570}},
            "arms": [
                {
                    "label": "step_150",
                    "checkpoint": "runs/phase1_step_150.pt",
                    "summary": {"correct": 412, "total": 570, "accuracy": 412 / 570},
                    "paired_vs_base": {"wins": 16, "losses": 25, "ties": 529, "sign_test_p_value": 0.21},
                },
                {
                    "label": "step_200",
                    "checkpoint": "runs/phase1_step_200.pt",
                    "summary": {"correct": 409, "total": 570, "accuracy": 409 / 570},
                    "paired_vs_base": {"wins": 15, "losses": 27, "ties": 528, "sign_test_p_value": 0.08},
                },
            ],
        },
    )


def _challenge_step150(path) -> None:
    _write(
        path,
        {
            "kind": "stage5_benchmark_suite",
            "run_id": "challenge_step150",
            "checkpoint": "runs/phase1_step_150.pt",
            "benchmarks": ["arc_challenge"],
            "results": {
                "arc_challenge": {
                    "label": {
                        "base": {"mean": {"correct": 167, "total": 299, "accuracy": 167 / 299}},
                        "recurrent": {"mean": {"correct": 169, "total": 299, "accuracy": 169 / 299}},
                    }
                }
            },
            "paired_comparisons": {
                "arc_challenge": {
                    "label": {
                        "mean": {
                            "wins": 24,
                            "losses": 22,
                            "ties": 253,
                            "sign_test_p_value": 0.88,
                        }
                    }
                }
            },
        },
    )


def _combined_suite(path) -> None:
    _write(
        path,
        {
            "kind": "stage5_benchmark_suite",
            "run_id": "combined_suite",
            "checkpoint": "runs/phase1_step_150.pt",
            "benchmarks": ["arc_easy", "arc_challenge"],
            "results": {
                "arc_easy": {
                    "label": {
                        "base": {"mean": {"correct": 421, "total": 570, "accuracy": 421 / 570}},
                        "recurrent": {"mean": {"correct": 422, "total": 570, "accuracy": 422 / 570}},
                    }
                },
                "arc_challenge": {
                    "label": {
                        "base": {"mean": {"correct": 167, "total": 299, "accuracy": 167 / 299}},
                        "recurrent": {"mean": {"correct": 170, "total": 299, "accuracy": 170 / 299}},
                    }
                },
            },
            "paired_comparisons": {
                "arc_easy": {"label": {"mean": {"wins": 10, "losses": 9, "ties": 551}}},
                "arc_challenge": {"label": {"mean": {"wins": 12, "losses": 9, "ties": 278}}},
            },
        },
    )


def _challenge_recovery(path) -> None:
    _write(
        path,
        {
            "phase1_start": {"checkpoint": "runs/phase1_step_125.pt"},
            "best_checkpoint": {"checkpoint": "runs/phase1_step_200.pt"},
            "full_arc_final": {
                "base": {"mean": {"correct": 167, "total": 299, "accuracy": 167 / 299}},
                "phase1_start": {"mean": {"correct": 168, "total": 299, "accuracy": 168 / 299}},
                "phase1_best": {"mean": {"correct": 170, "total": 299, "accuracy": 170 / 299}},
                "best_checkpoint": "runs/phase1_step_200.pt",
                "best_vs_base": {"wins": 22, "losses": 19, "ties": 258},
            },
        },
    )


def test_checkpoint_label_pads_phase1_steps() -> None:
    assert checkpoint_label("outputs/run/phase1_step_50.pt") == "step_050"
    assert checkpoint_label("outputs/run/phase1_step_150.pt") == "step_150"


def test_balanced_assessment_prefers_step150_over_challenge_only_best(tmp_path) -> None:
    easy = tmp_path / "easy" / "summary.json"
    challenge150 = tmp_path / "challenge150" / "summary.json"
    recovery = tmp_path / "recovery" / "summary.json"
    _easy_sweep(easy)
    _challenge_step150(challenge150)
    _challenge_recovery(recovery)

    payload = build_assessment(
        arc_easy_sweep=easy,
        arc_challenge_summaries=[challenge150, recovery],
        required_benchmarks=("arc_easy", "arc_challenge"),
    )

    assert payload["status"] == "needs_competence_recovery"
    assert payload["best_checkpoint"]["label"] == "step_150"
    assert payload["ranked_checkpoints"][0]["micro_correct_delta"] == -7
    assert payload["ranked_checkpoints"][0]["combined_wins"] == 40
    assert payload["ranked_checkpoints"][0]["combined_losses"] == 47
    assert payload["ranked_checkpoints"][1]["label"] == "step_200"
    assert payload["ranked_checkpoints"][1]["micro_correct_delta"] == -9
    assert payload["ranked_checkpoints"][1]["combined_wins"] == 37
    assert payload["ranked_checkpoints"][1]["combined_losses"] == 46


def test_balanced_assessment_accepts_combined_benchmark_suite(tmp_path) -> None:
    combined = tmp_path / "combined" / "summary.json"
    _combined_suite(combined)

    payload = build_assessment(
        arc_easy_sweep=combined,
        arc_challenge_summaries=[combined],
        required_benchmarks=("arc_easy", "arc_challenge"),
    )

    assert payload["status"] == "balanced_nonnegative"
    assert payload["best_checkpoint"]["label"] == "step_150"
    assert payload["best_checkpoint"]["micro_correct_delta"] == 4
