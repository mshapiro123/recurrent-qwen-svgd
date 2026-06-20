from __future__ import annotations

from colab.run_stage5_arc_agi_curriculum_particle_autopilot import child_run_id, summarize_autopilot


def test_child_run_id_uses_parent_prefix() -> None:
    assert child_run_id("curriculum").endswith("_curriculum")


def test_summarize_autopilot_keeps_final_checkpoint_and_particle_decision() -> None:
    curriculum = {
        "final_checkpoint": "outputs/stage5/run/phase1_step_250.pt",
        "stages": [
            {
                "stage": {"name": "warmup", "synthetic_modes": "constant_output"},
                "selected_checkpoint": {
                    "checkpoint": "warmup.pt",
                    "summary": {"selected_exact": 1, "best_of_k_exact": 2},
                },
            },
            {
                "stage": {"name": "mixed", "synthetic_modes": "all"},
                "selected_checkpoint": {
                    "checkpoint": "mixed.pt",
                    "summary": {"selected_exact": 3, "best_of_k_exact": 4},
                },
            },
        ],
    }
    particle = {
        "particle_decision": {
            "passed": True,
            "evidence": {
                "best_replicated_variant": "k4_noise001_rep05",
                "variants": {
                    "k4_noise001_rep05": {
                        "mean_delta_vs_tuned": {"selected_delta": 1.0, "best_of_k_delta": 2.0},
                    }
                },
            },
        }
    }

    compact = summarize_autopilot(curriculum, particle)

    assert compact["final_checkpoint"] == "outputs/stage5/run/phase1_step_250.pt"
    assert compact["curriculum_stages"][1]["name"] == "mixed"
    assert compact["particle_passed"] is True
    assert compact["best_replicated_particle_variant"] == "k4_noise001_rep05"
    assert compact["particle_variant_mean_deltas"]["k4_noise001_rep05"]["best_of_k_delta"] == 2.0
