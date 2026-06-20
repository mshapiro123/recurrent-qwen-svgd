from __future__ import annotations

from colab.run_stage5_arc_agi_curriculum_particle_autopilot import (
    candidate_distill_curriculum_env,
    candidate_distill_rows,
    child_run_id,
    decide_candidate_distill_gate,
    summarize_autopilot,
)


def test_child_run_id_uses_parent_prefix() -> None:
    assert child_run_id("curriculum").endswith("_curriculum")


def test_decide_candidate_distill_gate_passes_on_nonnegative_lift() -> None:
    payload = {
        "comparison": {"candidate_distill": {"candidate_distill_rows": 4}},
        "delta_candidate_distill_vs_baseline": {
            "best_selected_delta": 0,
            "best_best_delta": 1,
            "best_valid_rate_delta": -0.01,
        },
    }

    passed, evidence = decide_candidate_distill_gate(payload)

    assert passed is True
    assert evidence["candidate_distill_rows"] == 4
    assert evidence["best_best_delta"] == 1


def test_decide_candidate_distill_gate_fails_without_rows() -> None:
    payload = {
        "comparison": {"candidate_distill": {"candidate_distill_rows": 0}},
        "delta_candidate_distill_vs_baseline": {
            "best_selected_delta": 3,
            "best_best_delta": 3,
            "best_valid_rate_delta": 0.0,
        },
    }

    passed, evidence = decide_candidate_distill_gate(payload)

    assert passed is False
    assert evidence["candidate_distill_rows"] == 0


def test_candidate_distill_curriculum_env_uses_gate_candidate_source() -> None:
    payload = {
        "metadata": {
            "distill_choice": "all_exact",
            "distill_completion_source": "canonical_grid",
        },
        "candidate_source": {"candidate_jsonl": "outputs/stage5/gate/candidates.jsonl"},
    }

    env = candidate_distill_curriculum_env(payload)

    assert env == {
        "STAGE5_ARC_AGI_CANDIDATE_DISTILL_JSONLS": "outputs/stage5/gate/candidates.jsonl",
        "STAGE5_ARC_AGI_CANDIDATE_DISTILL_CHOICE": "all_exact",
        "STAGE5_ARC_AGI_CANDIDATE_DISTILL_COMPLETION_SOURCE": "canonical_grid",
    }


def test_candidate_distill_rows_reads_compact_comparison() -> None:
    payload = {"comparison": {"candidate_distill": {"candidate_distill_rows": 7}}}

    assert candidate_distill_rows(payload) == 7


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
    candidate_gate = {
        "comparison": {"candidate_distill": {"candidate_distill_rows": 5}},
        "delta_candidate_distill_vs_baseline": {
            "best_selected_delta": 1,
            "best_best_delta": 1,
            "best_valid_rate_delta": 0.0,
        },
        "candidate_source": {"candidate_jsonl": "outputs/stage5/gate/candidates.jsonl"},
    }

    compact = summarize_autopilot(curriculum, particle, candidate_gate)

    assert compact["candidate_distillation_passed"] is True
    assert compact["candidate_distillation_jsonl"] == "outputs/stage5/gate/candidates.jsonl"
    assert compact["final_checkpoint"] == "outputs/stage5/run/phase1_step_250.pt"
    assert compact["curriculum_stages"][1]["name"] == "mixed"
    assert compact["particle_passed"] is True
    assert compact["best_replicated_particle_variant"] == "k4_noise001_rep05"
    assert compact["particle_variant_mean_deltas"]["k4_noise001_rep05"]["best_of_k_delta"] == 2.0
