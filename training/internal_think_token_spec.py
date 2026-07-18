"""Machine-readable Phase T0 contract for token-pathway learned halting."""

from __future__ import annotations

from typing import Any, Iterable


INTERNAL_CONTROL_TOKENS = (
    "<|recur_continue|>",
    "<|recur_stop|>",
    "<|recur_readout|>",
)


def validate_tokenizer_preflight(
    *,
    existing_vocabulary: Iterable[str],
    added_token_count: int,
) -> dict[str, Any]:
    """Fail if the base tokenizer already owns a planned control token."""

    vocabulary = set(map(str, existing_vocabulary))
    collisions = sorted(set(INTERNAL_CONTROL_TOKENS) & vocabulary)
    if collisions:
        raise AssertionError(f"Internal control-token collision: {collisions}")
    if int(added_token_count) != len(INTERNAL_CONTROL_TOKENS):
        raise AssertionError(
            f"Expected exactly {len(INTERNAL_CONTROL_TOKENS)} added tokens, "
            f"observed {added_token_count}"
        )
    return {
        "collisions": collisions,
        "added_token_count": int(added_token_count),
        "tokens": list(INTERNAL_CONTROL_TOKENS),
    }


def phase_t0_spec() -> dict[str, Any]:
    return {
        "kind": "internal_think_token_phase0_spec",
        "status": "preparation_only_not_authorized_for_training",
        "dependencies": {
            "paper1_experimental_closure": True,
            "adapter_budget_arm_e_landed": True,
            "first_phase_g_alpha_verdict_landed": True,
        },
        "tokens": {
            "values": list(INTERNAL_CONTROL_TOKENS),
            "count": len(INTERNAL_CONTROL_TOKENS),
            "base_tokenizer_collision_check": "required_at_launch",
            "resize_policy": (
                "resize input embeddings and LM head together while preserving the base "
                "model's global tie_word_embeddings policy; initialize each new input/output "
                "row identically and record the added parameter count"
            ),
            "visible_generation_policy": (
                "mask all control-token logits from user-visible autoregressive generation"
            ),
            "control_path": (
                "read continue/stop logits directly at the per-loop control position; "
                "intercept the decision before text decoding"
            ),
            "readout_token_role": (
                "dedicated hidden-state probe/readout anchor, never a visible answer token"
            ),
        },
        "identity_contract": {
            "control_inactive_max_loops_1_max_abs_diff": 1e-3,
            "control_inactive_preserves_loop_count": True,
            "control_tokens_never_emitted": True,
            "control_decision_identity_required_for_K1_width_parity": True,
        },
        "logging_contract": {
            "requested_loops": True,
            "executed_loops": True,
            "per_loop_continue_stop_logits": True,
            "selected_stop_loop": True,
            "forced_vs_self_halted_pair_id": True,
            "per_trajectory_unaveraged_control_readout": True,
        },
        "phase_t1": {
            "authorization": (
                "only_after Paper 1 closure, Arm E, and first Phase G-alpha verdict"
            ),
            "lineages": [
                "adapter_budget_from_fresh_base_surgery",
                "full_block_from_fresh_base_surgery",
            ],
            "training_mix": {
                "control_targets": "segmented_step_boundaries",
                "outcome_loss": True,
                "rehearsal_fraction": 0.30,
            },
            "gates": {
                "chain_diagonal_within_reference_points": 0.03,
                "self_halted_within_forced_points": 0.03,
                "control_selection_accuracy_each_depth": 0.90,
                "causal_override_falsification_required": True,
            },
            "claim_if_green": (
                "token-pathway halting succeeds where the tested pooled-head halting path failed"
            ),
        },
        "phase_t2": {
            "natural_trace_policy": (
                "traces provide segmented step counts and verified final answers only; "
                "they never supervise hidden latent states"
            ),
            "budget_force": (
                "compare expected executed transitions under self-halting against matched "
                "forced-compute references with a calibration-locked margin"
            ),
        },
        "phase_t3": {"named": True, "authorized": False},
        "forbidden": [
            "pooled_head_halting",
            "selector_sweep",
            "rank_ladder",
            "phase_t3_training",
            "averaging_control_readouts_across_trajectories",
        ],
    }


def natural_trace_survey() -> dict[str, Any]:
    """Current source-card survey; final row-level verification remains a T0 task."""

    return {
        "kind": "natural_reasoning_trace_dataset_survey",
        "as_of": "2026-07-18",
        "datasets": [
            {
                "id": "open-r1/OpenR1-Math-220k",
                "license": "apache-2.0",
                "trace_type": "DeepSeek-R1 mathematical reasoning traces",
                "answer_verifiability": (
                    "strong: Math Verify for most rows; model judge for a minority; "
                    "at least one reported-correct trace per problem"
                ),
                "phase_t2_priority": "primary_candidate",
                "url": "https://huggingface.co/datasets/open-r1/OpenR1-Math-220k",
            },
            {
                "id": "bespokelabs/Bespoke-Stratos-17k",
                "license": "apache-2.0",
                "trace_type": "DeepSeek-R1 math, code, science, and puzzle traces",
                "answer_verifiability": (
                    "mixed-strong: rejection-filtered math and executable code; retain "
                    "verifier metadata and reverify final answers locally"
                ),
                "phase_t2_priority": "secondary_candidate",
                "url": "https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k",
            },
            {
                "id": "simplescaling/s1K-1.1",
                "license": "mit",
                "trace_type": "Gemini and DeepSeek mathematical thinking trajectories",
                "answer_verifiability": (
                    "moderate-strong: ground-truth solution plus grader fields; locally "
                    "reverify normalized final answers"
                ),
                "phase_t2_priority": "high_quality_small_candidate",
                "url": "https://huggingface.co/datasets/simplescaling/s1K-1.1",
            },
            {
                "id": "GAIR/LIMO-v2",
                "license": "apache-2.0",
                "trace_type": "small curated long-form mathematical solutions",
                "answer_verifiability": (
                    "moderate: answer-bearing mathematical records; locally reverify"
                ),
                "phase_t2_priority": "high_quality_small_candidate",
                "url": "https://huggingface.co/datasets/GAIR/LIMO-v2",
            },
            {
                "id": "AI-MO/NuminaMath-CoT",
                "license": "apache-2.0_metadata",
                "trace_type": "large heterogeneous mathematical CoT collection",
                "answer_verifiability": (
                    "mixed: final-answer formatting is present, but source heterogeneity "
                    "requires source-aware license and verifier filters"
                ),
                "phase_t2_priority": "filtered_rehearsal_only_until_audit",
                "url": "https://huggingface.co/datasets/AI-MO/NuminaMath-CoT",
            },
            {
                "id": "Glint-Research/Fable-5-traces",
                "license": "agpl-3.0",
                "trace_type": "agentic coding and tool-use traces",
                "answer_verifiability": (
                    "weak for this use: no uniform mathematical final-answer verifier; "
                    "task success would require repository/test reconstruction"
                ),
                "phase_t2_priority": "exclude_from_primary_halting_curriculum",
                "url": "https://huggingface.co/datasets/Glint-Research/Fable-5-traces",
            },
        ],
        "row_level_acceptance": [
            "license and upstream-source metadata retained",
            "final answer independently verifiable",
            "trace segment boundaries reproducible",
            "no hidden-state or latent target inferred from prose",
            "deduplicate against evaluation benchmarks",
        ],
    }
