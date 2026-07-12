"""Machine-readable Phase G-alpha architecture and preregistration contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ALLOWED_TRAINABLE_PREFIXES = (
    "phase_g_prior_head.",
    "phase_g_posterior_head.",
    "phase_g_injection_scale",
)


def is_phase_g_trainable(name: str) -> bool:
    return any(str(name).startswith(prefix) for prefix in ALLOWED_TRAINABLE_PREFIXES)


def assert_frozen_parameter_contract(named_parameters: Iterable[tuple[str, Any]]) -> dict[str, list[str]]:
    allowed_trainable: list[str] = []
    unexpected_trainable: list[str] = []
    for name, parameter in named_parameters:
        if bool(parameter.requires_grad):
            if is_phase_g_trainable(name):
                allowed_trainable.append(str(name))
            else:
                unexpected_trainable.append(str(name))
    if unexpected_trainable:
        raise AssertionError(
            "Phase G-alpha freezes the deterministic substrate; unexpected trainable parameters: "
            + ", ".join(unexpected_trainable[:20])
        )
    required_groups = {
        "prior": any(name.startswith("phase_g_prior_head.") for name in allowed_trainable),
        "posterior": any(name.startswith("phase_g_posterior_head.") for name in allowed_trainable),
        "injection_scale": any(name.startswith("phase_g_injection_scale") for name in allowed_trainable),
    }
    missing = [name for name, present in required_groups.items() if not present]
    if missing:
        raise AssertionError(f"Phase G-alpha trainable groups missing: {missing}")
    return {"allowed_trainable": allowed_trainable, "unexpected_trainable": unexpected_trainable}


def assert_frozen_gradients_zero(named_parameters: Iterable[tuple[str, Any]]) -> None:
    violations: list[str] = []
    for name, parameter in named_parameters:
        if is_phase_g_trainable(name) or parameter.grad is None:
            continue
        nonzero = int(parameter.grad.detach().count_nonzero().item())
        if nonzero:
            violations.append(f"{name}:{nonzero}")
    if violations:
        raise AssertionError(
            "Frozen deterministic substrate received nonzero gradients: " + ", ".join(violations[:20])
        )


def preregistration_payload() -> dict[str, Any]:
    return {
        "kind": "phase_g_alpha_preregistration",
        "status": "forms_locked_numeric_margins_pending_power_calculation",
        "substrate_gate": {
            "required": (
                "constructive_abductive_injective_pass_plus_arbitrary_N24_calibration_competence_"
                "plus_green_guardrail"
            ),
            "checkpoint": "post_arbitrary_family_deterministic_keeper",
            "checkpoint_sha256_rule": (
                "resolve_from_the_final_green_deterministic_receipt_and_assert_exact_match_at_launch"
            ),
            "distribution_seam": (
                "the running N20 constructive fan is a screening gate; it cannot alone unlock "
                "stochastic training on arbitrary N24 functions"
            ),
        },
        "architecture": {
            "frozen": "entire_deterministic_recurrent_qwen_substrate",
            "trainable": list(ALLOWED_TRAINABLE_PREFIXES),
            "latent_location": "high_level_reentry_state_only",
            "deterministic_update": "u_t_untouched",
            "reentry": "h_t = u_t + softplus(injection_scale) * fixed_projection(z_t)",
            "latent_dimension": 64,
            "latent_projection": "fixed_seeded_orthonormal_buffer",
            "injection_scale_initial": 1e-3,
            "posterior_conditioning": (
                "current_reentry_state_plus_frozen_embedding_of_gold_next_symbolic_state; "
                "multimodal rows use the uniformly sampled valid chain stored in the row"
            ),
            "block_gradient_assertion": "identically_zero_after_every_backward",
        },
        "objective": {
            "task_loss": "per_loop_chain_CE_plus_final_valid_answer_CE",
            "kl": "per_loop_KL_q_p",
            "kl_balance": 0.8,
            "kl_coefficient_sweep": [1e-4, 1e-3, 1e-2],
            "ema": {"required": True, "decay": 0.999},
            "dose_levers": ["kl_coefficient", "training_steps"],
        },
        "frozen_evaluation": {
            "n_symbols": 24,
            "depths": [1, 2, 3, 4],
            "rows_per_preimage_stratum": 128,
            "preimage_strata": {"unique": "1", "small": "2-4", "large": ">=5"},
            "splits": ["calibration", "test"],
            "oracle": "independent_forward_orbit_enumeration",
        },
        "primary_gate_form": {
            "metric": "paired_per_instance_exact_oracle_coverage_at_K",
            "sample_counts": [1, 2, 4, 8, 20],
            "alpha": 0.05,
            "comparator_1": "deterministic_keeper_answer_head_sampling_at_matched_K_and_entropy",
            "comparator_2": "one_deterministic_trajectory_at_K_times_T_loops",
            "numeric_margin": "TODO_POWER_CALCULATION_LAST_BLANK_BEFORE_LAUNCH",
        },
        "secondary_metrics": [
            "valid_sample_rate",
            "unique_valid_count",
            "full_coverage_rate",
            "duplicate_rate",
            "coverage_by_preimage_stratum",
            "coverage_by_depth",
            "posterior_prior_KL_by_loop",
            "posterior_collapse_fraction",
        ],
        "failure_interpretations": {
            "no_win_vs_temperature_K": (
                "guided latent width adds nothing beyond output sampling on this substrate"
            ),
            "win_vs_temperature_but_loss_vs_iso_compute": (
                "trajectory width is real but uneconomical relative to deterministic depth"
            ),
            "win_vs_both": "open_G_beta",
        },
        "deferred_until_G_alpha_win": ["LPRM", "per_trajectory_halting", "SVGD"],
        "power_calculation_todo": {
            "only_remaining_preregistration_blank": True,
            "use_split": "calibration",
            "paired_n": 384,
            "stratify_by": "preimage_stratum",
            "lock_before_test_split_or_latent_run_is_scored": True,
        },
    }


def write_preregistration(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preregistration_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
