"""Validate and analyze the signed Stage 2B-A score-only autopsy receipts.

This script consumes landed receipts only. It never loads a model, constructs an
optimizer, or contacts CONFIRM/EVAL-E.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LOCK_SHA256 = "c35cda73642d52badb32fda54251ab9d9da47fb5d268fa3106762a64b204d167"
GAMMAS = (0.0, 0.01, 0.02, 0.05)
COMPONENTS = ("standard", "constitutive_off", "fresh_state_each_loop", "inherited_flow_off")
SEED_COLORS = {0: "#247BA0", 1: "#D95F02"}
EXECUTION_SESSIONS = (
    "gpu-a100-s-kkb-usc1f1-39l56xkbw0ya5",
    "gpu-a100-s-kkb-usc1b2-2ki9mjv2avcv5",
    "gpu-a100-s-kkb-ass1c2-sqq094w45cyp",
    "gpu-a100-s-kkb-use1b2-3v3wh1cmymx60",
    "gpu-a100-s-kkb-usc1c1-136xe36zf4pt5",
    "gpu-a100-s-kkb-usc1f1-22l3n9eirgmxj",
    "gpu-a100-s-kkb-usc1c0-34h8yrptjtva0",
    "gpu-a100-s-kkb-ass1c0-uhsx0epxjsgm",
    "gpu-a100-s-kkb-usc1c1-irzgm8lw61wx",
)
RUNNER_COMMITS = (
    "745ee5e7147227760b116a47e3606c80947d29bb",
    "1979f376272fbc7a6c06cf3388fc28e880e22234",
    "7030d80984f27607a652c58db051f362d9224cc4",
    "3ddec4b6142132cee4f54f56cf20b99d2d6dd1fc",
    "ee07684f5b7ab8c23fe2a20bd32ef10b4178e643",
    "508449544228e223dd4608d1fb7100e1686a089e",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def pooled_dev1(cell: Mapping[str, Any]) -> dict[str, Any]:
    battery = cell["battery"]
    pooled = {
        key: sum(int(value[key]) for value in battery.values())
        for key in ("rows", "current_correct", "base_correct", "initialization_correct")
    }
    pooled["delta_vs_base_rows"] = pooled["current_correct"] - pooled["base_correct"]
    pooled["delta_vs_registered_initialization_rows"] = (
        pooled["current_correct"] - pooled["initialization_correct"]
    )
    pooled["current_accuracy"] = pooled["current_correct"] / pooled["rows"]
    return pooled


def summarize_seed(path: Path, expected_seed: int) -> dict[str, Any]:
    source = read_json(path)
    if source.get("status") != "complete_score_only":
        raise AssertionError(f"Seed {expected_seed} did not complete score-only evaluation")
    if int(source.get("seed", -1)) != expected_seed:
        raise AssertionError("Seed identity changed")
    if source.get("lock_sha256") != EXPECTED_LOCK_SHA256:
        raise AssertionError("Signed autopsy lock SHA changed")
    if source.get("optimizer_constructed") or int(source.get("optimizer_steps", -1)) != 0:
        raise AssertionError("Optimizer contact occurred in score-only autopsy")
    if source.get("confirm_scored") or source.get("eval_e_scored"):
        raise AssertionError("A sealed partition was scored")
    if source["runtime"]["gpu"] != "NVIDIA A100-SXM4-40GB":
        raise AssertionError("Registered GPU changed")
    if source["runtime"]["weights_dtype"] != "bfloat16":
        raise AssertionError("Registered weight precision changed")
    if source["runtime"]["attention_backend"] != "sdpa":
        raise AssertionError("Registered attention backend changed")
    if not source["sparse_loop_projection_equivalence"]["all_pass"]:
        raise AssertionError("Sparse-loop projection equivalence failed")
    if not source["incremental_cache_transport"]["all_pass"]:
        raise AssertionError("Incremental-cache transport failed")

    amplitude_source = source["amplitude_response"]
    if not amplitude_source["zero_write_checkpoint_independent"]:
        raise AssertionError("Zero-write prediction identity failed")
    if not amplitude_source["zero_write_full_logit_bit_exact"]:
        raise AssertionError("Zero-write full-logit identity failed")
    zero_replay = amplitude_source.get("zero_write_cross_session_replay", {})
    if zero_replay.get("final_mismatch_count") != 0:
        raise AssertionError("Same-process zero-write replay did not restore identity")
    amplitude: dict[str, dict[str, Any]] = {"initialization": {}, "stop": {}}
    for state in amplitude:
        for gamma in GAMMAS:
            key = str(gamma)
            cell = amplitude_source["cells"][state][key]
            if int(cell["rows"]) != 1024:
                raise AssertionError("DEV-1 amplitude cell row count changed")
            amplitude[state][key] = pooled_dev1(cell)
    for gamma in GAMMAS:
        key = str(gamma)
        stop = amplitude["stop"][key]
        initialization = amplitude["initialization"][key]
        stop["matched_amplitude_delta_rows"] = (
            stop["current_correct"] - initialization["current_correct"]
        )
        stop["delta_vs_initialization_gamma_0p05_rows"] = (
            stop["current_correct"] - amplitude["initialization"]["0.05"]["current_correct"]
        )

    component_source = source["component_attribution"]
    if not component_source["pass_one_identity"]["all_pass"]:
        raise AssertionError("Component pass-one identity failed")
    if not component_source["disabled_component_activation"]["all_pass"]:
        raise AssertionError("Disabled component remained active")
    component = {}
    standard = pooled_dev1(component_source["cells"]["standard"]["dev1"])
    for mode in COMPONENTS:
        cell = component_source["cells"][mode]
        pooled = pooled_dev1(cell["dev1"])
        pooled["delta_vs_standard_rows"] = pooled["current_correct"] - standard["current_correct"]
        component[mode] = {
            "dev1": pooled,
            "dev2": cell["dev2"],
        }

    arm6 = source["correction_field_clusterability"]
    if arm6.get("optimizer_constructed") or arm6.get("parameter_mutation"):
        raise AssertionError("Arm 6 violated its read-only contract")
    for state in ("initialization", "stop"):
        cell = arm6[state]
        if not cell["parameter_versions_unchanged"]:
            raise AssertionError("Arm 6 parameter versions changed")
        if cell["parameter_state_digest_before"] != cell["parameter_state_digest_after"]:
            raise AssertionError("Arm 6 parameter state changed")

    attractor = source["attractor_discriminators"]
    objective = source["objective_task_divergence"]
    onset = source["onset_trajectory"]
    if set(onset["checkpointed_score_endpoints"]) != {"0", "1000"}:
        raise AssertionError("Checkpointed onset endpoint set changed")

    return {
        "seed": expected_seed,
        "source_summary": file_receipt(path),
        "runtime": source["runtime"],
        "state_digests": source["state_digests"],
        "checkpoint_chain": source["checkpoint_chain"],
        "amplitude_response": amplitude,
        "component_attribution": component,
        "arm6": arm6,
        "attractor_discriminators": attractor,
        "objective_task_divergence": objective,
        "onset_trajectory": onset,
        "integrity": {
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
            "zero_write_predictions_exact": True,
            "zero_write_full_logits_exact": True,
            "zero_write_cross_session_replay": zero_replay,
            "component_pass_one_exact": True,
            "disabled_component_activation_exact_zero": True,
            "arm6_parameter_mutation": False,
            "sparse_loop_projection_equivalence": True,
            "incremental_cache_transport": True,
        },
    }


def normalize_svg(path: Path) -> None:
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )


def build_figure(seeds: list[dict[str, Any]], stem: Path) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "svg.fonttype": "none",
            "svg.hashsalt": "paper2-stage2b-autopsy-20260820",
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 12.8), constrained_layout=True)

    x = np.array(GAMMAS)
    for seed in seeds:
        matched = [
            seed["amplitude_response"]["stop"][str(gamma)]["matched_amplitude_delta_rows"]
            for gamma in GAMMAS
        ]
        axes[0, 0].plot(
            x,
            matched,
            marker="o",
            linewidth=2.2,
            color=SEED_COLORS[seed["seed"]],
            label=f"Seed {seed['seed']}",
        )
    axes[0, 0].axhline(0, color="#333333", linewidth=1)
    axes[0, 0].set(
        title="Amplitude response, matched initialization",
        xlabel="Write ceiling gamma",
        ylabel="Stop minus initialization (correct rows)",
    )
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)

    modes = COMPONENTS[1:]
    labels = ["Constitutive off", "Fresh state", "Inherited flow off"]
    bx = np.arange(len(modes))
    width = 0.36
    for index, seed in enumerate(seeds):
        values = [
            seed["component_attribution"][mode]["dev1"]["delta_vs_standard_rows"]
            for mode in modes
        ]
        axes[0, 1].bar(
            bx + (index - 0.5) * width,
            values,
            width,
            color=SEED_COLORS[seed["seed"]],
            label=f"Seed {seed['seed']}",
        )
    axes[0, 1].axhline(0, color="#333333", linewidth=1)
    axes[0, 1].set_xticks(bx, labels, rotation=16)
    axes[0, 1].set(
        title="Component attribution at gamma = 0.05",
        ylabel="Correct-row change vs full stop model",
    )
    axes[0, 1].grid(axis="y", alpha=0.2)
    axes[0, 1].legend(frameon=False)

    loops = np.arange(1, 5)
    for seed in seeds:
        for state, linestyle, alpha in (("initialization", "--", 0.55), ("stop", "-", 1.0)):
            values = seed["component_attribution"]["standard"]["dev2"][
                "per_loop_mean_teacher_token_margin"
            ] if state == "stop" else seed["attractor_discriminators"][state]["margin_summary"][
                "per_loop_mean_teacher_token_margin"
            ]
            axes[1, 0].plot(
                loops,
                values,
                marker="o",
                linestyle=linestyle,
                linewidth=2.0,
                alpha=alpha,
                color=SEED_COLORS[seed["seed"]],
                label=f"Seed {seed['seed']} {state}",
            )
    axes[1, 0].axhline(0, color="#333333", linewidth=1)
    axes[1, 0].set_xticks(loops)
    axes[1, 0].set(
        title="Teacher-token margin across recurrent depth",
        xlabel="Forced loop K",
        ylabel="Mean margin",
    )
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)

    for seed in seeds:
        init_k = seed["attractor_discriminators"]["initialization"]["generative_k_sweep"]
        stop_k = seed["attractor_discriminators"]["stop"]["generative_k_sweep"]
        delta = [int(stop_k[str(k)]["correct"]) - int(init_k[str(k)]["correct"]) for k in loops]
        axes[1, 1].plot(
            loops,
            delta,
            marker="o",
            linewidth=2.2,
            color=SEED_COLORS[seed["seed"]],
            label=f"Seed {seed['seed']}",
        )
    axes[1, 1].axhline(0, color="#333333", linewidth=1)
    axes[1, 1].set_xticks(loops)
    axes[1, 1].set(
        title="Generative capability by forced depth",
        xlabel="Forced loop K",
        ylabel="Stop minus initialization (correct rows)",
    )
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(frameon=False)

    states = ("initialization", "stop")
    state_labels = ("Initialization", "Step 1,000")
    cx = np.arange(len(states))
    width = 0.34
    for index, seed in enumerate(seeds):
        observed = []
        null_mean = []
        for state in states:
            cluster = seed["arm6"][state]["correction_field_clusterability_loop4"][
                "spherical_kmeans"
            ]
            observed.append(cluster["selected_silhouette"])
            null_mean.append(cluster["null_mean"])
        offset = (index - 0.5) * width
        axes[2, 0].bar(
            cx + offset,
            observed,
            width,
            color=SEED_COLORS[seed["seed"]],
            alpha=0.9,
            label=f"Seed {seed['seed']} observed",
        )
        axes[2, 0].scatter(
            cx + offset,
            null_mean,
            marker="_",
            s=180,
            linewidths=2.2,
            color="#222222",
            label="Isotropic-null mean" if index == 0 else None,
            zorder=3,
        )
    axes[2, 0].set_xticks(cx, state_labels)
    axes[2, 0].set(
        title="Arm 6 correction-field clusterability",
        ylabel="Selected spherical-k-means silhouette",
    )
    axes[2, 0].grid(axis="y", alpha=0.2)
    axes[2, 0].legend(frameon=False, fontsize=8)

    for seed in seeds:
        objective = seed["objective_task_divergence"]
        ce_delta = np.array(objective["stop"]["per_loop_ce"]) - np.array(
            objective["initialization"]["per_loop_ce"]
        )
        kl_delta = np.array(objective["stop"]["per_loop_forward_kl"]) - np.array(
            objective["initialization"]["per_loop_forward_kl"]
        )
        axes[2, 1].plot(
            loops,
            ce_delta,
            marker="o",
            linewidth=2.0,
            color=SEED_COLORS[seed["seed"]],
            label=f"Seed {seed['seed']} CE",
        )
        axes[2, 1].plot(
            loops,
            kl_delta,
            marker="s",
            linestyle="--",
            linewidth=1.7,
            color=SEED_COLORS[seed["seed"]],
            alpha=0.7,
            label=f"Seed {seed['seed']} KL",
        )
    axes[2, 1].axhline(0, color="#333333", linewidth=1)
    axes[2, 1].set_xticks(loops)
    axes[2, 1].set(
        title="Heldout objective change at step 1,000",
        xlabel="Forced loop K",
        ylabel="Step 1,000 minus initialization",
    )
    axes[2, 1].grid(alpha=0.2)
    axes[2, 1].legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle("Stage 2B-A score-only autopsy", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        "Two seeds; DEV receipts only; no optimizer, CONFIRM, or EVAL-E contact. "
        "One Colab infrastructure replacement; same A100 40GB class.",
        ha="center",
        color="#555555",
        fontsize=8.5,
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": "2026-08-20"})
    normalize_svg(svg)
    fig.savefig(png, dpi=190, bbox_inches="tight", metadata={"Creation Time": "2026-08-20"})
    plt.close(fig)
    return [svg, png]


def analyze(seed_paths: list[Path], output: Path, figure_stem: Path) -> dict[str, Any]:
    if len(seed_paths) != 2:
        raise AssertionError("Exactly two seed summaries are required")
    seeds = [summarize_seed(path, index) for index, path in enumerate(seed_paths)]
    h_b_by_seed = {}
    for seed in seeds:
        registered = seed["amplitude_response"]["initialization"]["0.05"]["current_correct"]
        h_b_by_seed[str(seed["seed"])] = {
            str(gamma): seed["amplitude_response"]["stop"][str(gamma)]["current_correct"]
            - registered
            for gamma in GAMMAS[:-1]
        }
    h_b_common_gamma = [
        gamma
        for gamma in GAMMAS[:-1]
        if all(h_b_by_seed[str(seed)][str(gamma)] > 0 for seed in (0, 1))
    ]

    component_effects = {
        mode: {
            str(seed["seed"]): seed["component_attribution"][mode]["dev1"][
                "delta_vs_standard_rows"
            ]
            for seed in seeds
        }
        for mode in COMPONENTS[1:]
    }
    figures = build_figure(seeds, figure_stem)
    result = {
        "kind": "paper2_stage2b_autopsy_analysis_v1",
        "analysis_date": "2026-08-20",
        "status": "complete_score_only_analysis",
        "scope": "Two signed Stage 2B-A seeds; DEV-1, fixed DEV-2 subsample, heldout training slice; no sealed partitions",
        "lock_sha256": EXPECTED_LOCK_SHA256,
        "seeds": seeds,
        "registered_decision_read": {
            "h_b_magnitude": {
                "delta_vs_registered_initialization_by_seed_gamma": h_b_by_seed,
                "common_lower_gamma_beating_registered_initialization": h_b_common_gamma,
                "supported_both_seeds": bool(h_b_common_gamma),
                "successor_if_supported": "radius_control_successor",
            },
            "h_c_constitutive": {
                "component_effects_vs_standard": component_effects,
                "constitutive_restores_more_than_inherited_flow_both_seeds": all(
                    component_effects["constitutive_off"][str(seed)]
                    > component_effects["inherited_flow_off"][str(seed)]
                    for seed in (0, 1)
                ),
                "materiality_threshold": "not numerically defined in signed lock; report effect sizes",
                "successor_if_supported": "gated_additive_constructor_successor",
            },
            "h_a_attractor": {
                "status": "descriptive_composite_requires_scientific_read",
                "inputs": {
                    str(seed["seed"]): {
                        "margin_correlation": seed["attractor_discriminators"]["stop"][
                            "k1_k4_margin_correlation"
                        ],
                        "state_similarity": seed["attractor_discriminators"]["stop"][
                            "state_similarity"
                        ],
                        "objective_initialization": seed["objective_task_divergence"][
                            "initialization"
                        ],
                        "objective_stop": seed["objective_task_divergence"]["stop"],
                    }
                    for seed in seeds
                },
                "successor_if_supported": "task_preservation_anchor_required",
            },
        },
        "arm6": {
            str(seed["seed"]): {
                "initialization_clusterability": seed["arm6"]["initialization"][
                    "correction_field_clusterability_loop4"
                ],
                "stop_clusterability": seed["arm6"]["stop"][
                    "correction_field_clusterability_loop4"
                ],
                "mean_field_confirmation": seed["arm6"]["mean_field_confirmation"],
            }
            for seed in seeds
        },
        "integrity": {
            "all_seed_receipts_pass": True,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
            "same_hardware_configuration": True,
            "same_session_for_all_cells": False,
            "execution_deviation": {
                "signed_condition": "same_session_for_all_cells",
                "reason": (
                    "Colab infrastructure and DriveFS interruptions required eight resumptions "
                    "before batch completion"
                ),
                "session_endpoints_in_order": list(EXECUTION_SESSIONS),
                "runner_commits_in_order": list(RUNNER_COMMITS),
                "repair_scope": (
                    "The second runner commit archives cross-session gamma-zero files and "
                    "replays both checkpoint states in one evaluator process before retaining "
                    "the hard zero-write identity gate. The third runner commit moves hot-path "
                    "receipt writes to local scratch and mirrors them durably every five minutes. "
                    "The fourth adds strict correction-field artifact resume, the fifth replaces "
                    "the algebraically identical pairwise cosine silhouette implementation with "
                    "its vectorized form, and the sixth adds per-batch atomic K-sweep resume plus "
                    "validated reuse of complete registered cells."
                ),
                "scope": "Resume used the same A100-SXM4-40GB class, bfloat16 weights, and SDPA backend",
                "scientific_substitution": False,
            },
        },
        "figures": [file_receipt(path) for path in figures],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-summary", action="append", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/stage2b_autopsy_20260820/analysis/analysis_summary.json",
    )
    parser.add_argument(
        "--figure-stem",
        type=Path,
        default=ROOT / "docs/figures/stage2b_autopsy_20260820",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(args.seed_summary, args.output, args.figure_stem)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
