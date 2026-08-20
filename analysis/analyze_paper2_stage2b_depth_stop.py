"""Analyze the registered Stage 2B-D step-1,000 hard-stop receipts.

This script consumes only landed DEV receipts. It does not load a model,
resume training, or contact CONFIRM/EVAL-E.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/stage2b_depth_20260819/analysis/analysis_summary.json"
DEFAULT_FIGURE = ROOT / "docs/figures/stage2b_depth_step1000_stop_20260819"
EXPECTED_LOCK_SHA256 = "30a97e175200d3a58bc0cc0c200acec301d3a4f4cd662466d4c3491b9f816597"
BATTERIES = ("arc_challenge", "arc_easy", "gsm8k", "mbpp", "mmlu", "tier1")
BATTERY_LABELS = {
    "arc_challenge": "ARC-C",
    "arc_easy": "ARC-E",
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
    "mmlu": "MMLU",
    "tier1": "Tier-1",
}
SEED_COLORS = {0: "#247BA0", 1: "#D95F02"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalize_svg(path: Path) -> None:
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )


def summarize_seed(path: Path, expected_seed: int) -> dict[str, Any]:
    summary = read_json(path)
    if summary["seed"] != expected_seed:
        raise AssertionError(f"Expected seed {expected_seed}, found {summary['seed']}")
    if summary["lock_sha256"] != EXPECTED_LOCK_SHA256:
        raise AssertionError("Stage 2B signed-lock SHA changed")
    if summary["status"] != "stopped" or summary["stop_reason"] != "dev1_hard_floor":
        raise AssertionError("Seed did not exit through the registered DEV-1 hard floor")
    if summary["step"] != 1000 or summary["target_step"] != 5000:
        raise AssertionError("Stage 2B stop/target step changed")
    if summary["confirm_scored"] or summary["eval_e_scored"]:
        raise AssertionError("A sealed partition was scored")
    if len(summary["history"]) != 1:
        raise AssertionError("Expected exactly one registered look")

    look = summary["history"][0]
    if look["step"] != 1000 or look["look"] != 1 or look["stage"] != "M2":
        raise AssertionError("First-look identity changed")
    if look["pass_one_max_abs_difference"] != 0.0:
        raise AssertionError("Pass-one identity failed")
    if look["confirm_scored"] or look["eval_e_scored"]:
        raise AssertionError("A sealed partition was scored at the look")
    if not look["dev1"]["both_comparators_reported"]:
        raise AssertionError("Both-comparator reporting is incomplete")
    if look["dev1"]["safety"]["pass"]:
        raise AssertionError("DEV-1 safety unexpectedly passed")
    if look["finite_horizon"]["catastrophe"]:
        raise AssertionError("Finite-horizon catastrophe flag unexpectedly fired")
    if not summary["last_gradient_audit"]["pass"]:
        raise AssertionError("Gradient audit failed")
    if summary["last_gradient_audit"]["missing_active"]:
        raise AssertionError("Active trainable gradients are missing")

    battery = look["dev1"]["battery"]
    if tuple(battery) != BATTERIES:
        raise AssertionError("DEV-1 battery order or composition changed")
    pooled = {
        key: sum(int(cell[key]) for cell in battery.values())
        for key in ("rows", "base_correct", "initialization_correct", "current_correct")
    }
    if pooled["rows"] != 1024:
        raise AssertionError("DEV-1 row count changed")
    pooled.update(
        {
            "delta_vs_base_rows": pooled["current_correct"] - pooled["base_correct"],
            "delta_vs_initialization_rows": (
                pooled["current_correct"] - pooled["initialization_correct"]
            ),
            "current_accuracy": pooled["current_correct"] / pooled["rows"],
            "base_accuracy": pooled["base_correct"] / pooled["rows"],
            "initialization_accuracy": pooled["initialization_correct"] / pooled["rows"],
        }
    )

    generative_names = ("gsm8k", "mbpp", "tier1")
    multiple_choice_names = ("arc_challenge", "arc_easy", "mmlu")

    def family(names: tuple[str, ...]) -> dict[str, int]:
        return {
            key: sum(int(battery[name][key]) for name in names)
            for key in ("rows", "base_correct", "initialization_correct", "current_correct")
        }

    margins = [float(value) for value in look["dev2"]["per_loop_mean_teacher_token_margin"]]
    transitions = [margins[index + 1] - margins[index] for index in range(3)]
    if not all(value < 0 for value in transitions):
        raise AssertionError("Additional-loop margins are not strictly decreasing")
    if margins[-1] >= 0:
        raise AssertionError("Loop-4 mean margin did not cross below zero")

    return {
        "seed": expected_seed,
        "source_summary": file_receipt(path),
        "status": summary["status"],
        "stop_reason": summary["stop_reason"],
        "step": summary["step"],
        "target_step": summary["target_step"],
        "frozen_digest": summary["frozen_digest"],
        "teacher_cache_index_sha256": summary["teacher_cache_index_sha256"],
        "ema_checkpoint": look["ema_checkpoint"],
        "pooled_dev1": pooled,
        "battery": battery,
        "safety": look["dev1"]["safety"],
        "generative_family": family(generative_names),
        "multiple_choice_family": family(multiple_choice_names),
        "dev2": {
            "rows": look["dev2"]["rows"],
            "per_loop_mean_teacher_token_margin": margins,
            "transition_means": look["dev2"]["transition_means"],
            "all_additional_loops_reduce_margin": True,
            "loop4_margin_below_zero": True,
        },
        "objective_components": look["objective_components"],
        "pass_one_max_abs_difference": look["pass_one_max_abs_difference"],
        "finite_horizon": look["finite_horizon"],
        "gradient_audit": {
            "pass": summary["last_gradient_audit"]["pass"],
            "finite_parameter_tensors": summary["last_gradient_audit"][
                "finite_parameter_tensors"
            ],
            "missing_active": summary["last_gradient_audit"]["missing_active"],
            "missing_expected_count": len(summary["last_gradient_audit"]["missing_expected"]),
            "stage": summary["last_gradient_audit"]["stage"],
        },
        "confirm_scored": False,
        "eval_e_scored": False,
    }


def build_figure(seeds: list[dict[str, Any]], stem: Path) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "svg.fonttype": "none",
            "svg.hashsalt": "paper2-stage2b-stop-20260819",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)

    labels = ["Seed 0", "Seed 1"]
    x = np.arange(2)
    width = 0.24
    for offset, key, label, color in (
        (-width, "base_correct", "Frozen base", "#9AA0A6"),
        (0.0, "initialization_correct", "Initialization", "#4C78A8"),
        (width, "current_correct", "Step 1,000", "#D1495B"),
    ):
        values = [seed["pooled_dev1"][key] for seed in seeds]
        axes[0].bar(x + offset, values, width, label=label, color=color)
        for xpos, value in zip(x + offset, values):
            axes[0].text(xpos, value + 8, str(value), ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 570)
    axes[0].set(title="DEV-1 capability at the first look", ylabel="Correct rows of 1,024")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    bx = np.arange(len(BATTERIES))
    bar_width = 0.36
    for index, seed in enumerate(seeds):
        values = [seed["battery"][name]["delta_vs_initialization_rows"] for name in BATTERIES]
        axes[1].bar(
            bx + (index - 0.5) * bar_width,
            values,
            bar_width,
            label=f"Seed {seed['seed']}",
            color=SEED_COLORS[seed["seed"]],
        )
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_xticks(bx, [BATTERY_LABELS[name] for name in BATTERIES], rotation=25)
    axes[1].set(title="Losses are largest on generation", ylabel="Correct-row change vs initialization")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)

    loops = np.arange(1, 5)
    for seed in seeds:
        axes[2].plot(
            loops,
            seed["dev2"]["per_loop_mean_teacher_token_margin"],
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=SEED_COLORS[seed["seed"]],
            label=f"Seed {seed['seed']}",
        )
    axes[2].axhline(0, color="#333333", linewidth=1)
    axes[2].set_xticks(loops)
    axes[2].set(
        title="Every additional loop reduces margin",
        xlabel="Forced recurrent loop",
        ylabel="Mean teacher-token margin on DEV-2",
    )
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].annotate(
        "K=4 below zero",
        xy=(4, np.mean([seed["dev2"]["per_loop_mean_teacher_token_margin"][3] for seed in seeds])),
        xytext=(3.05, -0.65),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=8,
        color="#444444",
    )

    fig.suptitle(
        "Stage 2B-D: replicated registered hard stop at step 1,000",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.025,
        "Pass-one identity remained exact; CONFIRM and EVAL-E were not scored.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": "2026-08-19"})
    normalize_svg(svg)
    fig.savefig(png, dpi=190, bbox_inches="tight", metadata={"Creation Time": "2026-08-19"})
    plt.close(fig)
    return [svg, png]


def analyze(seed_paths: list[Path], output: Path, figure_stem: Path) -> dict[str, Any]:
    if len(seed_paths) != 2:
        raise AssertionError("Exactly two seed summaries are required")
    seeds = [summarize_seed(path, expected_seed=index) for index, path in enumerate(seed_paths)]
    if seeds[0]["teacher_cache_index_sha256"] != seeds[1]["teacher_cache_index_sha256"]:
        raise AssertionError("Seeds used different teacher caches")

    figures = build_figure(seeds, figure_stem)
    result = {
        "kind": "paper2_stage2b_depth_registered_stop_analysis_v1",
        "analysis_date": "2026-08-19",
        "status": "complete_registered_early_stop",
        "registered_verdict": "REPLICATED_DEV1_HARD_FLOOR_STOP_AT_STEP_1000",
        "scope": (
            "Two registered Stage 2B-D seeds, M2 through step 1,000, DEV-1 safety and "
            "DEV-2 margin telemetry; CONFIRM and EVAL-E sealed"
        ),
        "lock_sha256": EXPECTED_LOCK_SHA256,
        "seeds": seeds,
        "cross_seed": {
            "both_stopped_at_first_registered_look": True,
            "both_stop_reason": "dev1_hard_floor",
            "step_5000_adjudication_eligible": False,
            "optimizer_continuation_authorized": False,
            "pass_one_identity_exact_both_seeds": True,
            "gradient_audit_pass_both_seeds": True,
            "finite_horizon_catastrophe_any_seed": False,
            "all_additional_loops_reduce_margin_both_seeds": True,
            "loop4_mean_margin_below_zero_both_seeds": True,
            "mean_dev1_current_correct": float(
                np.mean([seed["pooled_dev1"]["current_correct"] for seed in seeds])
            ),
            "mean_dev1_delta_vs_initialization_rows": float(
                np.mean([seed["pooled_dev1"]["delta_vs_initialization_rows"] for seed in seeds])
            ),
            "mean_dev1_delta_vs_base_rows": float(
                np.mean([seed["pooled_dev1"]["delta_vs_base_rows"] for seed in seeds])
            ),
        },
        "interpretation": {
            "supported": [
                "The registered DEV-1 safety floor failed severely and independently in both seeds.",
                "The frozen pass-one serving path remained bit-exact, excluding frozen-substrate corruption.",
                "Finite gradients and near-unit finite-horizon gains exclude numerical explosion at the stop.",
                "Each additional recurrent loop reduced the mean teacher-token margin in both seeds.",
                "Generation batteries suffered more than multiple-choice batteries at this checkpoint.",
            ],
            "not_established": [
                "Which individual M2 component caused the harmful recurrent direction.",
                "Whether a smaller read amplitude, task rehearsal, or a different objective would rescue the route.",
                "Any claim about CONFIRM, EVAL-E, or out-of-DEV generalization.",
            ],
        },
        "sealed_partitions": {"confirm_scored": False, "eval_e_scored": False},
        "figure_receipts": [file_receipt(path) for path in figures],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-0", type=Path, required=True)
    parser.add_argument("--seed-1", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    analyze([args.seed_0, args.seed_1], args.output, args.figure_stem)
    print(args.output)
    print(args.figure_stem.with_suffix(".svg"))
    print(args.figure_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
