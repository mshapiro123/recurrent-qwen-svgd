"""Build the Stage 2B-S stopped-wave provenance receipt and figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


PREDICTION_KEYS = ("generated_token_ids", "prediction", "augmented_correct")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correct(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(bool(row["augmented_correct"]) for row in rows)


def compare_predictions(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    left_by_id = {str(row["item_id"]): row for row in left}
    right_by_id = {str(row["item_id"]): row for row in right}
    shared = sorted(set(left_by_id) & set(right_by_id))
    mismatches = []
    for item_id in shared:
        changed = [
            key for key in PREDICTION_KEYS if left_by_id[item_id].get(key) != right_by_id[item_id].get(key)
        ]
        if changed:
            mismatches.append({"item_id": item_id, "keys": changed})
    return {
        "left_rows": len(left),
        "right_rows": len(right),
        "shared_rows": len(shared),
        "left_only": len(set(left_by_id) - set(right_by_id)),
        "right_only": len(set(right_by_id) - set(left_by_id)),
        "prediction_exact_rows": len(shared) - len(mismatches),
        "prediction_mismatch_rows": len(mismatches),
        "all_shared_predictions_exact": not mismatches,
        "first_mismatches": mismatches[:10],
    }


def artifact(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "sha256": sha256_file(path),
        "rows": len(rows),
        "correct": correct(rows),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    k_paths = {
        loop: args.preflight_dir / f"k_sweep__stage2bs_preflight__k{loop}.jsonl"
        for loop in (1, 2, 3)
    }
    live = {loop: read_jsonl(path) for loop, path in k_paths.items()}
    autopsy_refs = {
        loop: read_jsonl(args.autopsy_dir / f"k{loop}.jsonl") for loop in (1, 2, 3)
    }
    live_k4 = read_jsonl(args.live_k4)
    live_k4_partial = read_jsonl(args.live_k4_partial)
    autopsy_k4 = read_jsonl(args.autopsy_k4)
    amplitude = read_jsonl(args.amplitude)
    amplitude_by_id = {str(row["item_id"]): row for row in amplitude}
    amplitude_generation = [amplitude_by_id[str(row["item_id"])] for row in autopsy_k4]
    prelude2 = json.loads(args.prelude2.read_text(encoding="utf-8"))

    expected = [162, 10, 2, 160]
    observed = [correct(live[loop]) for loop in (1, 2, 3)] + [correct(live_k4)]
    receipt = {
        "kind": "paper2_stage2bs_stopped_wave_analysis_v1",
        "status": "prelude1_stopped_prelude2_complete",
        "prelude1": {
            "preflight_expected_correct_by_k": expected,
            "preflight_observed_correct_by_k": observed,
            "mandatory_gate_pass": observed == expected,
            "probe_cells_run": False,
            "artifacts": {
                **{f"live_k{loop}": artifact(k_paths[loop], live[loop]) for loop in (1, 2, 3)},
                **{
                    f"autopsy_k{loop}_reference": artifact(
                        args.autopsy_dir / f"k{loop}.jsonl", autopsy_refs[loop]
                    )
                    for loop in (1, 2, 3)
                },
                "live_k4_quarantined": artifact(args.live_k4, live_k4),
                "live_k4_fresh_partial": artifact(args.live_k4_partial, live_k4_partial),
                "autopsy_k4_reference": artifact(args.autopsy_k4, autopsy_k4),
                "p35_amplitude_cell": artifact(args.amplitude, amplitude),
            },
            "provenance_tests": {
                **{
                    f"live_k{loop}_vs_autopsy_k{loop}": compare_predictions(
                        live[loop], autopsy_refs[loop]
                    )
                    for loop in (1, 2, 3)
                },
                "autopsy_k4_vs_p35_amplitude_generation": compare_predictions(
                    autopsy_k4, amplitude_generation
                ),
                "live_stage2b_k4_vs_autopsy_k4": compare_predictions(live_k4, autopsy_k4),
                "fresh_live_partial_vs_autopsy_k4": compare_predictions(
                    live_k4_partial, autopsy_k4
                ),
            },
            "scorer_provenance": {
                "k1_to_k3": "Stage2BTaskInferenceGraph live recomputation",
                "registered_autopsy_k4": "precomputed P3.5 amplitude cell reused by _k_sweep",
                "failed_live_k4": "Stage2BTaskInferenceGraph live recomputation",
            },
            "interpretation": (
                "The registered K-sweep combined two scoring graphs. The live Stage 2B graph "
                "reproduced K1-K3 but not the imported P3.5 K4 cell, so the authorized "
                "Stage 2B perturbation probes have no valid 160-correct native baseline."
            ),
        },
        "prelude2": {
            "receipt_path": (
                "receipts/prelude2/prelude2.json under the durable Stage 2B-S run root"
            ),
            "receipt_sha256": sha256_file(args.prelude2),
            "estimator": prelude2["F1"]["estimator"],
            "ratios": prelude2["F1"]["ratios"],
            "verdict": prelude2["F1"]["verdict"],
            "same_shape_asserted": prelude2["F1"]["same_shape_asserted"],
            "seed_raw_values": [row["F1_raw_values"] for row in prelude2["seeds"]],
            "optimizer_steps": prelude2["optimizer_steps"],
            "confirm_scored": prelude2["confirm_scored"],
            "eval_e_scored": prelude2["eval_e_scored"],
        },
        "sealed_partitions_untouched": True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x = np.arange(1, 5)
    width = 0.34
    axes[0].bar(x - width / 2, expected, width, label="Registered reference", color="#4B6B8A")
    axes[0].bar(x + width / 2, observed, width, label="Live Stage 2B", color="#C65D3B")
    axes[0].set_xticks(x, [f"K={value}" for value in x])
    axes[0].set_ylabel("Correct rows (of 461)")
    axes[0].set_title("A. Mandatory K-sweep reproduction")
    axes[0].legend(frameon=False, loc="upper center")
    axes[0].set_ylim(0, 180)
    for position, value in zip(x, observed):
        axes[0].text(position + width / 2, value + 4, str(value), ha="center", fontsize=9)

    ratios = np.asarray(prelude2["F1"]["ratios"], dtype=float)
    axes[1].bar([0, 1], ratios, color=["#2F7D6D", "#5A8F7B"], width=0.55)
    axes[1].axhline(0.25, color="#A23E48", linestyle="--", linewidth=1.5, label="STARVED ceiling")
    axes[1].axhline(0.75, color="#5B5F97", linestyle=":", linewidth=1.8, label="NOT_STARVED floor")
    axes[1].set_xticks([0, 1], ["Seed 0", "Seed 1"])
    axes[1].set_ylabel("||Delta W_P|| / ||Delta W_H||")
    axes[1].set_title("B. Prompt-path movement (F1)")
    axes[1].set_ylim(0, 2.75)
    axes[1].legend(frameon=False, loc="upper right")
    for position, value in enumerate(ratios):
        axes[1].text(position, value + 0.08, f"{value:.3f}", ha="center", fontsize=10)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
    figure.suptitle("Stage 2B-S prelude wave: gate failure and independent desk result", fontsize=14)
    figure.tight_layout()
    args.figure_svg.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure_svg, bbox_inches="tight")
    figure.savefig(args.figure_png, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight_dir", type=Path, required=True)
    parser.add_argument("--live_k4", type=Path, required=True)
    parser.add_argument("--live_k4_partial", type=Path, required=True)
    parser.add_argument("--autopsy_dir", type=Path, required=True)
    parser.add_argument("--autopsy_k4", type=Path, required=True)
    parser.add_argument("--amplitude", type=Path, required=True)
    parser.add_argument("--prelude2", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--figure_svg", type=Path, required=True)
    parser.add_argument("--figure_png", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
