"""Build the Stage 2B-S direct-cascade and desk-math closeout artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tarfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


EXPECTED_ARCHIVE_SHA256 = "6dd4c644a129a4e84cbe59edc772d4a5b8df176cc4a9ccf1f02c7459a965308e"
EXPECTED_DESK_SHA256 = "5d5f21f5a70a9b09f860b2fdd473ef75e760fdea5e50410dfa152b0cecd110aa"
NATIVE_COUNTS = {0: [162, 10, 2, 2], 1: [162, 9, 5, 1]}
RECOVERY_THRESHOLD = 182


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_member(archive: tarfile.TarFile, name: str) -> Any:
    handle = archive.extractfile(name)
    if handle is None:
        raise FileNotFoundError(name)
    return json.load(handle)


def read_jsonl_member(archive: tarfile.TarFile, name: str) -> list[dict[str, Any]]:
    handle = archive.extractfile(name)
    if handle is None:
        raise FileNotFoundError(name)
    return [json.loads(line) for line in handle if line.strip()]


def direct_rows_name(seed: int, k: int) -> str:
    return (
        f"stage2bs-durable/private/seed_{seed}/cascade_direct/initialization/generation/"
        f"deferred_terminal_write_no_reentry__k{k}__gamma_0p05.jsonl"
    )


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def summarize_direct(archive_path: Path) -> dict[str, Any]:
    observed_sha = sha256_file(archive_path)
    if observed_sha != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            f"Direct archive SHA mismatch: expected {EXPECTED_ARCHIVE_SHA256}, got {observed_sha}"
        )
    with tarfile.open(archive_path, "r:gz") as archive:
        wave = read_json_member(
            archive, "stage2bs-durable/receipts/cascade_direct_wave.json"
        )
        if wave["status"] != "FALLBACK_BRANCH_AUTHORIZED_AWAITING_RELAY":
            raise RuntimeError(f"Unexpected direct-wave status: {wave['status']}")
        if any(
            (
                wave["optimizer_constructed"],
                wave["optimizer_steps"],
                wave["confirm_scored"],
                wave["eval_e_scored"],
                wave["branch_execution_started"],
            )
        ):
            raise RuntimeError("Direct wave violated its score-only or sealed contract")

        rows_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
        cells = []
        for seed in (0, 1):
            seed_summary = read_json_member(
                archive, f"stage2bs-durable/receipts/seed_{seed}/summary.json"
            )
            summary_by_k = {int(cell["k"]): cell for cell in seed_summary["cells"]}
            if set(summary_by_k) != {1, 2, 3, 4}:
                raise RuntimeError(f"Seed {seed} lacks exact K1-K4 coverage")
            for k in (1, 2, 3, 4):
                rows = read_jsonl_member(archive, direct_rows_name(seed, k))
                if len(rows) != 461:
                    raise RuntimeError(f"Seed {seed} K{k} has {len(rows)} rows, expected 461")
                rows_by_cell[seed, k] = rows
                observed_correct = sum(bool(row["augmented_correct"]) for row in rows)
                if observed_correct != int(summary_by_k[k]["correct"]):
                    raise RuntimeError(f"Seed {seed} K{k} summary/row mismatch")

            k1 = rows_by_cell[seed, 1]
            for k in (1, 2, 3, 4):
                rows = rows_by_cell[seed, k]
                cells.append(
                    {
                        "seed": seed,
                        "k": k,
                        "correct": sum(bool(row["augmented_correct"]) for row in rows),
                        "accuracy": sum(bool(row["augmented_correct"]) for row in rows) / 461,
                        "by_battery": summary_by_k[k]["by_battery"],
                        "prediction_changes_vs_k1": sum(
                            left["prediction"] != right["prediction"]
                            for left, right in zip(k1, rows, strict=True)
                        ),
                        "correctness_changes_vs_k1": sum(
                            bool(left["augmented_correct"])
                            != bool(right["augmented_correct"])
                            for left, right in zip(k1, rows, strict=True)
                        ),
                        "mean_minimum_answer_margin": mean(rows, "answer_token_margin_minimum"),
                        "mean_position_gate": mean(rows, "position_gate_mean"),
                        "mean_position_gate_max": mean(rows, "position_gate_max"),
                        "mean_writeback_ratio": mean(rows, "realized_writeback_ratio_mean"),
                        "mean_writeback_ratio_max": mean(rows, "realized_writeback_ratio_max"),
                    }
                )

        return {
            "archive": {
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "sha256": observed_sha,
            },
            "wave": wave,
            "native_counts": {str(seed): values for seed, values in NATIVE_COUNTS.items()},
            "recovery_threshold_correct_rows": RECOVERY_THRESHOLD,
            "cells": cells,
            "all_cells_correct_rows": sorted({cell["correct"] for cell in cells}),
            "maximum_prediction_changes_vs_k1": max(
                cell["prediction_changes_vs_k1"] for cell in cells
            ),
            "maximum_correctness_changes_vs_k1": max(
                cell["correctness_changes_vs_k1"] for cell in cells
            ),
        }


def summarize_desk(desk_path: Path) -> dict[str, Any]:
    observed_sha = sha256_file(desk_path)
    if observed_sha != EXPECTED_DESK_SHA256:
        raise RuntimeError(
            f"Desk receipt SHA mismatch: expected {EXPECTED_DESK_SHA256}, got {observed_sha}"
        )
    payload = json.loads(desk_path.read_text(encoding="utf-8"))
    if any(
        (
            payload["optimizer_constructed"],
            payload["optimizer_steps"],
            payload["confirm_scored"],
            payload["eval_e_scored"],
        )
    ):
        raise RuntimeError("Desk receipt violated its score-only or sealed contract")

    d_m1 = []
    for seed_row in payload["d_m1_spectral_rmt"]:
        for matrix, matrix_row in seed_row["matrices"].items():
            alignments = matrix_row["top_hidden_direction_absolute_cosines"]
            d_m1.append(
                {
                    "seed": seed_row["seed"],
                    "matrix": matrix,
                    "outlier_spikes_under_iid_mp_fit": matrix_row["mp_bulk_fit"][
                        "outlier_spikes"
                    ],
                    "maximum_absolute_alignment": max(alignments.values()),
                    "alignments": alignments,
                }
            )

    d_m2 = []
    for seed_row in payload["d_m2_margin_recursion"]:
        for endpoint in ("initialization", "stop"):
            row = seed_row[endpoint]
            preferred = row[row["cv_winner"]]
            d_m2.append(
                {
                    "seed": seed_row["seed"],
                    "endpoint": endpoint,
                    "mean_margin_by_loop": row["mean_margin_by_loop"],
                    "cv_winner": row["cv_winner"],
                    "median_attenuation_r": preferred["r"]["median"],
                    "mean_absolute_signal_attenuation_contribution": preferred[
                        "mean_absolute_signal_attenuation_contribution"
                    ],
                    "mean_absolute_bias_accumulation_contribution": preferred[
                        "mean_absolute_bias_accumulation_contribution"
                    ],
                    "observed_positive_to_nonpositive_flips": preferred[
                        "observed_positive_to_nonpositive_flips"
                    ],
                }
            )

    return {
        "receipt": {
            "path": str(desk_path),
            "bytes": desk_path.stat().st_size,
            "sha256": observed_sha,
        },
        "d_m1": d_m1,
        "d_m2": d_m2,
        "d_m3": payload["d_m3_jvp"],
        "d_m4": payload["d_m4_bbp_feasibility"],
    }


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def build_direct_figure(summary: dict[str, Any], output_base: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    colors = {0: "#176B87", 1: "#C44E52"}
    ks = np.arange(1, 5)
    for seed in (0, 1):
        direct = [
            next(
                cell["correct"]
                for cell in summary["direct"]["cells"]
                if cell["seed"] == seed and cell["k"] == k
            )
            for k in ks
        ]
        axes[0].plot(
            ks,
            NATIVE_COUNTS[seed],
            linestyle="--",
            marker="o",
            linewidth=1.6,
            color=colors[seed],
            alpha=0.6,
            label=f"Seed {seed}, native interleaved",
        )
        if seed == 0:
            axes[0].plot(
                ks,
                direct,
                linestyle="-",
                marker="s",
                linewidth=2.5,
                color="#2F3E46",
                label="Terminal write, both seeds",
            )
    axes[0].axhline(
        RECOVERY_THRESHOLD,
        color="#222222",
        linewidth=1.2,
        linestyle=":",
        label="Recovery threshold (182)",
    )
    axes[0].set(
        title="A. Terminal-write accuracy is flat",
        xlabel="Sidecar updates (K)",
        ylabel="Correct rows (of 461)",
        xticks=ks,
        ylim=(-8, 200),
    )
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axes[0].legend(
        frameon=False,
        fontsize=8,
        loc="center right",
        bbox_to_anchor=(0.99, 0.51),
    )

    for seed in (0, 1):
        cells = sorted(
            (cell for cell in summary["direct"]["cells"] if cell["seed"] == seed),
            key=lambda row: row["k"],
        )
        axes[1].plot(
            ks,
            [100 * cell["mean_writeback_ratio_max"] for cell in cells],
            marker="o",
            linewidth=2.1,
            color=colors[seed],
            label=f"Seed {seed}",
        )
    axes[1].set(
        title="B. The terminal write shrinks with depth",
        xlabel="Sidecar updates (K)",
        ylabel="Mean per-row maximum write / RMS (%)",
        xticks=ks,
        ylim=(0, 4.2),
    )
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axes[1].legend(frameon=False)
    fig.suptitle("Stage 2B-S direct discriminator", fontsize=14, fontweight="bold")
    for suffix in ("png", "svg"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def build_desk_figure(summary: dict[str, Any], output_base: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    colors = {
        (0, "initialization"): "#176B87",
        (0, "stop"): "#64A7B7",
        (1, "initialization"): "#C44E52",
        (1, "stop"): "#E59A9D",
    }
    loops = np.arange(1, 5)
    for row in summary["desk"]["d_m2"]:
        axes[0].plot(
            loops,
            row["mean_margin_by_loop"],
            marker="o",
            linewidth=2,
            color=colors[row["seed"], row["endpoint"]],
            label=f"Seed {row['seed']}, {row['endpoint']}",
        )
    axes[0].axhline(0, color="#222222", linewidth=1, linestyle=":")
    axes[0].set(
        title="A. Native-loop answer margin attenuates",
        xlabel="Recurrent pass (K)",
        ylabel="Mean answer-token margin",
        xticks=loops,
    )
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=8)

    labels = []
    values = []
    bar_colors = []
    palette = {"delta_W_H": "#4C78A8", "delta_W_P": "#F58518", "delta_bridge_B_L": "#54A24B"}
    for row in summary["desk"]["d_m1"]:
        short = {"delta_W_H": "W_H", "delta_W_P": "W_P", "delta_bridge_B_L": "B_L"}[
            row["matrix"]
        ]
        labels.append(f"S{row['seed']} {short}")
        values.append(row["maximum_absolute_alignment"])
        bar_colors.append(palette[row["matrix"]])
    axes[1].bar(labels, values, color=bar_colors, width=0.72)
    axes[1].set(
        title="B. Learned updates weakly align with audited directions",
        ylabel="Maximum absolute cosine",
        ylim=(0, 0.23),
    )
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("Stage 2B-S desk-math diagnostics", fontsize=14, fontweight="bold")
    for suffix in ("png", "svg"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--desk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure_dir", type=Path, required=True)
    args = parser.parse_args()

    summary = {
        "kind": "paper2_stage2bs_direct_cascade_and_desk_math_analysis_v1",
        "status": "complete_awaiting_strategy_relay",
        "direct": summarize_direct(args.archive),
        "desk": summarize_desk(args.desk),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_direct_figure(summary, args.figure_dir / "paper2_stage2bs_direct_cascade_20260823")
    build_desk_figure(summary, args.figure_dir / "paper2_stage2bs_desk_math_20260823")
    print(f"summary={args.output}")
    print(f"summary_sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
