"""Analyze Stage 2A CV-1 crossed values and the D5 relevance probe."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import spearmanr

from eval.eval_paper2_stage2a import exact_sign_test, read_json, read_jsonl, write_json
from training.paper2_phase3_p31_completion import sha256_file


def compare_conditions(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("paired CV-1 conditions require identical row IDs")
    left_only = sum(
        bool(left[item]["augmented_correct"]) and not bool(right[item]["augmented_correct"])
        for item in left
    )
    right_only = sum(
        bool(right[item]["augmented_correct"]) and not bool(left[item]["augmented_correct"])
        for item in left
    )
    return {
        "rows": len(left),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "net_rows_left_minus_right": left_only - right_only,
        "paired_sign_test_p_two_sided": exact_sign_test(left_only, right_only),
    }


def permutation_spearman(
    values: Iterable[float],
    outcomes: Iterable[float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=np.float64)
    y = np.asarray(list(outcomes), dtype=np.float64)
    if x.size != y.size or x.size < 3:
        return {"rows": int(x.size), "rho": None, "permutation_p_two_sided": None}
    observed = float(spearmanr(x, y).statistic)
    if not np.isfinite(observed):
        return {"rows": int(x.size), "rho": None, "permutation_p_two_sided": None}
    generator = np.random.default_rng(int(seed))
    extreme = 0
    for _ in range(int(draws)):
        shuffled = generator.permutation(y)
        rho = float(spearmanr(x, shuffled).statistic)
        if np.isfinite(rho) and abs(rho) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "rows": int(x.size),
        "rho": observed,
        "permutation_draws": int(draws),
        "permutation_p_two_sided": (extreme + 1) / (int(draws) + 1),
    }


def holm_adjust(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    valid = sorted(
        ((name, float(value)) for name, value in p_values.items() if value is not None),
        key=lambda pair: pair[1],
    )
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    total = len(valid)
    for index, (name, value) in enumerate(valid):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def group_metric_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return {
        "rows": len(rows),
        "observed": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
    }


def _lookup(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    lookup = {str(row["item_id"]): row for row in rows}
    if len(lookup) != len(rows):
        raise RuntimeError(f"duplicate row IDs in {path}")
    return lookup


def build_d5(
    *,
    correct: Mapping[str, Mapping[str, Any]],
    shuffled: Mapping[str, Mapping[str, Any]],
    random_rows: Mapping[str, Mapping[str, Any]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if set(correct) != set(shuffled) or set(correct) != set(random_rows):
        raise RuntimeError("D5 value conditions do not cover identical rows")
    rows = []
    for item_id in sorted(correct):
        source = correct[item_id]
        content = int(bool(source["augmented_correct"]))
        shuffled_correct = int(bool(shuffled[item_id]["augmented_correct"]))
        random_correct = int(bool(random_rows[item_id]["augmented_correct"]))
        rows.append(
            {
                "item_id": item_id,
                "battery": source["battery"],
                "content_advantage": 2 * content - shuffled_correct - random_correct,
                "strict_content_win": bool(content and not shuffled_correct and not random_correct),
                "strict_content_loss": bool(not content and shuffled_correct and random_correct),
                "memory_retrieval_score_mean": source.get("memory_retrieval_score_mean"),
                "memory_retrieval_entropy_mean": source.get("memory_retrieval_entropy_mean"),
                "memory_compatibility_gate_mean": source.get("memory_compatibility_gate_mean"),
            }
        )

    metrics = (
        "memory_retrieval_score_mean",
        "memory_compatibility_gate_mean",
        "memory_retrieval_entropy_mean",
    )
    populations = {
        "mmlu_gsm8k_pooled": [row for row in rows if row["battery"] in {"mmlu", "gsm8k"}],
        "mmlu": [row for row in rows if row["battery"] == "mmlu"],
        "gsm8k": [row for row in rows if row["battery"] == "gsm8k"],
    }
    results: dict[str, Any] = {}
    for population_index, (name, selected) in enumerate(populations.items()):
        correlations = {}
        for metric_index, metric in enumerate(metrics):
            observed = [row for row in selected if row.get(metric) is not None]
            correlations[metric] = permutation_spearman(
                [float(row[metric]) for row in observed],
                [float(row["content_advantage"]) for row in observed],
                draws=draws,
                seed=seed + 100 * population_index + metric_index,
            )
        wins = [row for row in selected if row["strict_content_win"]]
        losses = [row for row in selected if row["strict_content_loss"]]
        results[name] = {
            "rows": len(selected),
            "content_advantage_counts": {
                str(value): sum(row["content_advantage"] == value for row in selected)
                for value in (-2, -1, 0, 1, 2)
            },
            "strict_content_wins": len(wins),
            "strict_content_losses": len(losses),
            "correlations": correlations,
            "strict_win_metric_means": {
                metric: group_metric_summary(wins, metric) for metric in metrics
            },
            "strict_loss_metric_means": {
                metric: group_metric_summary(losses, metric) for metric in metrics
            },
        }
    primary_p = {
        battery: results[battery]["correlations"]["memory_retrieval_score_mean"][
            "permutation_p_two_sided"
        ]
        for battery in ("mmlu", "gsm8k")
    }
    adjusted = holm_adjust(primary_p)
    for battery, value in adjusted.items():
        results[battery]["correlations"]["memory_retrieval_score_mean"][
            "holm_p_two_batteries"
        ] = value
    return {"row_definitions": rows, "populations": results}


def make_figure(summary: Mapping[str, Any], d5: Mapping[str, Any], png: Path, svg: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"correct": "#1f6f5f", "shuffled": "#c17c22", "random": "#8e4f66"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), constrained_layout=True)
    for host, linestyle in (("t3a", "-"), ("t3b", "--")):
        for condition in ("correct", "shuffled", "random"):
            cells = [
                summary["logical_cells"][f"{host}__{condition}__dose_{label}"]
                for label in ("0p0", "0p5", "1p0")
            ]
            axes[0].plot(
                (0.0, 0.5, 1.0),
                [cell["pooled"]["initialization"]["delta_rows"] for cell in cells],
                marker="o",
                linestyle=linestyle,
                color=colors[condition],
                label=f"{host.upper()} {condition}",
            )
    axes[0].axhline(0, color="#222222", linewidth=1)
    axes[0].set_title("Doorway dose surface")
    axes[0].set_xlabel("Gate-dose multiplier")
    axes[0].set_ylabel("Rows vs initialization (of 1,024)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.2)

    row_defs = d5["row_definitions"]
    groups = []
    labels = []
    for battery in ("mmlu", "gsm8k"):
        for advantage in (-2, -1, 0, 1, 2):
            values = [
                row["memory_retrieval_score_mean"]
                for row in row_defs
                if row["battery"] == battery
                and row["content_advantage"] == advantage
                and row["memory_retrieval_score_mean"] is not None
            ]
            if values:
                groups.append(values)
                labels.append(f"{battery}\n{advantage:+d}")
    axes[1].boxplot(groups, labels=labels, showfliers=False)
    axes[1].set_title("D5 relevance by content advantage")
    axes[1].set_ylabel("Mean top-1 retrieval score")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Stage 2A CV-1 crossed-value audit", fontsize=15, fontweight="bold")
    png.parent.mkdir(parents=True, exist_ok=True)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--figure_png", type=Path, required=True)
    parser.add_argument("--figure_svg", type=Path, required=True)
    args = parser.parse_args()

    summary = read_json(args.input_dir / "summary.json")
    if summary.get("status") != "complete_dev_score_only":
        raise RuntimeError("CV-1 summary is not complete")
    cells = args.input_dir / "cells"
    row_maps = {
        key: _lookup(cells / key / "dev_rows.jsonl")
        for key in summary["physical_cells"]
    }
    contrasts = {}
    for host in ("t3a", "t3b"):
        for dose_label in ("0p5", "1p0"):
            correct = row_maps[f"{host}__correct__dose_{dose_label}"]
            for control in ("shuffled", "random"):
                contrasts[f"{host}__correct_vs_{control}__dose_{dose_label}"] = compare_conditions(
                    correct, row_maps[f"{host}__{control}__dose_{dose_label}"]
                )

    d5 = build_d5(
        correct=row_maps["t3a__correct__dose_1p0"],
        shuffled=row_maps["t3a__shuffled__dose_1p0"],
        random_rows=row_maps["t3a__random__dose_1p0"],
        draws=10_000,
        seed=20_260_820,
    )
    analysis = {
        "kind": "paper2_stage2a_cv1_d5_analysis_v1",
        "status": "complete_descriptive_dev_reuse",
        "source_summary": str(args.input_dir / "summary.json"),
        "source_summary_sha256": sha256_file(args.input_dir / "summary.json"),
        "crossed_value_contrasts": contrasts,
        "d5": d5,
        "registered_t3_verdict_changed": False,
        "training_authorized": False,
        "optimizer_constructed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "analysis_summary.json", analysis)
    make_figure(summary, d5, args.figure_png, args.figure_svg)
    print(json.dumps(analysis, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
