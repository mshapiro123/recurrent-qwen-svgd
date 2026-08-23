"""Analyze and plot the banked Stage 2B-S final cascade cell."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


plt.rcParams["svg.hashsalt"] = "paper2-stage2bs-final-20260823"


NATIVE = {0: [162, 10, 2, 2], 1: [162, 9, 5, 1]}
DIRECT = {0: [159, 159, 159, 159], 1: [159, 159, 159, 159]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def bootstrap_mean_ci(values: Sequence[float], *, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(array), size=(10_000, len(array)))
    means = array[indexes].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def generation_path(root: Path, seed: int, k: int) -> Path:
    return (
        root
        / f"private/seed_{seed}/cascade_final/initialization/generation"
        / f"per_loop_write_no_reentry__k{k}__gamma_0p05.jsonl"
    )


def margin_path(root: Path, seed: int, k: int) -> Path:
    return (
        root
        / f"private/seed_{seed}/cascade_final_margins/initialization/margins"
        / f"deferred_terminal_write_no_reentry__k{k}__gamma_0p05.jsonl"
    )


def generation_analysis(root: Path, seed: int) -> dict[str, Any]:
    rows_by_k = {k: read_jsonl(generation_path(root, seed, k)) for k in range(1, 5)}
    by_id = {
        k: {str(row["item_id"]): row for row in rows} for k, rows in rows_by_k.items()
    }
    if any(len(rows) != 461 for rows in rows_by_k.values()):
        raise RuntimeError(f"seed {seed} final generation coverage changed")
    ids = set(by_id[1])
    if any(set(items) != ids for items in by_id.values()):
        raise RuntimeError(f"seed {seed} final generation identities changed")
    cells = []
    for k, rows in rows_by_k.items():
        correct = sum(bool(row["augmented_correct"]) for row in rows)
        fixes_vs_k1 = sum(
            bool(row["augmented_correct"])
            and not bool(by_id[1][str(row["item_id"])]["augmented_correct"])
            for row in rows
        )
        regressions_vs_k1 = sum(
            not bool(row["augmented_correct"])
            and bool(by_id[1][str(row["item_id"])]["augmented_correct"])
            for row in rows
        )
        prediction_changes_vs_k1 = sum(
            row["generated_token_ids"]
            != by_id[1][str(row["item_id"])]["generated_token_ids"]
            for row in rows
        )
        by_battery: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_battery[str(row["battery"])].append(row)
        cells.append(
            {
                "k": k,
                "correct": correct,
                "accuracy": correct / len(rows),
                "by_battery": {
                    name: {
                        "rows": len(items),
                        "correct": sum(bool(row["augmented_correct"]) for row in items),
                    }
                    for name, items in sorted(by_battery.items())
                },
                **{
                    key: float(np.mean([float(row[key]) for row in rows]))
                    for key in (
                        "accumulated_write_magnitude_mean",
                        "deployed_write_magnitude_mean",
                        "accumulated_write_ratio_mean",
                        "deployed_write_ratio_mean",
                    )
                },
                "deployed_over_accumulated": float(
                    np.mean([float(row["deployed_write_magnitude_mean"]) for row in rows])
                    / max(
                        np.mean(
                            [float(row["accumulated_write_magnitude_mean"]) for row in rows]
                        ),
                        1e-12,
                    )
                ),
                "prediction_changes_vs_k1": prediction_changes_vs_k1,
                "prediction_change_fraction_vs_k1": prediction_changes_vs_k1 / len(rows),
                "correctness_changes_vs_k1": fixes_vs_k1 + regressions_vs_k1,
                "fixes_vs_k1": fixes_vs_k1,
                "regressions_vs_k1": regressions_vs_k1,
                "source_sha256": sha256_file(generation_path(root, seed, k)),
            }
        )
    return {"seed": seed, "cells": cells}


def margin_analysis(root: Path, seed: int) -> dict[str, Any] | None:
    paths = {k: margin_path(root, seed, k) for k in (1, 4)}
    if not all(path.is_file() for path in paths.values()):
        return None
    rows = {k: read_jsonl(path) for k, path in paths.items()}
    if any(len(items) != 2048 for items in rows.values()):
        raise RuntimeError(f"seed {seed} final margin coverage changed")
    by_id = {k: {str(row["item_id"]): row for row in items} for k, items in rows.items()}
    if set(by_id[1]) != set(by_id[4]):
        raise RuntimeError(f"seed {seed} final margin identities changed")
    deltas = {
        item_id: float(by_id[4][item_id]["mean_teacher_token_margin"])
        - float(by_id[1][item_id]["mean_teacher_token_margin"])
        for item_id in by_id[1]
    }
    batteries: dict[str, list[float]] = defaultdict(list)
    for item_id, delta in deltas.items():
        batteries[str(by_id[1][item_id]["battery"])].append(delta)
    delta_array = np.asarray(list(deltas.values()), dtype=np.float64)
    return {
        "seed": seed,
        "rows": len(deltas),
        "mean_k4_minus_k1": float(np.mean(list(deltas.values()))),
        "median_k4_minus_k1": float(np.median(list(deltas.values()))),
        "positive_delta_fraction": float(np.mean(delta_array > 0)),
        "negative_delta_fraction": float(np.mean(delta_array < 0)),
        "zero_delta_fraction": float(np.mean(delta_array == 0)),
        "bootstrap_95_ci": bootstrap_mean_ci(list(deltas.values()), seed=20260823 + seed),
        "by_battery": {
            name: {
                "rows": len(values),
                "mean_k4_minus_k1": float(np.mean(values)),
                "bootstrap_95_ci": bootstrap_mean_ci(
                    values, seed=20260823 + seed + index * 100
                ),
            }
            for index, (name, values) in enumerate(sorted(batteries.items()))
        },
        "source_sha256": {str(k): sha256_file(path) for k, path in paths.items()},
    }


def build_figure(analysis: Mapping[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ks = np.arange(1, 5)
    colors = {0: "#007C91", 1: "#C84C3A"}
    for seed in (0, 1):
        final = analysis["generation"][seed]["cells"]
        axes[0].plot(ks, NATIVE[seed], ":", color="#71777D", alpha=0.65)
        axes[0].plot(ks, DIRECT[seed], "--", color="#71777D", alpha=0.8)
        axes[0].plot(
            ks,
            [cell["correct"] for cell in final],
            "o-",
            color=colors[seed],
            label=f"seed {seed} per-loop",
        )
        axes[1].plot(
            ks,
            [cell["accumulated_write_magnitude_mean"] for cell in final],
            "o-",
            color=colors[seed],
            label=f"seed {seed} accumulated",
        )
        axes[1].plot(
            ks,
            [cell["deployed_write_magnitude_mean"] for cell in final],
            "s--",
            color=colors[seed],
            alpha=0.8,
            label=f"seed {seed} deployed",
        )
    axes[0].axhline(182, color="#1B1F23", linewidth=1, alpha=0.55, label="effect floor")
    axes[0].set(title="Accuracy across schedules", xlabel="K", ylabel="Correct of 461")
    axes[0].set_xticks(ks)
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=colors[0], marker="o", label="per-loop seed 0"),
            Line2D([0], [0], color=colors[1], marker="o", label="per-loop seed 1"),
            Line2D([0], [0], color="#71777D", linestyle=":", label="native matched graph"),
            Line2D([0], [0], color="#71777D", linestyle="--", label="deferred terminal write"),
            Line2D([0], [0], color="#1B1F23", linewidth=1, label="+20 effect floor"),
        ],
        frameon=False,
        fontsize=8,
    )
    axes[1].set(title="Write path vs deployed displacement", xlabel="K", ylabel="Raw hidden RMS")
    axes[1].set_xticks(ks)
    axes[1].legend(frameon=False, fontsize=8)

    margins = [item for item in analysis["margins"] if item is not None]
    if margins:
        batteries = ["pooled", "gsm8k", "arc_challenge", "mbpp"]
        width = 0.34
        for seed, item in enumerate(margins):
            values = [
                item["mean_k4_minus_k1"]
                if name == "pooled"
                else item["by_battery"][name]["mean_k4_minus_k1"]
                for name in batteries
            ]
            intervals = [
                item["bootstrap_95_ci"]
                if name == "pooled"
                else item["by_battery"][name]["bootstrap_95_ci"]
                for name in batteries
            ]
            errors = np.asarray(
                [
                    [value - interval[0] for value, interval in zip(values, intervals)],
                    [interval[1] - value for value, interval in zip(values, intervals)],
                ]
            )
            axes[2].bar(
                np.arange(len(batteries)) + (seed - 0.5) * width,
                values,
                width,
                color=colors[seed],
                yerr=errors,
                capsize=3,
                label=f"seed {seed}",
            )
        axes[2].axhline(0, color="#1B1F23", linewidth=1)
        axes[2].set_xticks(
            np.arange(len(batteries)), ["Pooled", "GSM8K", "ARC-C", "MBPP"]
        )
        axes[2].set(title="Deferred K4 minus K1 margin", ylabel="Mean teacher-token margin")
        axes[2].legend(frameon=False, fontsize=8)
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "Margin panel not opened\nunder registered branch", ha="center", va="center")
    for index, axis in enumerate(axes):
        axis.text(-0.13, 1.04, chr(ord("A") + index), transform=axis.transAxes, fontweight="bold")
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    fig.suptitle("Stage 2B-S final write-schedule cell", fontsize=15, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    svg_path = output.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    wave_path = args.root / "receipts/cascade_final_wave.json"
    wave = read_json(wave_path)
    generation = [generation_analysis(args.root, seed) for seed in (0, 1)]
    margins = [margin_analysis(args.root, seed) for seed in (0, 1)]
    analysis = {
        "kind": "paper2_stage2bs_final_cell_analysis_v1",
        "status": wave["status"],
        "registered_decision": wave["registered_decision"],
        "generation": generation,
        "margins": margins,
        "integrity": {
            "wave_sha256": sha256_file(wave_path),
            "optimizer_constructed": wave["optimizer_constructed"],
            "optimizer_steps": wave["optimizer_steps"],
            "confirm_scored": wave["confirm_scored"],
            "eval_e_scored": wave["eval_e_scored"],
            "partial_interleave_executed": wave["partial_interleave_executed"],
        },
    }
    atomic_json(args.output, analysis)
    build_figure(analysis, args.figure)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
