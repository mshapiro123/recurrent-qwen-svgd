"""Build the pooled P3.3 result receipt and strategy-review figure."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np


BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20260811


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def ratio_bootstrap(
    rows: list[dict[str, Any]],
    *,
    numerator: str,
    denominator: str,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        values = grouped[str(row["document_id"])]
        values[0] += int(bool(row[numerator]))
        values[1] += int(bool(row[denominator]))
    contributions = np.asarray(list(grouped.values()), dtype=np.int64)
    point = float(contributions[:, 0].sum() / contributions[:, 1].sum())
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        selected = contributions[rng.integers(0, len(contributions), len(contributions))]
        denominator_sum = int(selected[:, 1].sum())
        if denominator_sum:
            samples.append(float(selected[:, 0].sum() / denominator_sum))
    return {
        "point": point,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "numerator": int(contributions[:, 0].sum()),
        "denominator": int(contributions[:, 1].sum()),
        "documents": len(contributions),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def clustered_metric_bootstrap(
    rows: list[dict[str, Any]],
    *,
    contribution: Callable[[dict[str, Any]], tuple[int, int]],
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        left, right = contribution(row)
        values = grouped[str(row["document_id"])]
        values[0] += left
        values[1] += right
    contributions = np.asarray(list(grouped.values()), dtype=np.int64)
    denominator = int(contributions[:, 1].sum())
    point = float(contributions[:, 0].sum() / denominator) if denominator else 0.0
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        selected = contributions[rng.integers(0, len(contributions), len(contributions))]
        selected_denominator = int(selected[:, 1].sum())
        if selected_denominator:
            samples.append(float(selected[:, 0].sum() / selected_denominator))
    return {
        "point": point,
        "ci95_low": float(np.quantile(samples, 0.025)) if samples else None,
        "ci95_high": float(np.quantile(samples, 0.975)) if samples else None,
        "numerator": int(contributions[:, 0].sum()),
        "denominator": denominator,
        "documents": len(contributions),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def seed_paths(results_root: Path, seed: int) -> tuple[Path, Path]:
    base = results_root / f"seed{seed}"
    summary = (
        base
        / "recurrent-qwen-svgd"
        / "outputs"
        / "stage5"
        / "stage5_paper2_phase3_p33_20260811"
        / f"seed_{seed}"
        / "summary.json"
    )
    rows = base / "p33-private" / f"seed_{seed}" / "audit_rows_step_1000.jsonl"
    return summary, rows


def build(results_root: Path) -> dict[str, Any]:
    summaries: dict[int, dict[str, Any]] = {}
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in (0, 1):
        summary_path, rows_path = seed_paths(results_root, seed)
        summaries[seed] = read_json(summary_path)
        rows_by_seed[seed] = read_jsonl(rows_path)
        if summaries[seed]["status"] != "complete" or summaries[seed]["step"] != 1000:
            raise RuntimeError(f"seed {seed} is not complete")
        if summaries[seed]["warnings"] or summaries[seed]["stop_reason"] is not None:
            raise RuntimeError(f"seed {seed} completed with a warning or stop")

    pooled_rows = rows_by_seed[0] + rows_by_seed[1]
    positives = [row for row in pooled_rows if row["population"] == "positive"]
    negatives = [row for row in pooled_rows if row["population"] == "negative"]
    gate_rows = positives + negatives

    pi_dir = ratio_bootstrap(
        positives,
        numerator="forced_trained_flip",
        denominator="forced_oracle_flip",
        seed=BOOTSTRAP_SEED,
    )
    pi_dep = ratio_bootstrap(
        positives,
        numerator="deployed_trained_flip",
        denominator="deployed_oracle_flip",
        seed=BOOTSTRAP_SEED + 1,
    )
    reader_matched = [row for row in positives if bool(row["base_reader_matches_cached_student"])]
    reader_mismatched = [row for row in positives if not bool(row["base_reader_matches_cached_student"])]
    reader_sensitivity = {
        "match_rate": len(reader_matched) / len(positives),
        "matched_rows": len(reader_matched),
        "mismatched_rows": len(positives) - len(reader_matched),
        "pi_dir": ratio_bootstrap(
            reader_matched,
            numerator="forced_trained_flip",
            denominator="forced_oracle_flip",
            seed=BOOTSTRAP_SEED + 10,
        ),
        "pi_dep": ratio_bootstrap(
            reader_matched,
            numerator="deployed_trained_flip",
            denominator="deployed_oracle_flip",
            seed=BOOTSTRAP_SEED + 11,
        ),
        "mismatched_pi_dir": ratio_bootstrap(
            reader_mismatched,
            numerator="forced_trained_flip",
            denominator="forced_oracle_flip",
            seed=BOOTSTRAP_SEED + 12,
        ),
        "mismatched_pi_dep": ratio_bootstrap(
            reader_mismatched,
            numerator="deployed_trained_flip",
            denominator="deployed_oracle_flip",
            seed=BOOTSTRAP_SEED + 13,
        ),
        "role": "sensitivity only; the locked primary includes all rows",
    }
    gate_recall = clustered_metric_bootstrap(
        positives,
        contribution=lambda row: (int(float(row["gate_unclamped"]) >= 0.5), 1),
        seed=BOOTSTRAP_SEED + 2,
    )
    gate_precision = clustered_metric_bootstrap(
        gate_rows,
        contribution=lambda row: (
            int(row["population"] == "positive" and float(row["gate_unclamped"]) >= 0.5),
            int(float(row["gate_unclamped"]) >= 0.5),
        ),
        seed=BOOTSTRAP_SEED + 3,
    )
    gate_fpr = clustered_metric_bootstrap(
        negatives,
        contribution=lambda row: (int(float(row["gate_unclamped"]) >= 0.5), 1),
        seed=BOOTSTRAP_SEED + 4,
    )
    collateral = clustered_metric_bootstrap(
        negatives,
        contribution=lambda row: (int(bool(row["collateral_change"])), 1),
        seed=BOOTSTRAP_SEED + 5,
    )

    deciles = {}
    for decile in range(10):
        local = [row for row in positives if int(row["teachability_decile"]) == decile]
        deciles[str(decile)] = {
            "rows": len(local),
            "pi_dir": sum(bool(row["forced_trained_flip"]) for row in local)
            / sum(bool(row["forced_oracle_flip"]) for row in local),
            "pi_dep": sum(bool(row["deployed_trained_flip"]) for row in local)
            / sum(bool(row["deployed_oracle_flip"]) for row in local),
        }

    per_seed = {}
    for seed, summary in summaries.items():
        final = summary["history"][-1]
        audit = final["audit"]
        per_seed[str(seed)] = {
            "pi_dir": audit["pi_dir"],
            "pi_dep": audit["pi_dep"],
            "mean_direction_cosine": audit["mean_direction_cosine"],
            "gate": audit["gate"],
            "collateral_chi": audit["collateral_chi"],
            "retention": final["retention"],
            "directional_share": final["directional_share"],
            "train_losses": final["train_losses"],
            "observatory": final["observatory"],
            "a_state_intervention_battery": summary["a_state_intervention_battery"],
            "checkpoint": summary["checkpoint"],
            "source_checkpoint": summary["source_checkpoint"],
            "schedule_sha256": summary["schedule_sha256"],
            "zero_loop_identity": summary["zero_loop_identity"],
            "warnings": summary["warnings"],
        }

    mean_pi_dir = float(np.mean([per_seed[str(seed)]["pi_dir"]["point"] for seed in (0, 1)]))
    mean_pi_dep = float(np.mean([per_seed[str(seed)]["pi_dep"]["point"] for seed in (0, 1)]))
    mean_aim_share = float(
        np.mean([per_seed[str(seed)]["directional_share"]["shares"]["aim"] for seed in (0, 1)])
    )
    mean_gate_share = float(
        np.mean([per_seed[str(seed)]["directional_share"]["shares"]["gate"] for seed in (0, 1)])
    )
    result = {
        "kind": "paper2_phase3_p33_combined_result_v1",
        "date": "2026-08-11",
        "status": "complete",
        "seeds": [0, 1],
        "steps_per_seed": 1000,
        "looks_per_seed": 20,
        "primary_reading": {
            "metric": "pi_dir",
            "pooled": pi_dir,
            "seed_mean": mean_pi_dir,
            "registered_band": "single_iteration",
            "registered_rule": "0.05 <= pi_dir < 0.25",
        },
        "deployed_reading": {"metric": "pi_dep", "pooled": pi_dep, "seed_mean": mean_pi_dep},
        "bf16_reader_sensitivity": reader_sensitivity,
        "gate": {"recall": gate_recall, "precision": gate_precision, "false_positive_rate": gate_fpr},
        "collateral_chi": collateral,
        "retention": {
            "seed_0": summaries[0]["history"][-1]["retention"],
            "seed_1": summaries[1]["history"][-1]["retention"],
        },
        "directional_share_diagnosis": {
            "mean_aim_share": mean_aim_share,
            "mean_gate_share": mean_gate_share,
            "interpretation": "The combined primary contract passed, but gate gradients dominated aim gradients at the endpoint.",
        },
        "by_teachability_decile": deciles,
        "per_seed": per_seed,
        "task_level_capability_scoring": False,
        "confirm_scored": False,
        "eval_e_scored": False,
        "do_not_claim": [
            "No task-level capability or gap_closed result exists from P3.3.",
            "pi_dir and pi_dep are audit ratios, not deployment properties.",
            "The primary full-speed threshold of pi_dir >= 0.25 was not met.",
            "The boundary threshold of pi_dir < 0.05 was not met.",
        ],
    }
    return result


def draw_figure(results_root: Path, result: dict[str, Any], output_prefix: Path) -> None:
    plt.rcParams["svg.hashsalt"] = "paper2-phase3-p33-20260811"
    summaries = {seed: read_json(seed_paths(results_root, seed)[0]) for seed in (0, 1)}
    colors = {0: "#176B87", 1: "#C44E52"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    for seed, summary in summaries.items():
        steps = [row["step"] for row in summary["history"]]
        pi_dir = [row["audit"]["pi_dir"]["point"] for row in summary["history"]]
        pi_dep = [row["audit"]["pi_dep"]["point"] for row in summary["history"]]
        ax.plot(steps, pi_dir, color=colors[seed], linewidth=2.2, label=f"seed {seed} $\\pi_{{dir}}$")
        ax.plot(steps, pi_dep, color=colors[seed], linewidth=1.8, linestyle="--", label=f"seed {seed} $\\pi_{{dep}}$")
    ax.axhline(0.25, color="#4D4D4D", linestyle=":", linewidth=1.4, label="full-speed threshold")
    ax.axhline(0.05, color="#999999", linestyle=":", linewidth=1.2, label="boundary threshold")
    ax.set(title="A. Aim capture plateaus in the middle band", xlabel="training step", ylabel="capture ratio")
    ax.set_ylim(0, 0.34)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[0, 1]
    for seed, summary in summaries.items():
        steps = [row["step"] for row in summary["history"]]
        recall = [row["audit"]["gate"]["recall"] for row in summary["history"]]
        precision = [row["audit"]["gate"]["precision"] for row in summary["history"]]
        ax.plot(steps, recall, color=colors[seed], linewidth=2.2, label=f"seed {seed} recall")
        ax.plot(steps, precision, color=colors[seed], linewidth=1.8, linestyle="--", label=f"seed {seed} precision")
    ax.axhline(1.0, color="#2F855A", linewidth=1.1, alpha=0.7, label="retention (both seeds)")
    ax.set(title="B. The gate learns cleanly; retention is exact", xlabel="training step", ylabel="rate")
    ax.set_ylim(0, 1.04)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1, 0]
    deciles = np.arange(10)
    direction = [result["by_teachability_decile"][str(i)]["pi_dir"] for i in deciles]
    deployed = [result["by_teachability_decile"][str(i)]["pi_dep"] for i in deciles]
    width = 0.38
    ax.bar(deciles - width / 2, direction, width, color="#176B87", label="$\\pi_{dir}$")
    ax.bar(deciles + width / 2, deployed, width, color="#E39C37", label="$\\pi_{dep}$")
    ax.axhline(0.25, color="#4D4D4D", linestyle=":", linewidth=1.2)
    ax.set(title="C. Capture improves in the highest-teachability decile", xlabel="teachability decile", ylabel="pooled capture ratio")
    ax.set_xticks(deciles)
    ax.set_ylim(0, 0.56)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for seed, summary in summaries.items():
        shares = summary["directional_share_history"]
        steps = [row["step"] for row in shares]
        aim = [row["shares"]["aim"] for row in shares]
        gate = [row["shares"]["gate"] for row in shares]
        ax.plot(steps, aim, color=colors[seed], linewidth=2.2, label=f"seed {seed} aim")
        ax.plot(steps, gate, color=colors[seed], linewidth=1.8, linestyle="--", label=f"seed {seed} gate")
    ax.set(title="D. Gate gradients dominate the primary objective", xlabel="training step", ylabel="post-clip gradient share")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle("Phase 3 P3.3 aimed-writeback pilot: replicated partial control", fontsize=15, fontweight="bold")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=180, bbox_inches="tight", metadata={"Software": "matplotlib"})
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("p33_results"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/stage5/stage5_paper2_phase3_p33_20260811/combined_summary.json"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/figures/paper2_phase3_p33_result_20260811"),
    )
    args = parser.parse_args()
    result = build(args.results_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    draw_figure(args.results_root, result, args.figure_prefix)
    print(json.dumps({"output_json": str(args.output_json), "figure_prefix": str(args.figure_prefix), "primary": result["primary_reading"]}, indent=2))


if __name__ == "__main__":
    main()
