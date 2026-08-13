"""Build the reproducible P3.4 campaign analysis and diagnostic figures.

This script consumes only landed DEV receipts. It does not load a model, touch
CONFIRM/EVAL-E rows, or modify campaign checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPTS = ROOT / ".codex_p34_final_download"
DEFAULT_PANEL = ROOT / (
    "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/"
    "p34_task_panel.jsonl"
)
DEFAULT_REFERENCE = ROOT / ".codex_p31_reference/p31_merged_dev_verified_scores.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_analysis_20260813"
DEFAULT_FIGURES = ROOT / "docs/figures"

CONDITIONS = ("main_seed_0", "main_seed_1", "slot_seed_0")
DISPLAY = {
    "main_seed_0": "Main seed 0",
    "main_seed_1": "Main seed 1",
    "slot_seed_0": "Slot seed 0",
}
COLORS = {
    "main_seed_0": "#2878B5",
    "main_seed_1": "#D95F02",
    "slot_seed_0": "#708238",
}
BATTERY_ORDER = ("arc_challenge", "arc_easy", "gsm8k", "mbpp", "mmlu", "tier1")
BATTERY_DISPLAY = {
    "arc_challenge": "ARC-C",
    "arc_easy": "ARC-E",
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
    "mmlu": "MMLU",
    "tier1": "Tier-1",
}
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_DRAWS = 10_000


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalize_svg(path: Path) -> None:
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )


def exact_sign_test(fixes: int, regressions: int) -> float:
    discordant = fixes + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(fixes, regressions) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def document_bootstrap(
    rows: list[dict[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["document_id"]].append(row)
    documents = sorted(grouped)
    delta = np.asarray(
        [sum(int(item["augmented_correct"]) - int(item["base_correct"]) for item in grouped[doc]) for doc in documents],
        dtype=np.float64,
    )
    counts = np.asarray([len(grouped[doc]) for doc in documents], dtype=np.float64)
    denominator = np.asarray(
        [sum(int(item["teacher_14b_correct"]) - int(item["base_correct"]) for item in grouped[doc]) for doc in documents],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    delta_draws: list[np.ndarray] = []
    gap_draws: list[np.ndarray] = []
    for start in range(0, draws, 250):
        width = min(250, draws - start)
        sample = rng.integers(0, len(documents), size=(width, len(documents)))
        sampled_delta = delta[sample].sum(axis=1)
        sampled_count = counts[sample].sum(axis=1)
        sampled_denominator = denominator[sample].sum(axis=1)
        delta_draws.append(sampled_delta / sampled_count)
        gap_draws.append(
            np.divide(
                sampled_delta,
                sampled_denominator,
                out=np.full_like(sampled_delta, np.nan),
                where=sampled_denominator != 0,
            )
        )
    delta_values = np.concatenate(delta_draws)
    gap_values = np.concatenate(gap_draws)
    return {
        "accuracy_delta": [float(value) for value in np.quantile(delta_values, [0.025, 0.975])],
        "gap_closed": [float(value) for value in np.nanquantile(gap_values, [0.025, 0.975])],
    }


def score_rows(rows: list[dict[str, Any]], *, bootstrap_seed: int) -> dict[str, Any]:
    n = len(rows)
    base_correct = sum(bool(row["base_correct"]) for row in rows)
    teacher_correct = sum(bool(row["teacher_14b_correct"]) for row in rows)
    augmented_correct = sum(bool(row["augmented_correct"]) for row in rows)
    fixes = sum(not row["base_correct"] and row["augmented_correct"] for row in rows)
    regressions = sum(row["base_correct"] and not row["augmented_correct"] for row in rows)
    denominator = teacher_correct - base_correct
    delta_correct = augmented_correct - base_correct
    intervals = document_bootstrap(rows, seed=bootstrap_seed)
    return {
        "rows": n,
        "documents": len({row["document_id"] for row in rows}),
        "base_correct": base_correct,
        "base_accuracy": base_correct / n,
        "teacher_14b_correct": teacher_correct,
        "teacher_14b_accuracy": teacher_correct / n,
        "augmented_correct": augmented_correct,
        "augmented_accuracy": augmented_correct / n,
        "delta_correct": delta_correct,
        "delta_accuracy": delta_correct / n,
        "teacher_gap_denominator": denominator,
        "gap_closed": delta_correct / denominator if denominator else None,
        "fixes": fixes,
        "regressions": regressions,
        "discordant": fixes + regressions,
        "paired_exact_sign_test_p_two_sided": exact_sign_test(fixes, regressions),
        "document_bootstrap_95ci": intervals,
    }


def attach_reference(
    rows: list[dict[str, Any]], reference: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    attached = []
    for row in rows:
        ref = reference.get(row["item_id"])
        if ref is None:
            raise AssertionError(f"Missing P3.1 reference row: {row['item_id']}")
        if bool(row["base_correct"]) != bool(ref["base_correct"]):
            raise AssertionError(f"Base-score mismatch: {row['item_id']}")
        enriched = dict(row)
        enriched["teacher_14b_correct"] = bool(ref["teacher_14b_correct"])
        attached.append(enriched)
    return attached


def endpoint_breakdowns(rows: list[dict[str, Any]], seed_offset: int) -> dict[str, Any]:
    output: dict[str, Any] = {
        "pooled": score_rows(rows, bootstrap_seed=BOOTSTRAP_SEED + seed_offset),
        "by_panel_group": {},
        "by_battery": {},
    }
    for index, group in enumerate(("target", "floor"), start=1):
        subset = [row for row in rows if row["panel_group"] == group]
        output["by_panel_group"][group] = score_rows(
            subset, bootstrap_seed=BOOTSTRAP_SEED + seed_offset + index
        )
    for index, battery in enumerate(BATTERY_ORDER, start=10):
        subset = [row for row in rows if row["battery"] == battery]
        output["by_battery"][battery] = score_rows(
            subset, bootstrap_seed=BOOTSTRAP_SEED + seed_offset + index
        )
    return output


def minimum_share_margin(event: dict[str, Any]) -> float:
    read = event["read"]
    margins = []
    for name, bound in read["bounds"].items():
        share = read["shares"][name]
        margins.append(bound - share if name == "preserve" else share - bound)
    return min(margins)


def summarize_condition(
    condition: str,
    receipt_root: Path,
    reference: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    root = receipt_root / condition
    summary_path = root / "outputs/summary.json"
    status_path = receipt_root / f"{condition}-status.json"
    summary = read_json(summary_path)
    status = read_json(status_path)
    histories = []
    for entry in summary["history"]:
        histories.append(
            {
                "look": entry["look"],
                "step": entry["step"],
                "depth": entry["depth"],
                "augmented_accuracy": entry["task"]["augmented_accuracy"],
                "base_accuracy": entry["task"]["base_accuracy"],
                "delta_correct": round(
                    (entry["task"]["augmented_accuracy"] - entry["task"]["base_accuracy"])
                    * entry["task"]["rows"]
                ),
                "delta_accuracy": entry["task"]["augmented_accuracy"] - entry["task"]["base_accuracy"],
                "gap_closed": (
                    (entry["task"]["augmented_accuracy"] - entry["task"]["base_accuracy"])
                    * entry["task"]["rows"]
                    / 292
                ),
                "controller_rung_before": entry["controller"]["rung_before"],
                "controller_rung_after": entry["controller"]["rung_after"],
                "share_classification": entry["trailing_shares"]["classification"],
                "share_consecutive_misses": entry["trailing_shares"]["consecutive_misses"],
                "share_failed_contracts": entry["trailing_shares"]["failed_contracts"],
            }
        )
    final_history = summary["history"][-1]
    final_look = int(final_history["look"])
    final_rows_path = root / "private" / f"task_rows_look_{final_look:02d}.jsonl"
    final_rows = attach_reference(read_jsonl(final_rows_path), reference)
    if len(final_rows) != 1024:
        raise AssertionError(f"{condition}: expected 1024 final task rows, found {len(final_rows)}")
    final = endpoint_breakdowns(final_rows, seed_offset=100 * (CONDITIONS.index(condition) + 1))
    if not math.isclose(final["pooled"]["augmented_accuracy"], final_history["task"]["augmented_accuracy"]):
        raise AssertionError(f"{condition}: endpoint row/summary accuracy mismatch")
    best = max(histories, key=lambda entry: (entry["delta_correct"], -entry["step"]))
    audits = [entry["audit"] for entry in summary["history"] if entry.get("audit")]
    share_windows = [
        {
            "step": event["step"],
            "window_index": event["window_index"],
            "classification": event["read"]["classification"],
            "consecutive_misses": event["read"]["consecutive_misses"],
            "failed_contracts": event["read"]["failed_contracts"],
            "shares": event["read"]["shares"],
            "bounds": event["read"]["bounds"],
            "minimum_contract_margin": minimum_share_margin(event),
            "controller": event.get("controller"),
        }
        for event in summary["share_contract_events"]
    ]
    return {
        "condition": condition,
        "arm": summary["arm"],
        "seed": summary["seed"],
        "status": summary["status"],
        "step": summary["step"],
        "target_steps": summary["target_steps"],
        "stop_reason": summary["stop_reason"],
        "return_code": status["return_code"],
        "training_release": status["training_release"],
        "lock_sha256": summary["lock_sha256"],
        "schedule_sha256": summary["schedule_sha256"],
        "frozen_digest_before": summary["frozen_digest_before"],
        "frozen_digest_after": summary["frozen_digest_after"],
        "frozen_lineage_unchanged": summary["frozen_digest_before"] == summary["frozen_digest_after"],
        "confirm_scored": summary["confirm_scored"],
        "eval_e_scored": summary["eval_e_scored"],
        "trajectory": histories,
        "best_pooled_look": best,
        "final": final,
        "causal_audits": audits,
        "share_windows": share_windows,
        "source_receipts": {
            "summary": file_receipt(summary_path),
            "status": file_receipt(status_path),
            "final_rows": file_receipt(final_rows_path),
            "bundle": file_receipt(receipt_root / f"{condition}-latest-receipts.tar.zst"),
        },
    }


def write_task_figure(conditions: dict[str, dict[str, Any]], figures: Path) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "paper2-p34-20260813",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    for condition in CONDITIONS:
        trajectory = conditions[condition]["trajectory"]
        x = [0] + [entry["step"] for entry in trajectory]
        y = [0.0] + [100 * entry["delta_accuracy"] for entry in trajectory]
        linestyle = "--" if condition == "slot_seed_0" else "-"
        axes[0].plot(
            x,
            y,
            marker="o",
            markersize=4,
            linewidth=2,
            linestyle=linestyle,
            color=COLORS[condition],
            label=DISPLAY[condition],
        )
        axes[0].scatter([x[-1]], [y[-1]], marker="X", s=70, color=COLORS[condition], zorder=5)
        axes[0].annotate(
            f"stop {x[-1]:,}",
            (x[-1], y[-1]),
            xytext=(4, 7 if condition != "main_seed_1" else -15),
            textcoords="offset points",
            color=COLORS[condition],
            fontsize=8,
        )
    axes[0].axhline(0, color="#444444", linewidth=1)
    axes[0].set(
        title="DEV task trajectory",
        xlabel="Optimizer steps",
        ylabel="Accuracy change from frozen base (points)",
    )
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")

    x = np.arange(len(BATTERY_ORDER))
    width = 0.24
    for index, condition in enumerate(CONDITIONS):
        values = [
            100 * conditions[condition]["final"]["by_battery"][battery]["delta_accuracy"]
            for battery in BATTERY_ORDER
        ]
        axes[1].bar(
            x + (index - 1) * width,
            values,
            width,
            color=COLORS[condition],
            label=DISPLAY[condition],
        )
    axes[1].axhline(0, color="#444444", linewidth=1)
    axes[1].set_xticks(x, [BATTERY_DISPLAY[battery] for battery in BATTERY_ORDER], rotation=20)
    axes[1].set(
        title="Endpoint change by battery",
        ylabel="Accuracy change from frozen base (points)",
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].text(
        0.99,
        0.02,
        "X marks registered loss-share stop; all panels are DEV-only",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.suptitle("P3.4: replicated positive task signal, interrupted by objective-allocation stops", fontsize=14, fontweight="bold")
    figures.mkdir(parents=True, exist_ok=True)
    stem = figures / "p34_campaign_task_curve_20260813"
    paths = [stem.with_suffix(".svg"), stem.with_suffix(".png")]
    fig.savefig(paths[0], bbox_inches="tight", metadata={"Date": "2026-08-13"})
    normalize_svg(paths[0])
    fig.savefig(paths[1], dpi=190, bbox_inches="tight", metadata={"Creation Time": "2026-08-13"})
    plt.close(fig)
    return paths


def write_controller_figure(conditions: dict[str, dict[str, Any]], figures: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    for condition in CONDITIONS:
        windows = conditions[condition]["share_windows"]
        axes[0].plot(
            [entry["step"] for entry in windows],
            [100 * entry["minimum_contract_margin"] for entry in windows],
            marker="o",
            markersize=3.5,
            linewidth=2,
            linestyle="--" if condition == "slot_seed_0" else "-",
            color=COLORS[condition],
            label=DISPLAY[condition],
        )
    axes[0].axhline(0, color="#444444", linewidth=1)
    axes[0].set(
        title="Loss-share contract margin",
        xlabel="Optimizer steps",
        ylabel="Worst share margin (points; negative = breach)",
    )
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    audit_conditions = [condition for condition in CONDITIONS if conditions[condition]["causal_audits"]]
    y_positions = np.arange(len(audit_conditions))
    for metric, offset, marker, label in (
        ("pi_dir", -0.11, "o", "pi_dir"),
        ("pi_dep", 0.11, "s", "pi_dep"),
    ):
        points = []
        lows = []
        highs = []
        for condition in audit_conditions:
            value = conditions[condition]["causal_audits"][0][metric]
            points.append(value["point"] * 100)
            lows.append((value["point"] - value["ci95_low"]) * 100)
            highs.append((value["ci95_high"] - value["point"]) * 100)
        axes[1].errorbar(
            points,
            y_positions + offset,
            xerr=np.asarray([lows, highs]),
            fmt=marker,
            markersize=6,
            capsize=3,
            linewidth=1.5,
            color="#2878B5" if metric == "pi_dir" else "#D95F02",
            label=label,
        )
    axes[1].set_yticks(y_positions, [DISPLAY[condition] for condition in audit_conditions])
    axes[1].set(
        title="Causal capture at look 5 (step 1,000)",
        xlabel="Oracle flip capture (%) with document-bootstrap 95% CI",
    )
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")
    callouts = []
    for condition in audit_conditions:
        audit = conditions[condition]["causal_audits"][0]
        callouts.append(
            f"{DISPLAY[condition]}: gate recall {audit['gate']['recall'] * 100:.1f}%, "
            f"precision {audit['gate']['precision'] * 100:.1f}%, chi {audit['collateral_chi']:.3f}"
        )
    axes[1].text(
        0.98,
        0.03,
        "\n".join(callouts),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.92},
    )
    fig.suptitle("P3.4 controller and causal diagnostics", fontsize=14, fontweight="bold")
    figures.mkdir(parents=True, exist_ok=True)
    stem = figures / "p34_campaign_controller_20260813"
    paths = [stem.with_suffix(".svg"), stem.with_suffix(".png")]
    fig.savefig(paths[0], bbox_inches="tight", metadata={"Date": "2026-08-13"})
    normalize_svg(paths[0])
    fig.savefig(paths[1], dpi=190, bbox_inches="tight", metadata={"Creation Time": "2026-08-13"})
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-root", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--p31-scores", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()

    panel = read_jsonl(args.panel)
    reference_rows = read_jsonl(args.p31_scores)
    reference = {row["item_id"]: row for row in reference_rows}
    if len(panel) != 1024 or len({row["item_id"] for row in panel}) != 1024:
        raise AssertionError("P3.4 panel must contain exactly 1,024 unique rows")
    missing = sorted({row["item_id"] for row in panel} - set(reference))
    if missing:
        raise AssertionError(f"Panel is missing {len(missing)} P3.1 reference scores")

    conditions = {
        condition: summarize_condition(condition, args.receipt_root, reference)
        for condition in CONDITIONS
    }
    lock_hashes = {entry["lock_sha256"] for entry in conditions.values()}
    schedules = {entry["schedule_sha256"] for entry in conditions.values()}
    if len(lock_hashes) != 1:
        raise AssertionError("Campaign conditions do not share one governing lock")
    if len(schedules) != len(CONDITIONS):
        raise AssertionError("Campaign conditions do not have distinct pinned schedules")
    if not all(entry["frozen_lineage_unchanged"] for entry in conditions.values()):
        raise AssertionError("Frozen lineage changed in at least one condition")
    if any(entry["confirm_scored"] or entry["eval_e_scored"] for entry in conditions.values()):
        raise AssertionError("A sealed CONFIRM/EVAL-E partition was scored")

    task_figures = write_task_figure(conditions, args.figures_dir)
    controller_figures = write_controller_figure(conditions, args.figures_dir)
    result = {
        "kind": "paper2_phase3_p34_campaign_analysis_v1",
        "status": "complete_dev_only",
        "analysis_date": "2026-08-13",
        "verdict": "positive_signal_with_objective_controller_stop",
        "scope": "DEV-only three-condition P3.4 campaign; no CONFIRM or EVAL-E scoring",
        "registered_target_steps": 4000,
        "bootstrap": {
            "unit": "document_id",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "panel": {
            "rows": len(panel),
            "documents": len({row["document_id"] for row in panel}),
            "base_correct": sum(reference[row["item_id"]]["base_correct"] for row in panel),
            "teacher_14b_correct": sum(reference[row["item_id"]]["teacher_14b_correct"] for row in panel),
            "teacher_gap_denominator": sum(
                int(reference[row["item_id"]]["teacher_14b_correct"])
                - int(reference[row["item_id"]]["base_correct"])
                for row in panel
            ),
            "by_battery": {
                battery: {
                    "rows": sum(row["battery"] == battery for row in panel),
                    "base_correct": sum(
                        reference[row["item_id"]]["base_correct"]
                        for row in panel
                        if row["battery"] == battery
                    ),
                    "teacher_14b_correct": sum(
                        reference[row["item_id"]]["teacher_14b_correct"]
                        for row in panel
                        if row["battery"] == battery
                    ),
                }
                for battery in BATTERY_ORDER
            },
            "source": file_receipt(args.panel),
            "reference_source": file_receipt(args.p31_scores),
        },
        "conditions": conditions,
        "figure_receipts": [file_receipt(path) for path in task_figures + controller_figures],
        "integrity": {
            "one_lock_sha256": next(iter(lock_hashes)),
            "condition_schedule_sha256": {
                condition: conditions[condition]["schedule_sha256"] for condition in CONDITIONS
            },
            "all_frozen_lineages_unchanged": True,
            "confirm_scored": False,
            "eval_e_scored": False,
            "all_stops_registered_loss_share_contract": all(
                entry["stop_reason"] == "loss_share_contract" for entry in conditions.values()
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    for path in task_figures + controller_figures:
        print(path)


if __name__ == "__main__":
    main()
