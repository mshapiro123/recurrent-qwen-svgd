"""CPU-only expert-choice re-scoring of the frozen D0 allocation audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_speculative_depth_d1_causal_allocation import (
    AUDIT_SEED,
    FOLDS,
    MAX_LOOPS,
    _ridge_scores,
    cross_fitted_utility,
    source_fold,
)


BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.27)
WINDOWS = (256, 1024)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def binary_auc(scores: torch.Tensor, *, helps: torch.Tensor, hurts: torch.Tensor) -> float:
    """Mann-Whitney AUC over helps versus hurts, excluding neutral positions."""

    positive = scores[helps].double()
    negative = scores[hurts].double()
    if not len(positive) or not len(negative):
        return float("nan")
    values = torch.cat([positive, negative])
    labels = torch.cat(
        [torch.ones(len(positive), dtype=torch.bool), torch.zeros(len(negative), dtype=torch.bool)]
    )
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    average_ranks = torch.empty(len(values), dtype=torch.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = average_ranks[labels].sum()
    mann_whitney = rank_sum - len(positive) * (len(positive) + 1) / 2.0
    return float(mann_whitney / (len(positive) * len(negative)))


def _capacity(size: int, budget_fraction: float) -> int:
    return max(1, min(size, int(math.ceil(float(budget_fraction) * size))))


def causal_window_expert_choice(
    scores: torch.Tensor,
    *,
    row_indices: torch.Tensor,
    budget_fraction: float,
    window: int,
) -> torch.Tensor:
    """Select online top-budget scores using current and preceding row-local positions.

    Source-row boundaries reset the rolling window. Ties favor earlier positions,
    which makes the rule deterministic and prevents a current tied score from
    displacing information already available to the router.
    """

    if scores.ndim != 1 or row_indices.shape != scores.shape:
        raise ValueError("scores and row_indices must be aligned one-dimensional tensors")
    if not 0 < budget_fraction <= 1 or int(window) < 1:
        raise ValueError("budget_fraction and window must be positive")
    ranks, sizes = causal_window_ranks(scores, row_indices=row_indices, window=window)
    capacities = torch.ceil(sizes.float() * float(budget_fraction)).long().clamp_min(1)
    return ranks.lt(capacities)


def causal_window_ranks(
    scores: torch.Tensor, *, row_indices: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic online rank and available-window size per position."""

    ranks = torch.empty(len(scores), dtype=torch.long)
    sizes = torch.empty(len(scores), dtype=torch.long)
    start = 0
    while start < len(scores):
        row = int(row_indices[start])
        end = start + 1
        while end < len(scores) and int(row_indices[end]) == row:
            end += 1
        values = scores[start:end]
        count = len(values)
        current = torch.arange(count).unsqueeze(1)
        prior = torch.arange(count).unsqueeze(0)
        available = prior.le(current) & prior.ge(current - int(window) + 1)
        higher = values.unsqueeze(0).gt(values.unsqueeze(1))
        earlier_tie = values.unsqueeze(0).eq(values.unsqueeze(1)) & prior.lt(current)
        ranks[start:end] = ((higher | earlier_tie) & available).sum(dim=1)
        sizes[start:end] = torch.minimum(
            torch.arange(1, count + 1), torch.full((count,), int(window))
        )
        start = end
    return ranks, sizes


def global_expert_choice(scores: torch.Tensor, budget_fraction: float) -> torch.Tensor:
    capacity = _capacity(len(scores), budget_fraction)
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    selected = torch.zeros(len(scores), dtype=torch.bool)
    selected[order[:capacity]] = True
    return selected


def score_selected_second_loop(matches: torch.Tensor, selected: torch.Tensor) -> dict[str, Any]:
    if matches.ndim != 2 or matches.shape[1] < 2 or selected.shape != matches[:, 0].shape:
        raise ValueError("second-loop scoring requires aligned [positions, loops] outcomes")
    helps = ~matches[:, 0] & matches[:, 1]
    hurts = matches[:, 0] & ~matches[:, 1]
    neutral = ~(helps | hurts)
    final = torch.where(selected, matches[:, 1], matches[:, 0])
    selected_count = int(selected.sum())
    selected_non_neutral = int((selected & (helps | hurts)).sum())
    selected_helps = int((selected & helps).sum())
    selected_hurts = int((selected & hurts).sum())
    return {
        "selected": selected_count,
        "selected_fraction": selected_count / len(matches),
        "helps": selected_helps,
        "hurts": selected_hurts,
        "neutral": int((selected & neutral).sum()),
        "net_correct_delta": selected_helps - selected_hurts,
        "correct": int(final.sum()),
        "total": len(matches),
        "accuracy": float(final.float().mean()),
        "mean_loops": 1.0 + selected_count / len(matches),
        "precision_help_all_selected": selected_helps / selected_count if selected_count else 0.0,
        "precision_help_non_neutral": (
            selected_helps / selected_non_neutral if selected_non_neutral else 0.0
        ),
    }


def reconstruct_oof_scores(cache: dict[str, Any]) -> torch.Tensor:
    scalars = cache["scalars"].float()
    matches = cache["matches"].bool()
    metadata = cache["metadata"]
    structural = torch.tensor(
        [
            [
                math.log1p(float(row["sequence_length"])),
                float(row["local_position"]) / max(1.0, float(row["sequence_length"] - 1)),
                float(str(row["stratum"]) == "code"),
            ]
            for row in metadata
        ],
        dtype=torch.float32,
    )
    features = torch.cat(
        [scalars[:, : MAX_LOOPS - 1], structural[:, None, :].repeat(1, MAX_LOOPS - 1, 1)],
        dim=-1,
    )
    folds = torch.tensor(
        [source_fold(int(row["row_index"]), str(row["stratum"])) for row in metadata]
    )
    scores = torch.empty((len(matches), MAX_LOOPS - 1), dtype=torch.float32)
    for outer in range(FOLDS):
        validation_fold = (outer + 1) % FOLDS
        train = (folds.ne(outer) & folds.ne(validation_fold)).nonzero().flatten()
        test = folds.eq(outer).nonzero().flatten()
        for loop in range(MAX_LOOPS - 1):
            labels = (~matches[:, loop]) & matches[:, loop + 1]
            scores[test, loop] = _ridge_scores(features[:, loop], labels, train, test)
    return scores


def curve_replay_diagnostics(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    *,
    maximum_accuracy_rate_delta: float = 1e-5,
    maximum_loop_mean_delta: float = 5e-5,
) -> dict[str, Any]:
    """Compare a reconstructed policy curve to its banked aggregate receipt.

    The D1 cache banked features and outcomes but not fitted OOF scores. Refit
    ridge scores can move at decision boundaries across BLAS/PyTorch builds.
    Structural fields must match exactly; decision counts must remain within a
    scale-normalized numerical-equivalence envelope that is recorded here.
    """

    diagnostics: dict[str, Any] = {
        "left_rows": len(left),
        "right_rows": len(right),
        "maximum_accuracy_rate_delta_allowed": float(maximum_accuracy_rate_delta),
        "maximum_loop_mean_delta_allowed": float(maximum_loop_mean_delta),
        "structural_fields_match": True,
        "decision_fields_bit_exact": True,
        "within_numerical_equivalence_envelope": True,
        "maximum_correct_count_difference": 0,
        "maximum_loop_sum_difference": 0.0,
        "maximum_derived_absolute_difference": 0.0,
        "differences": [],
        "mismatches": [],
    }
    if len(left) != len(right):
        diagnostics["structural_fields_match"] = False
        diagnostics["within_numerical_equivalence_envelope"] = False
        diagnostics["mismatches"].append(
            {"field": "curve_length", "left": len(left), "right": len(right)}
        )
        diagnostics["pass"] = False
        return diagnostics

    for row_index, (a, b) in enumerate(zip(left, right, strict=True)):
        for key in ("penalty", "total"):
            if a[key] != b[key]:
                diagnostics["structural_fields_match"] = False
                diagnostics["within_numerical_equivalence_envelope"] = False
                diagnostics["mismatches"].append(
                    {"row": row_index, "field": key, "left": a[key], "right": b[key]}
                )
        total = max(int(a["total"]), int(b["total"]), 1)
        correct_difference = abs(int(a["correct"]) - int(b["correct"]))
        loop_sum_difference = abs(
            float(a["mean_loops"]) * int(a["total"])
            - float(b["mean_loops"]) * int(b["total"])
        )
        diagnostics["maximum_correct_count_difference"] = max(
            diagnostics["maximum_correct_count_difference"], correct_difference
        )
        diagnostics["maximum_loop_sum_difference"] = max(
            diagnostics["maximum_loop_sum_difference"], loop_sum_difference
        )
        if correct_difference or loop_sum_difference:
            diagnostics["decision_fields_bit_exact"] = False
            diagnostics["differences"].append(
                {
                    "row": row_index,
                    "correct_count_difference": correct_difference,
                    "loop_sum_difference": loop_sum_difference,
                }
            )
        if correct_difference / total > maximum_accuracy_rate_delta:
            diagnostics["within_numerical_equivalence_envelope"] = False
            diagnostics["mismatches"].append(
                {
                    "row": row_index,
                    "field": "correct",
                    "left": a["correct"],
                    "right": b["correct"],
                    "rate_difference": correct_difference / total,
                }
            )
        if loop_sum_difference / total > maximum_loop_mean_delta:
            diagnostics["within_numerical_equivalence_envelope"] = False
            diagnostics["mismatches"].append(
                {
                    "row": row_index,
                    "field": "mean_loops",
                    "left": a["mean_loops"],
                    "right": b["mean_loops"],
                    "absolute_difference": loop_sum_difference / total,
                }
            )
        for key in ("accuracy", "mean_loops", "net_utility"):
            difference = abs(float(a[key]) - float(b[key]))
            diagnostics["maximum_derived_absolute_difference"] = max(
                diagnostics["maximum_derived_absolute_difference"], difference
            )
    diagnostics["status"] = (
        "bit_exact"
        if diagnostics["structural_fields_match"]
        and diagnostics["decision_fields_bit_exact"]
        else "numerically_equivalent_not_bit_exact"
        if diagnostics["structural_fields_match"]
        and diagnostics["within_numerical_equivalence_envelope"]
        else "outside_equivalence_envelope"
    )
    diagnostics["pass"] = bool(
        diagnostics["structural_fields_match"]
        and diagnostics["within_numerical_equivalence_envelope"]
    )
    return diagnostics


def floor_transition_archaeology(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matches = torch.tensor(
        [
            row["matches"][:2]
            if "matches" in row
            else [
                int(token) == int(row["teacher_7b"])
                for token in row["predictions"][:2]
            ]
            for row in rows
        ],
        dtype=torch.bool,
    )
    helps = int((~matches[:, 0] & matches[:, 1]).sum())
    hurts = int((matches[:, 0] & ~matches[:, 1]).sum())
    return {
        "positions": len(matches),
        "helps": helps,
        "hurts": hurts,
        "net_correct_delta": helps - hurts,
        "harm_to_help_ratio": hurts / helps if helps else None,
    }


def write_report(summary: dict[str, Any], output_summary: Path) -> None:
    lines = [
        "# D0 Expert-Choice Rung 0",
        "",
        "CPU-only re-scoring of the frozen out-of-fold scalar probe. No features, folds, or model weights changed.",
        "",
        f"- OOF help-vs-hurt AUC: `{summary['score_auc_help_vs_hurt']:.4f}`",
        f"- Local verdict: `{summary['local_signal_verdict']}`",
        f"- Banked curve replay: `{summary['banked_curve_replay']['status']}`",
        f"- Pre-D0 harm/help ratio: `{summary['pre_d0_floor_transition_1_to_2']['harm_to_help_ratio']}`",
        f"- Post-D0 harm/help ratio: `{summary['post_d0_transition_1_to_2']['harm_to_help_ratio']}`",
        "",
        "## Causal Local Windows",
        "",
        "| Window | Budget | Helps | Hurts | Net delta | Mean loops |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for window, rows in summary["causal_local_expert_choice"].items():
        for row in rows:
            lines.append(
                f"| {window} | {100 * row['budget_fraction']:.1f}% | {row['helps']} | "
                f"{row['hurts']} | {row['net_correct_delta']} | {row['mean_loops']:.4f} |"
            )
    output_summary.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for window, rows in summary["causal_local_expert_choice"].items():
        axes[0].plot(
            [100 * row["budget_fraction"] for row in rows],
            [row["net_correct_delta"] for row in rows],
            marker="o",
            label=f"causal W={window}",
        )
    axes[0].plot(
        [100 * row["budget_fraction"] for row in summary["global_expert_choice"]],
        [row["net_correct_delta"] for row in summary["global_expert_choice"]],
        marker="s",
        linestyle="--",
        label="global ceiling",
    )
    axes[0].plot(
        [100 * row["budget_fraction"] for row in summary["matched_capacity_oracle"]],
        [row["net_correct_delta"] for row in summary["matched_capacity_oracle"]],
        marker="^",
        linestyle=":",
        label="matched-capacity oracle",
    )
    axes[0].axhline(0, color="#555555", linewidth=1)
    axes[0].set_xlabel("Allocated positions (%)")
    axes[0].set_ylabel("Correct-token delta (helps - hurts)")
    axes[0].set_title("Expert-choice utility")
    axes[0].legend(frameon=False)

    pre = summary["pre_d0_floor_transition_1_to_2"]
    post = summary["post_d0_transition_1_to_2"]
    axes[1].bar([0, 1], [pre["helps"], post["helps"]], width=0.35, label="helps")
    axes[1].bar(
        [0.37, 1.37], [pre["hurts"], post["hurts"]], width=0.35, label="hurts"
    )
    axes[1].set_xticks([0.185, 1.185], ["pre-D0 floor", "post-D0"])
    axes[1].set_ylabel("Positions")
    axes[1].set_title("1 to 2 transition archaeology")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_summary.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output_summary.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_cache", required=True)
    parser.add_argument("--audit_summary", required=True)
    parser.add_argument("--floor_private_rows", required=True)
    parser.add_argument("--output_summary", required=True)
    args = parser.parse_args()

    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    audit = json.loads(Path(args.audit_summary).read_text(encoding="utf-8"))
    replay = cross_fitted_utility(
        scalars=cache["scalars"], matches=cache["matches"], metadata=cache["metadata"]
    )
    banked_curve = audit["evaluation"]["cross_fitted_scalar_policy"]["curve"]
    replay_diagnostics = curve_replay_diagnostics(replay["curve"], banked_curve)
    print(
        "banked_curve_replay=" + json.dumps(replay_diagnostics, sort_keys=True),
        flush=True,
    )
    if not replay_diagnostics["pass"]:
        raise RuntimeError(
            "frozen cross-fit reconstruction does not replay the banked audit curve: "
            + json.dumps(replay_diagnostics, sort_keys=True)
        )

    scores = reconstruct_oof_scores(cache)[:, 0]
    matches = cache["matches"].bool()
    metadata = cache["metadata"]
    row_indices = torch.tensor([int(row["row_index"]) for row in metadata])
    helps = ~matches[:, 0] & matches[:, 1]
    hurts = matches[:, 0] & ~matches[:, 1]
    global_rows = []
    local_rows = {str(window): [] for window in WINDOWS}
    local_rank_cache = {
        window: causal_window_ranks(scores, row_indices=row_indices, window=window)
        for window in WINDOWS
    }
    for budget in BUDGETS:
        global_rows.append(
            {"budget_fraction": budget, **score_selected_second_loop(matches, global_expert_choice(scores, budget))}
        )
        for window in WINDOWS:
            ranks, sizes = local_rank_cache[window]
            capacities = torch.ceil(sizes.float() * budget).long().clamp_min(1)
            selected = ranks.lt(capacities)
            local_rows[str(window)].append(
                {"budget_fraction": budget, **score_selected_second_loop(matches, selected)}
            )
    oracle = []
    for budget in BUDGETS:
        capacity = _capacity(len(matches), budget)
        selected = torch.zeros(len(matches), dtype=torch.bool)
        selected[helps.nonzero().flatten()[:capacity]] = True
        oracle.append({"budget_fraction": budget, **score_selected_second_loop(matches, selected)})

    floor_payload = json.loads(Path(args.floor_private_rows).read_text(encoding="utf-8"))
    floor_rows = floor_payload.get("all_position_rows", floor_payload.get("rows", []))
    summary = {
        "kind": "paper2_d0_expert_choice_rung0",
        "status": "complete",
        "training_started": False,
        "optimizer_steps": 0,
        "audit_seed": AUDIT_SEED,
        "budgets": list(BUDGETS),
        "causal_windows": list(WINDOWS),
        "oof_score_source": "deterministic reconstruction of frozen grouped five-fold ridge probe",
        "banked_curve_replay": replay_diagnostics,
        "banked_curve_replay_exact_counts": replay_diagnostics[
            "decision_fields_bit_exact"
        ],
        "banked_oof_scores_were_saved": False,
        "score_auc_help_vs_hurt": binary_auc(scores, helps=helps, hurts=hurts),
        "post_d0_transition_1_to_2": floor_transition_archaeology(
            [{"matches": row.tolist()} for row in matches]
        ),
        "pre_d0_floor_transition_1_to_2": floor_transition_archaeology(floor_rows),
        "global_expert_choice": global_rows,
        "causal_local_expert_choice": local_rows,
        "matched_capacity_oracle": oracle,
        "references": {
            "fixed_depth_1": score_selected_second_loop(
                matches, torch.zeros(len(matches), dtype=torch.bool)
            ),
            "fixed_depth_2": score_selected_second_loop(
                matches, torch.ones(len(matches), dtype=torch.bool)
            ),
            "audit_oracle_interpretation": "matched-capacity one-step ceiling reported above",
        },
        "local_signal_verdict": (
            "all_local_budgets_negative"
            if all(
                row["net_correct_delta"] < 0
                for rows in local_rows.values()
                for row in rows
            )
            else "some_local_budget_nonnegative"
        ),
    }
    write_json(args.output_summary, summary)
    write_report(summary, Path(args.output_summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
