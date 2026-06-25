"""Evaluate whether a loop-depth selector transfers to held-out MCQ slices.

The forced-depth diagnostic gives an in-sample oracle over recurrent loop
logits. This script uses one sweep as the discovery set, selects a simple
inspectable router on that sweep, and evaluates the same router on a separate
held-out forced-depth sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analyze_depth_sweep import (  # noqa: E402
    depth_interaction_summary,
    joined_examples,
    load_loop_payloads,
    path_for_cli,
    resolve_path,
)
from eval.evaluate_depth_selector_split import (  # noqa: E402
    evaluate_candidate,
    loop1_baseline,
    oracle_summary,
    score_candidates,
    select_best,
    selector_signature,
    summarize_prediction_counts,
    threshold_candidates,
)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_examples(
    sweep_summary: Path,
    *,
    benchmark: str,
    score_target: str,
    aggregate: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    _sweep, loop_payloads = load_loop_payloads(sweep_summary)
    loops = sorted(loop_payloads)
    return loops, joined_examples(loop_payloads, benchmark, score_target, aggregate)


def candidate_pool(loops: list[int]) -> list[dict[str, Any]]:
    return threshold_candidates(loops) + score_candidates(loops)


def fit_selector(examples: list[dict[str, Any]], loops: list[int]) -> dict[str, Any]:
    rows = [evaluate_candidate(examples, candidate) for candidate in candidate_pool(loops)]
    return select_best(rows)


def evaluated_row(
    examples: list[dict[str, Any]],
    selector: dict[str, Any],
    loops: list[int],
) -> dict[str, Any]:
    baseline = loop1_baseline(examples)
    oracle = oracle_summary(examples, loops)
    selected = evaluate_candidate(examples, selector)
    selector_gain = int(selected["delta_vs_loop1"])
    oracle_gain = int(oracle["any_depth_gain_vs_loop1"])
    base_correct = int(baseline["base_correct"])
    selected_correct = int(selected["correct"])
    return {
        "baseline": baseline,
        "oracle": oracle,
        "selected_selector": selected,
        "selected_delta_vs_base": selected_correct - base_correct,
        "oracle_gap_capture": selector_gain / oracle_gain if oracle_gain > 0 else None,
        "prediction_bias": summarize_prediction_counts(selected),
        "depth_interactions": depth_interaction_summary(examples, loops),
    }


def transfer_summary(
    *,
    discovery_sweep: Path,
    heldout_sweep: Path,
    train_benchmark: str,
    score_target: str,
    aggregate: str,
    min_oracle_gap_capture: float,
) -> dict[str, Any]:
    discovery_loops, discovery_examples = load_examples(
        discovery_sweep,
        benchmark=train_benchmark,
        score_target=score_target,
        aggregate=aggregate,
    )
    selector = fit_selector(discovery_examples, discovery_loops)
    discovery_eval = evaluated_row(discovery_examples, selector, discovery_loops)

    heldout_sweep_payload, heldout_loop_payloads = load_loop_payloads(heldout_sweep)
    heldout_loops = sorted(heldout_loop_payloads)
    if heldout_loops != discovery_loops:
        raise ValueError(f"Loop mismatch: discovery={discovery_loops}, heldout={heldout_loops}")

    benchmarks = list(heldout_loop_payloads[heldout_loops[0]].get("benchmarks", []))
    heldout: dict[str, Any] = {}
    for benchmark in benchmarks:
        examples = joined_examples(heldout_loop_payloads, benchmark, score_target, aggregate)
        heldout[benchmark] = evaluated_row(examples, selector, heldout_loops)

    deltas_vs_loop1 = [
        int(row["selected_selector"]["delta_vs_loop1"])
        for row in heldout.values()
    ]
    deltas_vs_base = [int(row["selected_delta_vs_base"]) for row in heldout.values()]
    captures = [
        float(row["oracle_gap_capture"])
        for row in heldout.values()
        if row["oracle_gap_capture"] is not None
    ]
    positive_loop1 = sum(1 for value in deltas_vs_loop1 if value > 0)
    negative_loop1 = sum(1 for value in deltas_vs_loop1 if value < 0)
    positive_base = sum(1 for value in deltas_vs_base if value > 0)
    mean_delta_loop1 = mean([float(value) for value in deltas_vs_loop1])
    mean_delta_base = mean([float(value) for value in deltas_vs_base])
    mean_capture = mean(captures)

    if (
        positive_loop1 >= max(1, len(heldout) // 2 + 1)
        and positive_base >= max(1, len(heldout) // 2 + 1)
        and mean_capture is not None
        and mean_capture >= min_oracle_gap_capture
    ):
        gate_status = "router_transfer_passed"
        recommended_next = "learned_depth_router_training"
    elif mean_delta_loop1 is not None and mean_delta_loop1 > 0:
        gate_status = "router_transfer_partial"
        recommended_next = "manifold_alignment_before_router"
    else:
        gate_status = "router_transfer_failed"
        recommended_next = "scale_probe_or_depth_signal_rethink"

    return {
        "kind": "stage5_depth_router_transfer",
        "discovery_sweep_summary": path_for_cli(discovery_sweep),
        "heldout_sweep_summary": path_for_cli(heldout_sweep),
        "heldout_sweep_run_id": heldout_sweep_payload.get("run_id"),
        "train_benchmark": train_benchmark,
        "score_target": score_target,
        "aggregate": aggregate,
        "loops": discovery_loops,
        "selector": selector,
        "selector_signature": selector_signature(selector),
        "discovery": discovery_eval,
        "heldout": heldout,
        "min_oracle_gap_capture": min_oracle_gap_capture,
        "transfer_summary": {
            "benchmarks": list(heldout),
            "positive_delta_vs_loop1_benchmarks": positive_loop1,
            "negative_delta_vs_loop1_benchmarks": negative_loop1,
            "positive_delta_vs_base_benchmarks": positive_base,
            "mean_delta_vs_loop1": mean_delta_loop1,
            "mean_delta_vs_base": mean_delta_base,
            "mean_oracle_gap_capture": mean_capture,
        },
        "gate_status": gate_status,
        "recommended_next": recommended_next,
        "notes": [
            "Content scoring is the capability gate; cyclic scoring is an invariance guard.",
            "Oracle is reported as an upper bound, not as achieved gain.",
            "Subspace overlap is not measured by MCQ rows; use re-entry drift diagnostics for that instrument.",
        ],
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Depth Router Transfer - {payload['heldout_sweep_run_id']}",
        "",
        f"- Discovery: `{payload['discovery_sweep_summary']}`",
        f"- Held-out: `{payload['heldout_sweep_summary']}`",
        f"- Score target / aggregate: `{payload['score_target']}` / `{payload['aggregate']}`",
        f"- Selector: `{payload['selector_signature']}`",
        f"- Gate status: `{payload['gate_status']}`",
        f"- Recommended next: `{payload['recommended_next']}`",
        "",
        "## Discovery",
        "",
    ]
    discovery = payload["discovery"]
    base = discovery["baseline"]
    selected = discovery["selected_selector"]
    oracle = discovery["oracle"]
    lines.extend(
        [
            f"- loop1: `{base['loop1_correct']}/{base['total']}`",
            f"- base: `{base['base_correct']}/{base['total']}`",
            f"- selected: `{selected['correct']}/{selected['total']}` "
            f"(delta vs loop1 `{selected['delta_vs_loop1']}`, "
            f"delta vs base `{discovery['selected_delta_vs_base']}`)",
            f"- any-depth oracle: `{oracle['any_depth_correct']}/{base['total']}` "
            f"(gain vs loop1 `{oracle['any_depth_gain_vs_loop1']}`)",
            f"- oracle gap capture: `{discovery['oracle_gap_capture']}`",
            "",
            "## Held-Out",
            "",
        ]
    )
    for benchmark, row in payload["heldout"].items():
        base = row["baseline"]
        selected = row["selected_selector"]
        oracle = row["oracle"]
        churn = row["depth_interactions"]
        lines.extend(
            [
                f"### {benchmark}",
                "",
                f"- loop1: `{base['loop1_correct']}/{base['total']}`",
                f"- base: `{base['base_correct']}/{base['total']}`",
                f"- selected: `{selected['correct']}/{selected['total']}` "
                f"(delta vs loop1 `{selected['delta_vs_loop1']}`, "
                f"delta vs base `{row['selected_delta_vs_base']}`, "
                f"W/L `{selected['wins_vs_loop1']}/{selected['losses_vs_loop1']}`)",
                f"- any-depth oracle: `{oracle['any_depth_correct']}/{base['total']}` "
                f"(gain vs loop1 `{oracle['any_depth_gain_vs_loop1']}`)",
                f"- oracle gap capture: `{row['oracle_gap_capture']}`",
                f"- deeper unique over loop1: `{churn['deeper_unique_over_loop1']}`",
                f"- loop1 harmed by deeper loops: `{churn['loop1_harmed_by_any_deeper']}`",
                f"- hit patterns: `{churn['depth_hit_patterns']}`",
                "",
            ]
        )
    summary = payload["transfer_summary"]
    lines.extend(
        [
            "## Transfer Summary",
            "",
            f"- positive delta vs loop1 benchmarks: `{summary['positive_delta_vs_loop1_benchmarks']}`",
            f"- negative delta vs loop1 benchmarks: `{summary['negative_delta_vs_loop1_benchmarks']}`",
            f"- positive delta vs base benchmarks: `{summary['positive_delta_vs_base_benchmarks']}`",
            f"- mean delta vs loop1: `{summary['mean_delta_vs_loop1']}`",
            f"- mean delta vs base: `{summary['mean_delta_vs_base']}`",
            f"- mean oracle gap capture: `{summary['mean_oracle_gap_capture']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery_sweep_summary", required=True)
    parser.add_argument("--heldout_sweep_summary", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--train_benchmark", default="arc_challenge")
    parser.add_argument("--score_target", default="content_question_only")
    parser.add_argument("--aggregate", default="mean")
    parser.add_argument("--min_oracle_gap_capture", type=float, default=0.2)
    args = parser.parse_args()

    discovery = resolve_path(args.discovery_sweep_summary)
    heldout = resolve_path(args.heldout_sweep_summary)
    output_dir = resolve_path(args.output_dir) if args.output_dir else heldout.parent / f"router_transfer_{args.score_target}"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = transfer_summary(
        discovery_sweep=discovery,
        heldout_sweep=heldout,
        train_benchmark=args.train_benchmark,
        score_target=args.score_target,
        aggregate=args.aggregate,
        min_oracle_gap_capture=args.min_oracle_gap_capture,
    )
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, output_dir / "summary.md")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
