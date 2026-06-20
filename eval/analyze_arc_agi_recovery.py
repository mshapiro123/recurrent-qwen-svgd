"""Analyze ARC-AGI recurrent recovery gaps from saved eval summaries.

This is a no-GPU diagnostic layer. It reads base, recurrent-start, and
recovered recurrent eval summaries and identifies where to spend the next
training run: generation/curriculum, selector/TTA, parser/format discipline, or
checkpoint rollback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from eval.compare_arc_agi_runs import compare_payloads
    from eval.eval_arc_agi import task_family
except ModuleNotFoundError:  # pragma: no cover - direct ``python eval/script.py`` execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.compare_arc_agi_runs import compare_payloads
    from eval.eval_arc_agi import task_family


ExampleKey = tuple[str, int]


def read_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def example_key(example: dict[str, Any]) -> ExampleKey:
    return str(example["task_id"]), int(example.get("test_index", 0))


def examples_by_key(payload: dict[str, Any] | None) -> dict[ExampleKey, dict[str, Any]]:
    if not payload:
        return {}
    return {
        example_key(example): example
        for example in payload.get("examples", [])
        if example.get("has_target")
    }


def exact(example: dict[str, Any] | None, metric: str) -> bool:
    return bool((example or {}).get(metric))


def valid_candidates(example: dict[str, Any] | None) -> int:
    return int((example or {}).get("valid_candidates", 0))


def candidate_count(example: dict[str, Any] | None) -> int:
    return int((example or {}).get("num_candidates", 0))


def classify_example(example: dict[str, Any] | None) -> str:
    if not example or not example.get("has_target"):
        return "unscored"
    if exact(example, "selected_exact"):
        return "selected_exact"
    if exact(example, "best_of_k_exact"):
        return "selector_miss"
    if valid_candidates(example) == 0:
        return "no_valid_candidate"
    return "no_exact_candidate"


def increment(bucket: dict[str, int], key: str, value: int = 1) -> None:
    bucket[key] = bucket.get(key, 0) + value


def model_failure_buckets(payload: dict[str, Any] | None) -> dict[str, int]:
    buckets: dict[str, int] = {}
    for example in (payload or {}).get("examples", []):
        increment(buckets, classify_example(example))
    return buckets


def summary_snapshot(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "summary": payload.get("summary", {}),
        "candidate_source_summary": payload.get("candidate_source_summary", {}),
        "parse_method_summary": payload.get("parse_method_summary", {}),
        "program_verifier_summary": payload.get("program_verifier_summary", {}),
        "failure_buckets": model_failure_buckets(payload),
    }


def paired_metric_delta(
    reference: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    *,
    reference_label: str,
    candidate_label: str,
) -> dict[str, Any] | None:
    if not reference or not candidate:
        return None
    return compare_payloads(
        reference,
        candidate,
        reference_label=reference_label,
        candidate_label=candidate_label,
        bootstrap_samples=0,
        seed=0,
    )


def family_gap_rows(
    reference: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    *,
    reference_label: str,
    candidate_label: str,
) -> list[dict[str, Any]]:
    ref_examples = examples_by_key(reference)
    cand_examples = examples_by_key(candidate)
    by_family: dict[str, dict[str, Any]] = {}

    for key in sorted(set(ref_examples) & set(cand_examples)):
        ref = ref_examples[key]
        cand = cand_examples[key]
        family = task_family(key[0])
        row = by_family.setdefault(
            family,
            {
                "family": family,
                "paired_examples": 0,
                f"{reference_label}_selected_exact": 0,
                f"{candidate_label}_selected_exact": 0,
                f"{reference_label}_best_of_k_exact": 0,
                f"{candidate_label}_best_of_k_exact": 0,
                f"{candidate_label}_selector_misses": 0,
                f"{candidate_label}_no_valid_candidate": 0,
                f"{candidate_label}_no_exact_candidate": 0,
                "selected_base_only": 0,
                "selected_candidate_only": 0,
                "best_base_only": 0,
                "best_candidate_only": 0,
            },
        )
        row["paired_examples"] += 1
        ref_selected = exact(ref, "selected_exact")
        cand_selected = exact(cand, "selected_exact")
        ref_best = exact(ref, "best_of_k_exact")
        cand_best = exact(cand, "best_of_k_exact")
        row[f"{reference_label}_selected_exact"] += int(ref_selected)
        row[f"{candidate_label}_selected_exact"] += int(cand_selected)
        row[f"{reference_label}_best_of_k_exact"] += int(ref_best)
        row[f"{candidate_label}_best_of_k_exact"] += int(cand_best)
        row[f"{candidate_label}_selector_misses"] += int(cand_best and not cand_selected)
        row[f"{candidate_label}_no_valid_candidate"] += int(valid_candidates(cand) == 0)
        row[f"{candidate_label}_no_exact_candidate"] += int(valid_candidates(cand) > 0 and not cand_best)
        row["selected_base_only"] += int(ref_selected and not cand_selected)
        row["selected_candidate_only"] += int(cand_selected and not ref_selected)
        row["best_base_only"] += int(ref_best and not cand_best)
        row["best_candidate_only"] += int(cand_best and not ref_best)

    rows = []
    for row in by_family.values():
        row["selected_delta"] = row[f"{candidate_label}_selected_exact"] - row[f"{reference_label}_selected_exact"]
        row["best_of_k_delta"] = row[f"{candidate_label}_best_of_k_exact"] - row[f"{reference_label}_best_of_k_exact"]
        rows.append(row)
    return sorted(
        rows,
        key=lambda item: (
            int(item["selected_delta"]),
            int(item["best_of_k_delta"]),
            -int(item["paired_examples"]),
            str(item["family"]),
        ),
    )


def regression_examples(
    reference: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    *,
    metric: str,
    limit: int,
) -> list[dict[str, Any]]:
    ref_examples = examples_by_key(reference)
    cand_examples = examples_by_key(candidate)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(ref_examples) & set(cand_examples)):
        ref = ref_examples[key]
        cand = cand_examples[key]
        if exact(ref, metric) and not exact(cand, metric):
            rows.append(
                {
                    "task_id": key[0],
                    "test_index": key[1],
                    "family": task_family(key[0]),
                    "candidate_failure": classify_example(cand),
                    "candidate_valid_candidates": valid_candidates(cand),
                    "candidate_num_candidates": candidate_count(cand),
                    "candidate_selected_index": cand.get("selected_index"),
                }
            )
    return rows[:limit]


def recommendation_rows(
    *,
    recovered_vs_base: dict[str, Any] | None,
    recovered_vs_start: dict[str, Any] | None,
    recovered: dict[str, Any],
) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    base_selected_delta = int(((recovered_vs_base or {}).get("metrics", {}).get("selected_exact") or {}).get("delta_exact", 0))
    base_best_delta = int(((recovered_vs_base or {}).get("metrics", {}).get("best_of_k_exact") or {}).get("delta_exact", 0))
    start_best_delta = int(((recovered_vs_start or {}).get("metrics", {}).get("best_of_k_exact") or {}).get("delta_exact", 0))
    buckets = model_failure_buckets(recovered)
    selector_misses = buckets.get("selector_miss", 0)
    no_valid = buckets.get("no_valid_candidate", 0)
    no_exact = buckets.get("no_exact_candidate", 0)

    if base_best_delta < 0:
        recommendations.append(
            {
                "area": "curriculum_generation",
                "reason": "Recovered recurrent has fewer best-of-K exact candidates than base; train generation/latent reasoning before relying on reranking.",
            }
        )
    if base_selected_delta < 0 and selector_misses > 0:
        recommendations.append(
            {
                "area": "selector_or_tta",
                "reason": "Recovered recurrent sometimes contains an exact candidate but fails to select it; prioritize selector/TTA/self-consistency diagnostics.",
            }
        )
    if no_valid > 0:
        recommendations.append(
            {
                "area": "format_parse",
                "reason": "Some recovered examples produce no valid parsed grid; tighten output-format training and parser-aware loss/eval.",
            }
        )
    if start_best_delta > 0 and (base_selected_delta < 0 or base_best_delta < 0):
        recommendations.append(
            {
                "area": "scale_recovery",
                "reason": "Recovery improved over the recurrent start but still trails base; scale deterministic recovery before adding stronger particle pressure.",
            }
        )
    if start_best_delta < 0:
        recommendations.append(
            {
                "area": "rollback_or_data_audit",
                "reason": "Recovered checkpoint regressed versus the recurrent start; audit candidate-distillation targets before more SFT.",
            }
        )
    if no_exact > selector_misses and base_best_delta < 0:
        recommendations.append(
            {
                "area": "family_targeted_sft",
                "reason": "Most recovered misses lack any exact candidate; favor family-targeted SFT/curriculum over selector tuning.",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "area": "confirm",
                "reason": "No obvious recovery gap was detected; increase ARC limit or run full-split confirmation.",
            }
        )
    return recommendations


def analyze_recovery(
    *,
    base: dict[str, Any],
    recovered: dict[str, Any],
    start: dict[str, Any] | None = None,
    top_examples: int = 20,
) -> dict[str, Any]:
    recovered_vs_base = paired_metric_delta(base, recovered, reference_label="base", candidate_label="recovered")
    recovered_vs_start = (
        paired_metric_delta(start, recovered, reference_label="phase1_start", candidate_label="recovered")
        if start
        else None
    )
    start_vs_base = (
        paired_metric_delta(base, start, reference_label="base", candidate_label="phase1_start")
        if start
        else None
    )
    return {
        "models": {
            "base": summary_snapshot(base),
            "phase1_start": summary_snapshot(start),
            "recovered": summary_snapshot(recovered),
        },
        "paired": {
            "phase1_start_vs_base": start_vs_base,
            "recovered_vs_start": recovered_vs_start,
            "recovered_vs_base": recovered_vs_base,
        },
        "family_gaps": {
            "recovered_vs_base": family_gap_rows(base, recovered, reference_label="base", candidate_label="recovered"),
            "recovered_vs_start": family_gap_rows(start, recovered, reference_label="phase1_start", candidate_label="recovered")
            if start
            else [],
        },
        "regression_examples": {
            "base_selected_recovered_missed": regression_examples(
                base,
                recovered,
                metric="selected_exact",
                limit=top_examples,
            ),
            "base_best_recovered_missed": regression_examples(
                base,
                recovered,
                metric="best_of_k_exact",
                limit=top_examples,
            ),
        },
        "recommendations": recommendation_rows(
            recovered_vs_base=recovered_vs_base,
            recovered_vs_start=recovered_vs_start,
            recovered=recovered,
        ),
    }


def metric_line(payload: dict[str, Any] | None, metric: str) -> str:
    if not payload:
        return "n/a"
    stats = (payload.get("metrics") or {}).get(metric) or {}
    return (
        f"delta `{stats.get('delta_exact')}`; "
        f"candidate `{stats.get('candidate_exact')}` / `{stats.get('paired_examples')}`; "
        f"reference `{stats.get('reference_exact')}` / `{stats.get('paired_examples')}`; "
        f"W/L/T `{stats.get('wins')}/{stats.get('losses')}/{stats.get('ties')}`"
    )


def write_markdown(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    lines = [
        "# ARC-AGI Recurrent Recovery Analysis",
        "",
        "## Paired Metrics",
        "",
    ]
    for name, comparison in payload["paired"].items():
        if not comparison:
            continue
        lines += [
            f"### {name}",
            f"- Selected exact: {metric_line(comparison, 'selected_exact')}",
            f"- Best-of-K exact: {metric_line(comparison, 'best_of_k_exact')}",
            "",
        ]

    lines += ["## Recovered Failure Buckets", ""]
    buckets = payload["models"]["recovered"].get("failure_buckets", {})
    for key, value in sorted(buckets.items()):
        lines.append(f"- `{key}`: `{value}`")

    lines += ["", "## Worst Family Gaps: Recovered vs Base", ""]
    lines.append("| Family | Selected delta | Best-of-K delta | Base-only selected | Base-only best | Recovered selector misses | Recovered no exact |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["family_gaps"]["recovered_vs_base"][:10]:
        lines.append(
            f"| `{row['family']}` | {row['selected_delta']} | {row['best_of_k_delta']} | "
            f"{row['selected_base_only']} | {row['best_base_only']} | "
            f"{row['recovered_selector_misses']} | {row['recovered_no_exact_candidate']} |"
        )

    lines += ["", "## Top Base-Correct / Recovered-Missed Examples", ""]
    for row in payload["regression_examples"]["base_selected_recovered_missed"][:10]:
        lines.append(
            f"- `{row['task_id']}#{row['test_index']}` family `{row['family']}`: "
            f"{row['candidate_failure']}, valid `{row['candidate_valid_candidates']}` / `{row['candidate_num_candidates']}`"
        )

    lines += ["", "## Recommended Next Moves", ""]
    for item in payload["recommendations"]:
        lines.append(f"- `{item['area']}`: {item['reason']}")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_summary_json", required=True)
    parser.add_argument("--start_summary_json")
    parser.add_argument("--recovered_summary_json", required=True)
    parser.add_argument("--top_examples", type=int, default=20)
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    args = parser.parse_args()

    base = read_json(args.base_summary_json)
    recovered = read_json(args.recovered_summary_json)
    start = read_json(args.start_summary_json)
    assert base is not None and recovered is not None
    payload = analyze_recovery(base=base, start=start, recovered=recovered, top_examples=args.top_examples)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(json.dumps({"recommendations": payload["recommendations"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
