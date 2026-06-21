"""Assess whether selector work converts same-recipe recurrent coverage.

Use this after ``assess_stage5_recipe_control.py`` returns
``needs_selector_conversion`` and ``run_stage5_arc_agi_rescore_selectors.py``
has rescored the recurrent candidate files. The gate compares each rescored
recurrent selector summary against the dense SFT control from the same recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
try:
    from colab.assess_stage5_recipe_control import (
        difficulty_stats,
        eval_payload_from_summary,
        evidence_fragment,
        metric_stats,
        paired_examples,
        supports_nonnegative,
        supports_positive,
    )
    from eval.compare_arc_agi_runs import compare_payloads
except ModuleNotFoundError:  # pragma: no cover - direct ``python colab/script.py`` execution
    sys.path.insert(0, str(ROOT))
    from colab.assess_stage5_recipe_control import (
        difficulty_stats,
        eval_payload_from_summary,
        evidence_fragment,
        metric_stats,
        paired_examples,
        supports_nonnegative,
        supports_positive,
    )
    from eval.compare_arc_agi_runs import compare_payloads


RUN_ID = os.environ.get("STAGE5_RECIPE_SELECTOR_CONVERSION_RUN_ID") or time.strftime(
    "stage5_recipe_selector_conversion_%Y%m%d_%H%M%S"
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def latest_matching(scan_root: Path, predicate) -> Path | None:
    candidates: list[Path] = []
    if not scan_root.exists():
        return None
    for path in scan_root.glob("**/summary.json"):
        payload = safe_read_json(path)
        if payload and predicate(payload):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def is_recipe_needing_selector(payload: dict[str, Any]) -> bool:
    return payload.get("gate") == "stage5_same_recipe_architecture" and payload.get("status") == "needs_selector_conversion"


def is_selector_rescore(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("rows"), list) and "best_by_label" in payload and "strategies" in payload


def selector_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        strategy = str(row.get("selection_strategy", ""))
        if strategy.startswith("original:"):
            continue
        if not row.get("output_summary_json"):
            continue
        rows.append(row)
    return rows


def row_label(row: dict[str, Any]) -> str:
    return f"{row.get('label')}:{row.get('selection_strategy')}"


def evaluate_selector_row(
    *,
    dense_payload: dict[str, Any],
    row: dict[str, Any],
    hard_bucket: str,
    min_total_examples: int,
    min_hard_examples: int,
) -> dict[str, Any]:
    selector_summary_path = resolve_path(str(row["output_summary_json"]))
    selector_payload = read_json(selector_summary_path)
    label = row_label(row)
    comparison = compare_payloads(
        dense_payload,
        selector_payload,
        reference_label="dense_tuned",
        candidate_label=label,
        bootstrap_samples=1000,
        seed=0,
    )
    aggregate = metric_stats(comparison, "selected_exact")
    hard = difficulty_stats(comparison, metric_name="selected_exact", bucket=hard_bucket)
    aggregate_best = metric_stats(comparison, "best_of_k_exact")
    hard_best = difficulty_stats(comparison, metric_name="best_of_k_exact", bucket=hard_bucket)
    total_sufficient = paired_examples(aggregate) >= min_total_examples
    hard_sufficient = paired_examples(hard) >= min_hard_examples
    aggregate_positive = supports_positive(aggregate)
    aggregate_nonnegative = supports_nonnegative(aggregate)
    hard_positive = hard_sufficient and supports_positive(hard)
    hard_nonnegative = bool(hard) and hard_sufficient and supports_nonnegative(hard)
    passed = total_sufficient and hard_sufficient and (
        (aggregate_positive and hard_nonnegative)
        or (hard_positive and aggregate_nonnegative)
    )
    tradeoff = hard_positive and not aggregate_nonnegative
    return {
        "label": str(row.get("label")),
        "selection_strategy": str(row.get("selection_strategy")),
        "selector_summary_json": path_for_cli(selector_summary_path),
        "passed": passed,
        "tradeoff": tradeoff,
        "total_sufficient": total_sufficient,
        "hard_sufficient": hard_sufficient,
        "aggregate_positive": aggregate_positive,
        "aggregate_nonnegative": aggregate_nonnegative,
        "hard_positive": hard_positive,
        "hard_nonnegative": hard_nonnegative,
        "aggregate": aggregate,
        "hard": hard,
        "aggregate_best_of_k": aggregate_best,
        "hard_best_of_k": hard_best,
        "aggregate_evidence": evidence_fragment(aggregate),
        "hard_evidence": evidence_fragment(hard),
        "aggregate_best_of_k_evidence": evidence_fragment(aggregate_best),
        "hard_best_of_k_evidence": evidence_fragment(hard_best),
        "comparison": comparison,
    }


def assess_recipe_selector_conversion(
    *,
    recipe_control_summary: Path,
    selector_rescore_summary: Path,
    hard_bucket: str = "hard",
    min_total_examples: int = 20,
    min_hard_examples: int = 5,
) -> dict[str, Any]:
    recipe_payload = read_json(recipe_control_summary)
    selector_payload = read_json(selector_rescore_summary)
    dense_summary_value = recipe_payload.get("dense_summary")
    if not dense_summary_value:
        raise ValueError("Recipe-control summary is missing dense_summary path.")
    dense_summary_path = resolve_path(str(dense_summary_value))
    dense_summary = read_json(dense_summary_path)
    dense_payload = eval_payload_from_summary(dense_summary_path, dense_summary, "dense_tuned")

    evaluated_rows = [
        evaluate_selector_row(
            dense_payload=dense_payload,
            row=row,
            hard_bucket=hard_bucket,
            min_total_examples=min_total_examples,
            min_hard_examples=min_hard_examples,
        )
        for row in selector_rows(selector_payload)
    ]
    passing = [row for row in evaluated_rows if row["passed"]]
    tradeoffs = [row for row in evaluated_rows if row["tradeoff"]]
    insufficient = [row for row in evaluated_rows if not row["total_sufficient"] or not row["hard_sufficient"]]

    if passing:
        status = "passed"
        reason = "At least one recurrent selector converts best-of-K coverage into selected-answer lift versus the dense control."
        next_step = "Rerun or update the same-recipe architecture assessment using the passing selector setting."
    elif not evaluated_rows:
        status = "needs_selector_rescore"
        reason = "No rescored recurrent selector summaries were available."
        next_step = "Run colab/run_stage5_arc_agi_rescore_selectors.py for the recurrent same-recipe run."
    elif tradeoffs:
        status = "needs_review"
        reason = "A selector improves the hard bucket versus dense but harms aggregate selected accuracy."
        next_step = "Inspect selector tradeoffs before treating conversion as architecture evidence."
    elif insufficient:
        status = "needs_more_evidence"
        reason = "Selector conversion evidence has too few paired examples or hard-bucket examples."
        next_step = "Replicate the selector conversion check on a larger stratified slice."
    else:
        status = "failed"
        reason = "No recurrent selector improved selected-answer accuracy versus the dense control."
        next_step = "Improve selector/verifier logic or recurrent training before claiming architecture value."

    best = max(
        evaluated_rows,
        key=lambda row: (
            int((row["aggregate"] or {}).get("delta_exact", 0) or 0),
            int((row["hard"] or {}).get("delta_exact", 0) or 0),
            int((row["aggregate_best_of_k"] or {}).get("delta_exact", 0) or 0),
        ),
        default=None,
    )
    return {
        "run_id": RUN_ID,
        "gate": "stage5_same_recipe_selector_conversion",
        "kind": "recipe_selector_conversion",
        "status": status,
        "passed": status == "passed",
        "reason": reason,
        "next_step": next_step,
        "recipe_control_summary": path_for_cli(recipe_control_summary),
        "selector_rescore_summary": path_for_cli(selector_rescore_summary),
        "dense_summary": path_for_cli(dense_summary_path),
        "hard_bucket": hard_bucket,
        "min_total_examples": min_total_examples,
        "min_hard_examples": min_hard_examples,
        "passing_selectors": [
            {"label": row["label"], "selection_strategy": row["selection_strategy"]}
            for row in passing
        ],
        "tradeoff_selectors": [
            {"label": row["label"], "selection_strategy": row["selection_strategy"]}
            for row in tradeoffs
        ],
        "best_selector": (
            {"label": best["label"], "selection_strategy": best["selection_strategy"]}
            if best
            else None
        ),
        "selector_evidence": evaluated_rows,
    }


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    write_json(output_json, payload)
    lines = [
        f"# Stage 5 Same-Recipe Selector Conversion - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Recipe control: `{payload['recipe_control_summary']}`",
        f"- Selector rescore: `{payload['selector_rescore_summary']}`",
        f"- Dense summary: `{payload['dense_summary']}`",
        f"- Reason: {payload['reason']}",
        f"- Next step: {payload['next_step']}",
        "",
        "## Selector Evidence",
        "",
        "| Label | Selector | Passed | Aggregate selected | Hard selected |",
        "|---|---|---:|---|---|",
    ]
    for row in payload["selector_evidence"]:
        lines.append(
            f"| `{row['label']}` | `{row['selection_strategy']}` | `{row['passed']}` | "
            f"{row['aggregate_evidence']} | {row['hard_evidence']} |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe_control_summary")
    parser.add_argument("--selector_rescore_summary")
    parser.add_argument("--scan_root", default="outputs/stage5")
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    parser.add_argument("--hard_bucket", default="hard")
    parser.add_argument("--min_total_examples", type=int, default=20)
    parser.add_argument("--min_hard_examples", type=int, default=5)
    args = parser.parse_args()

    scan_root = resolve_path(args.scan_root)
    recipe_summary = (
        resolve_path(args.recipe_control_summary)
        if args.recipe_control_summary
        else latest_matching(scan_root, is_recipe_needing_selector)
    )
    selector_summary = (
        resolve_path(args.selector_rescore_summary)
        if args.selector_rescore_summary
        else latest_matching(scan_root, is_selector_rescore)
    )
    if recipe_summary is None:
        raise FileNotFoundError("No recipe-control summary with needs_selector_conversion found.")
    if selector_summary is None:
        raise FileNotFoundError("No selector-rescore summary found.")
    output_dir = ROOT / "outputs" / "stage5" / RUN_ID
    output_json = resolve_path(args.output_json) if args.output_json else output_dir / "summary.json"
    output_md = resolve_path(args.output_md) if args.output_md else output_json.with_suffix(".md")
    payload = assess_recipe_selector_conversion(
        recipe_control_summary=recipe_summary,
        selector_rescore_summary=selector_summary,
        hard_bucket=args.hard_bucket,
        min_total_examples=args.min_total_examples,
        min_hard_examples=args.min_hard_examples,
    )
    write_report(payload, output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
