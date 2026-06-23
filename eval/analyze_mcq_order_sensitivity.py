"""Diagnose whether MCQ content losses are concentrated in order-sensitive rows.

This is the cheap test suggested by the current Stage 5 strategy review:
if ARC-Easy content losses mostly occur on rows whose cyclic/permutation
predictions disagree, the immediate repair should favor conditional invariance
over more base-logit knowledge distillation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def prediction_count(row: dict[str, Any]) -> int:
    counts = row.get("permutation_prediction_counts")
    if not isinstance(counts, dict):
        return 0
    return sum(int(value) for value in counts.values())


def is_order_sensitive(row: dict[str, Any]) -> bool:
    counts = row.get("permutation_prediction_counts")
    if not isinstance(counts, dict):
        return False
    nonzero = [label for label, count in counts.items() if int(count) > 0]
    return len(nonzero) > 1


def change_type(base_row: dict[str, Any], candidate_row: dict[str, Any]) -> str:
    base_hit = bool(base_row.get("hit"))
    candidate_hit = bool(candidate_row.get("hit"))
    if base_hit and not candidate_hit:
        return "loss"
    if candidate_hit and not base_hit:
        return "win"
    if base_hit and candidate_hit:
        return "tie_correct"
    return "tie_wrong"


def ratio(num: int, den: int) -> float:
    return float(num) / den if den else 0.0


def row_summary(
    row_id: str,
    base_content: dict[str, Any],
    candidate_content: dict[str, Any],
    candidate_cyclic: dict[str, Any] | None,
    base_cyclic: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_order_sensitive = is_order_sensitive(candidate_cyclic or {})
    base_order_sensitive = is_order_sensitive(base_cyclic or {})
    change = change_type(base_content, candidate_content)
    return {
        "id": row_id,
        "change": change,
        "answer": base_content.get("answer"),
        "base_content_prediction": base_content.get("prediction"),
        "candidate_content_prediction": candidate_content.get("prediction"),
        "candidate_cyclic_prediction": (candidate_cyclic or {}).get("prediction"),
        "base_cyclic_prediction": (base_cyclic or {}).get("prediction"),
        "base_content_hit": bool(base_content.get("hit")),
        "candidate_content_hit": bool(candidate_content.get("hit")),
        "candidate_cyclic_hit": bool((candidate_cyclic or {}).get("hit")),
        "base_cyclic_hit": bool((base_cyclic or {}).get("hit")),
        "candidate_order_sensitive": candidate_order_sensitive,
        "base_order_sensitive": base_order_sensitive,
        "candidate_permutation_prediction_counts": (candidate_cyclic or {}).get("permutation_prediction_counts") or {},
        "base_permutation_prediction_counts": (base_cyclic or {}).get("permutation_prediction_counts") or {},
        "candidate_num_permutations": prediction_count(candidate_cyclic or {}),
        "base_num_permutations": prediction_count(base_cyclic or {}),
        "cyclic_rescues_content_loss": change == "loss" and bool((candidate_cyclic or {}).get("hit")),
    }


def summarize(rows: list[dict[str, Any]], *, benchmark: str) -> dict[str, Any]:
    changes = Counter(row["change"] for row in rows)
    losses = [row for row in rows if row["change"] == "loss"]
    wins = [row for row in rows if row["change"] == "win"]
    base_correct = changes["loss"] + changes["tie_correct"]
    candidate_correct = changes["win"] + changes["tie_correct"]
    order_sensitive = [row for row in rows if row["candidate_order_sensitive"]]
    stable = [row for row in rows if not row["candidate_order_sensitive"]]
    losses_order_sensitive = [row for row in losses if row["candidate_order_sensitive"]]
    losses_rescued = [row for row in losses if row["cyclic_rescues_content_loss"]]
    wins_order_sensitive = [row for row in wins if row["candidate_order_sensitive"]]
    base_correct_rows = [row for row in rows if row["base_content_hit"]]
    base_correct_order_sensitive = [row for row in base_correct_rows if row["candidate_order_sensitive"]]
    base_correct_stable = [row for row in base_correct_rows if not row["candidate_order_sensitive"]]
    stable_losses = [row for row in base_correct_stable if row["change"] == "loss"]
    sensitive_losses = [row for row in base_correct_order_sensitive if row["change"] == "loss"]

    order_sensitivity_lift = ratio(len(sensitive_losses), len(base_correct_order_sensitive)) - ratio(
        len(stable_losses),
        len(base_correct_stable),
    )
    likely_order_sensitivity = bool(
        losses
        and (
            ratio(len(losses_order_sensitive), len(losses)) >= 0.5
            or order_sensitivity_lift >= 0.15
        )
    )
    likely_content_route_specific_issue = bool(
        losses
        and not likely_order_sensitivity
        and ratio(len(losses_rescued), len(losses)) >= 0.5
    )
    if likely_order_sensitivity:
        recommendation = "prioritize_conditional_invariance_repair"
    elif likely_content_route_specific_issue:
        recommendation = "diagnose_content_route_scoring_or_prompt_alignment_before_more_distillation"
    else:
        recommendation = "prioritize_direct_distillation_or_data_repair"

    return {
        "benchmark": benchmark,
        "paired_examples": len(rows),
        "base_content_correct": base_correct,
        "candidate_content_correct": candidate_correct,
        "content_delta": candidate_correct - base_correct,
        "changes": dict(changes),
        "candidate_order_sensitive_rows": len(order_sensitive),
        "candidate_order_stable_rows": len(stable),
        "candidate_order_sensitive_fraction": ratio(len(order_sensitive), len(rows)),
        "content_losses": len(losses),
        "content_losses_order_sensitive": len(losses_order_sensitive),
        "content_losses_order_sensitive_fraction": ratio(len(losses_order_sensitive), len(losses)),
        "content_losses_rescued_by_cyclic": len(losses_rescued),
        "content_losses_rescued_by_cyclic_fraction": ratio(len(losses_rescued), len(losses)),
        "content_wins": len(wins),
        "content_wins_order_sensitive": len(wins_order_sensitive),
        "content_wins_order_sensitive_fraction": ratio(len(wins_order_sensitive), len(wins)),
        "base_correct_order_sensitive_rows": len(base_correct_order_sensitive),
        "base_correct_order_stable_rows": len(base_correct_stable),
        "loss_rate_on_base_correct_order_sensitive": ratio(len(sensitive_losses), len(base_correct_order_sensitive)),
        "loss_rate_on_base_correct_order_stable": ratio(len(stable_losses), len(base_correct_stable)),
        "order_sensitivity_loss_rate_lift": order_sensitivity_lift,
        "likely_order_sensitivity_issue": likely_order_sensitivity,
        "likely_content_route_specific_issue": likely_content_route_specific_issue,
        "recommendation": recommendation,
        "loss_examples": losses[:12],
        "win_examples": wins[:12],
    }


def analyze(
    *,
    benchmark: str,
    base_content_path: Path,
    candidate_content_path: Path,
    candidate_cyclic_path: Path,
    base_cyclic_path: Path | None = None,
) -> dict[str, Any]:
    base_content = by_id(read_jsonl(base_content_path))
    candidate_content = by_id(read_jsonl(candidate_content_path))
    candidate_cyclic = by_id(read_jsonl(candidate_cyclic_path))
    base_cyclic = by_id(read_jsonl(base_cyclic_path)) if base_cyclic_path else {}
    rows = [
        row_summary(
            row_id,
            base_content[row_id],
            candidate_content[row_id],
            candidate_cyclic.get(row_id),
            base_cyclic.get(row_id),
        )
        for row_id in sorted(set(base_content) & set(candidate_content))
    ]
    return {
        "summary": summarize(rows, benchmark=benchmark),
        "rows": rows,
        "inputs": {
            "base_content": str(base_content_path),
            "candidate_content": str(candidate_content_path),
            "candidate_cyclic": str(candidate_cyclic_path),
            "base_cyclic": str(base_cyclic_path) if base_cyclic_path else None,
        },
    }


def markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        f"# MCQ Order-Sensitivity Diagnosis - {summary['benchmark']}",
        "",
        f"- Content delta: `{summary['content_delta']:+d}` "
        f"({summary['candidate_content_correct']}/{summary['paired_examples']} vs "
        f"{summary['base_content_correct']}/{summary['paired_examples']})",
        f"- Candidate order-sensitive rows: `{summary['candidate_order_sensitive_rows']}/{summary['paired_examples']}` "
        f"({summary['candidate_order_sensitive_fraction']:.3f})",
        f"- Content losses order-sensitive: `{summary['content_losses_order_sensitive']}/{summary['content_losses']}` "
        f"({summary['content_losses_order_sensitive_fraction']:.3f})",
        f"- Content losses rescued by cyclic aggregation: "
        f"`{summary['content_losses_rescued_by_cyclic']}/{summary['content_losses']}` "
        f"({summary['content_losses_rescued_by_cyclic_fraction']:.3f})",
        f"- Loss rate on base-correct order-sensitive rows: "
        f"`{summary['loss_rate_on_base_correct_order_sensitive']:.3f}`",
        f"- Loss rate on base-correct order-stable rows: "
        f"`{summary['loss_rate_on_base_correct_order_stable']:.3f}`",
        f"- Order-sensitivity loss-rate lift: `{summary['order_sensitivity_loss_rate_lift']:.3f}`",
        f"- Recommendation: `{summary['recommendation']}`",
        "",
        "## Loss Examples",
        "",
        "| id | answer | base content | recurrent content | recurrent cyclic | order-sensitive | cyclic rescue | perm counts |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for row in summary["loss_examples"]:
        lines.append(
            "| `{id}` | `{answer}` | `{base}` | `{cand}` | `{cyc}` | {sensitive} | {rescue} | `{counts}` |".format(
                id=row["id"],
                answer=row["answer"],
                base=row["base_content_prediction"],
                cand=row["candidate_content_prediction"],
                cyc=row["candidate_cyclic_prediction"],
                sensitive=row["candidate_order_sensitive"],
                rescue=row["cyclic_rescues_content_loss"],
                counts=json.dumps(row["candidate_permutation_prediction_counts"], sort_keys=True),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--base_content", required=True, type=Path)
    parser.add_argument("--candidate_content", required=True, type=Path)
    parser.add_argument("--candidate_cyclic", required=True, type=Path)
    parser.add_argument("--base_cyclic", type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_md", required=True, type=Path)
    args = parser.parse_args()

    payload = analyze(
        benchmark=args.benchmark,
        base_content_path=args.base_content,
        candidate_content_path=args.candidate_content,
        candidate_cyclic_path=args.candidate_cyclic,
        base_cyclic_path=args.base_cyclic,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_md.write_text(markdown_report(payload), encoding="utf-8")
    print(f"wrote_json={args.output_json}")
    print(f"wrote_md={args.output_md}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
