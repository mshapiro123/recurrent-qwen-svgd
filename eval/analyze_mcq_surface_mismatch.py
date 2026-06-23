"""Diagnose MCQ content-vs-cyclic scoring surface mismatches.

This complements ``analyze_mcq_order_sensitivity.py``.  The order diagnostic
asks whether content losses are concentrated in rows whose permutation
predictions disagree.  This diagnostic asks the next question: when content
scoring regresses, does cyclic aggregation recover the answer on otherwise
stable rows?  If yes, the failure looks more like a prompt/scoring-surface
mismatch than pure knowledge erosion.
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


def is_order_sensitive(row: dict[str, Any] | None) -> bool:
    counts = (row or {}).get("permutation_prediction_counts")
    if not isinstance(counts, dict):
        return False
    return sum(1 for count in counts.values() if int(count) > 0) > 1


def score_rank(row: dict[str, Any] | None, label: str | None) -> int | None:
    scores = (row or {}).get("scores")
    if not isinstance(scores, dict) or not label or label not in scores:
        return None
    ordered = sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
    for index, (candidate, _score) in enumerate(ordered, start=1):
        if candidate == label:
            return index
    return None


def answer_margin(row: dict[str, Any] | None, answer: str | None) -> float | None:
    scores = (row or {}).get("scores")
    prediction = (row or {}).get("prediction")
    if not isinstance(scores, dict) or not answer or not prediction:
        return None
    if answer not in scores or prediction not in scores:
        return None
    return float(scores[answer]) - float(scores[prediction])


def ratio(num: int, den: int) -> float:
    return float(num) / den if den else 0.0


def row_summary(
    row_id: str,
    base_content: dict[str, Any],
    candidate_content: dict[str, Any],
    candidate_cyclic: dict[str, Any] | None,
    base_cyclic: dict[str, Any] | None,
) -> dict[str, Any]:
    answer = str(base_content.get("answer"))
    base_content_hit = bool(base_content.get("hit"))
    candidate_content_hit = bool(candidate_content.get("hit"))
    candidate_cyclic_hit = bool((candidate_cyclic or {}).get("hit"))
    content_loss = base_content_hit and not candidate_content_hit
    content_win = candidate_content_hit and not base_content_hit
    cyclic_rescue = content_loss and candidate_cyclic_hit
    content_cyclic_disagree = candidate_content.get("prediction") != (candidate_cyclic or {}).get("prediction")
    candidate_order_sensitive = is_order_sensitive(candidate_cyclic)

    return {
        "id": row_id,
        "answer": answer,
        "base_content_prediction": base_content.get("prediction"),
        "candidate_content_prediction": candidate_content.get("prediction"),
        "candidate_cyclic_prediction": (candidate_cyclic or {}).get("prediction"),
        "base_cyclic_prediction": (base_cyclic or {}).get("prediction"),
        "base_content_hit": base_content_hit,
        "candidate_content_hit": candidate_content_hit,
        "candidate_cyclic_hit": candidate_cyclic_hit,
        "base_cyclic_hit": bool((base_cyclic or {}).get("hit")),
        "content_loss": content_loss,
        "content_win": content_win,
        "cyclic_rescues_content_loss": cyclic_rescue,
        "candidate_order_sensitive": candidate_order_sensitive,
        "content_cyclic_disagree": content_cyclic_disagree,
        "stable_cyclic_rescue": cyclic_rescue and not candidate_order_sensitive,
        "candidate_content_answer_rank": score_rank(candidate_content, answer),
        "candidate_cyclic_answer_rank": score_rank(candidate_cyclic, answer),
        "base_content_answer_rank": score_rank(base_content, answer),
        "base_cyclic_answer_rank": score_rank(base_cyclic, answer),
        "candidate_content_answer_margin": answer_margin(candidate_content, answer),
        "candidate_cyclic_answer_margin": answer_margin(candidate_cyclic, answer),
        "candidate_permutation_prediction_counts": (candidate_cyclic or {}).get("permutation_prediction_counts") or {},
    }


def histogram(values: list[Any]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


def summarize(rows: list[dict[str, Any]], *, benchmark: str) -> dict[str, Any]:
    content_losses = [row for row in rows if row["content_loss"]]
    content_wins = [row for row in rows if row["content_win"]]
    rescued = [row for row in content_losses if row["cyclic_rescues_content_loss"]]
    stable_rescued = [row for row in rescued if row["stable_cyclic_rescue"]]
    unrescued = [row for row in content_losses if not row["cyclic_rescues_content_loss"]]
    content_cyclic_disagree = [row for row in rows if row["content_cyclic_disagree"]]
    loss_disagree = [row for row in content_losses if row["content_cyclic_disagree"]]
    order_sensitive_losses = [row for row in content_losses if row["candidate_order_sensitive"]]

    content_base_correct = sum(1 for row in rows if row["base_content_hit"])
    content_candidate_correct = sum(1 for row in rows if row["candidate_content_hit"])
    cyclic_candidate_correct = sum(1 for row in rows if row["candidate_cyclic_hit"])

    if content_losses and ratio(len(order_sensitive_losses), len(content_losses)) >= 0.5:
        recommendation = "prioritize_conditional_invariance_repair"
    elif content_losses and ratio(len(stable_rescued), len(content_losses)) >= 0.5:
        recommendation = "prioritize_content_cyclic_surface_alignment"
    elif content_losses and ratio(len(unrescued), len(content_losses)) >= 0.5:
        recommendation = "prioritize_direct_distillation_or_data_repair"
    else:
        recommendation = "no_dominant_surface_failure_pattern"

    return {
        "benchmark": benchmark,
        "paired_examples": len(rows),
        "base_content_correct": content_base_correct,
        "candidate_content_correct": content_candidate_correct,
        "candidate_cyclic_correct": cyclic_candidate_correct,
        "content_delta": content_candidate_correct - content_base_correct,
        "cyclic_vs_candidate_content_delta": cyclic_candidate_correct - content_candidate_correct,
        "content_losses": len(content_losses),
        "content_wins": len(content_wins),
        "content_losses_rescued_by_cyclic": len(rescued),
        "content_losses_stably_rescued_by_cyclic": len(stable_rescued),
        "content_losses_unrescued_by_cyclic": len(unrescued),
        "content_losses_order_sensitive": len(order_sensitive_losses),
        "content_losses_with_content_cyclic_disagreement": len(loss_disagree),
        "content_cyclic_disagreement_rows": len(content_cyclic_disagree),
        "content_losses_rescued_by_cyclic_fraction": ratio(len(rescued), len(content_losses)),
        "content_losses_stably_rescued_by_cyclic_fraction": ratio(len(stable_rescued), len(content_losses)),
        "content_losses_unrescued_by_cyclic_fraction": ratio(len(unrescued), len(content_losses)),
        "content_losses_order_sensitive_fraction": ratio(len(order_sensitive_losses), len(content_losses)),
        "content_losses_with_content_cyclic_disagreement_fraction": ratio(len(loss_disagree), len(content_losses)),
        "candidate_content_prediction_counts": histogram([row["candidate_content_prediction"] for row in rows]),
        "candidate_cyclic_prediction_counts": histogram([row["candidate_cyclic_prediction"] for row in rows]),
        "content_loss_answer_rank_histogram": histogram(
            [row["candidate_content_answer_rank"] for row in content_losses]
        ),
        "stable_rescue_answer_rank_histogram": histogram(
            [row["candidate_content_answer_rank"] for row in stable_rescued]
        ),
        "recommendation": recommendation,
        "stable_rescue_examples": stable_rescued[:12],
        "unrescued_loss_examples": unrescued[:12],
        "content_win_examples": content_wins[:12],
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
    row_ids = sorted(set(base_content) & set(candidate_content))
    rows = [
        row_summary(
            row_id,
            base_content[row_id],
            candidate_content[row_id],
            candidate_cyclic.get(row_id),
            base_cyclic.get(row_id),
        )
        for row_id in row_ids
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
        f"# MCQ Surface-Mismatch Diagnosis - {summary['benchmark']}",
        "",
        f"- Content delta: `{summary['content_delta']:+d}` "
        f"({summary['candidate_content_correct']}/{summary['paired_examples']} vs "
        f"{summary['base_content_correct']}/{summary['paired_examples']})",
        f"- Candidate cyclic vs candidate content delta: `{summary['cyclic_vs_candidate_content_delta']:+d}`",
        f"- Content losses: `{summary['content_losses']}`",
        f"- Content losses rescued by cyclic: `{summary['content_losses_rescued_by_cyclic']}` "
        f"({summary['content_losses_rescued_by_cyclic_fraction']:.3f})",
        f"- Stable cyclic rescues: `{summary['content_losses_stably_rescued_by_cyclic']}` "
        f"({summary['content_losses_stably_rescued_by_cyclic_fraction']:.3f})",
        f"- Unrescued content losses: `{summary['content_losses_unrescued_by_cyclic']}` "
        f"({summary['content_losses_unrescued_by_cyclic_fraction']:.3f})",
        f"- Order-sensitive content losses: `{summary['content_losses_order_sensitive']}` "
        f"({summary['content_losses_order_sensitive_fraction']:.3f})",
        f"- Recommendation: `{summary['recommendation']}`",
        "",
        "## Stable Cyclic Rescue Examples",
        "",
        "| id | answer | content pred | cyclic pred | content answer rank | content answer margin | perm counts |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in summary["stable_rescue_examples"]:
        margin = row["candidate_content_answer_margin"]
        lines.append(
            "| `{id}` | `{answer}` | `{content}` | `{cyclic}` | {rank} | {margin} | `{counts}` |".format(
                id=row["id"],
                answer=row["answer"],
                content=row["candidate_content_prediction"],
                cyclic=row["candidate_cyclic_prediction"],
                rank=row["candidate_content_answer_rank"],
                margin=f"{margin:.4f}" if margin is not None else "",
                counts=json.dumps(row["candidate_permutation_prediction_counts"], sort_keys=True),
            )
        )
    lines.extend(["", "## Unrescued Content Loss Examples", ""])
    lines.append("| id | answer | content pred | cyclic pred | content answer rank | content answer margin |")
    lines.append("|---|---|---|---|---:|---:|")
    for row in summary["unrescued_loss_examples"]:
        margin = row["candidate_content_answer_margin"]
        lines.append(
            "| `{id}` | `{answer}` | `{content}` | `{cyclic}` | {rank} | {margin} |".format(
                id=row["id"],
                answer=row["answer"],
                content=row["candidate_content_prediction"],
                cyclic=row["candidate_cyclic_prediction"],
                rank=row["candidate_content_answer_rank"],
                margin=f"{margin:.4f}" if margin is not None else "",
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
