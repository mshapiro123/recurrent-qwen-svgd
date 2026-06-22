"""Build direct-preservation SFT rows from base-correct MCQ eval outputs."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import MCQExample, format_completion, format_prompt  # noqa: E402


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_margin(row: dict[str, Any]) -> float | None:
    scores = row.get("scores") or {}
    answer = row.get("answer")
    if answer not in scores:
        return None
    other_scores = [float(score) for label, score in scores.items() if label != answer]
    if not other_scores:
        return None
    return float(scores[answer]) - max(other_scores)


def mcq_to_sft(
    mcq: dict[str, Any],
    eval_row: dict[str, Any],
    *,
    prompt_style: str,
    score_target: str,
    target_loop_count: int,
    routing_type: str,
) -> dict[str, Any]:
    choices_dict = dict(mcq["choices"])
    answer = str(mcq["answer"])
    example = MCQExample(
        id=str(mcq["id"]),
        question=str(mcq["question"]),
        choices=[(str(label), str(text)) for label, text in choices_dict.items()],
        answer=answer,
    )
    margin = score_margin(eval_row)
    return {
        "prompt": format_prompt(example, prompt_style),
        "completion": format_completion(answer, str(choices_dict[answer]), score_target),
        "cot_tokens": 1,
        "source_dataset": "base-correct-mcq-direct-preservation",
        "category": str(eval_row.get("benchmark") or eval_row.get("mode") or "mcq"),
        "difficulty": "base_correct_direct",
        "arc_id": str(mcq["id"]),
        "answer": answer,
        "target_loop_count": int(target_loop_count),
        "routing_type": routing_type,
        "base_margin": margin,
        "base_prediction": eval_row.get("prediction"),
        "base_scores": eval_row.get("scores"),
    }


def select_rows(
    mcq_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    min_base_margin: float,
    prompt_style: str = "with_options",
    score_target: str = "label",
    target_loop_count: int = 1,
    routing_type: str = "direct_base_preserve",
    balance_labels: bool = True,
    max_rows_per_label: int | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mcq_by_id = {str(row["id"]): row for row in mcq_rows}
    selected: list[dict[str, Any]] = []
    skipped = Counter()
    for eval_row in eval_rows:
        row_id = str(eval_row.get("id"))
        mcq = mcq_by_id.get(row_id)
        if mcq is None:
            skipped["missing_mcq"] += 1
            continue
        if not bool(eval_row.get("hit")):
            skipped["base_wrong"] += 1
            continue
        margin = score_margin(eval_row)
        if margin is None or not math.isfinite(margin):
            skipped["missing_margin"] += 1
            continue
        if margin < min_base_margin:
            skipped["low_margin"] += 1
            continue
        selected.append(
            mcq_to_sft(
                mcq,
                eval_row,
                prompt_style=prompt_style,
                score_target=score_target,
                target_loop_count=target_loop_count,
                routing_type=routing_type,
            )
        )

    pre_balance_counts = Counter(str(row["answer"]) for row in selected)
    if balance_labels and selected:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[str(row["answer"])].append(row)
        rng = random.Random(seed)
        for rows in grouped.values():
            rng.shuffle(rows)
        cap = min(len(rows) for rows in grouped.values())
        if max_rows_per_label is not None:
            cap = min(cap, int(max_rows_per_label))
        selected = [row for label in sorted(grouped) for row in grouped[label][:cap]]
        rng.shuffle(selected)
    elif max_rows_per_label is not None:
        grouped = defaultdict(list)
        for row in selected:
            grouped[str(row["answer"])].append(row)
        rng = random.Random(seed)
        selected = []
        for label in sorted(grouped):
            rows = grouped[label]
            rng.shuffle(rows)
            selected.extend(rows[: int(max_rows_per_label)])
        rng.shuffle(selected)

    summary = {
        "input_mcq_rows": len(mcq_rows),
        "input_eval_rows": len(eval_rows),
        "selected_rows": len(selected),
        "skipped": dict(skipped),
        "pre_balance_answer_counts": dict(sorted(pre_balance_counts.items())),
        "answer_counts": dict(sorted(Counter(str(row["answer"]) for row in selected).items())),
        "min_base_margin": min_base_margin,
        "balance_labels": balance_labels,
        "max_rows_per_label": max_rows_per_label,
        "target_loop_count": target_loop_count,
        "routing_type": routing_type,
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcq_jsonl", required=True)
    parser.add_argument("--base_eval_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json")
    parser.add_argument("--prompt_style", choices=("with_options", "question_only"), default="with_options")
    parser.add_argument("--score_target", choices=("label", "option_text", "label_and_text"), default="label")
    parser.add_argument("--target_loop_count", type=int, default=1)
    parser.add_argument("--routing_type", default="direct_base_preserve")
    parser.add_argument("--min_base_margin", type=float, default=1.0)
    parser.add_argument("--balance_labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_rows_per_label", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows, summary = select_rows(
        read_jsonl(args.mcq_jsonl),
        read_jsonl(args.base_eval_jsonl),
        min_base_margin=args.min_base_margin,
        prompt_style=args.prompt_style,
        score_target=args.score_target,
        target_loop_count=args.target_loop_count,
        routing_type=args.routing_type,
        balance_labels=args.balance_labels,
        max_rows_per_label=args.max_rows_per_label,
        seed=args.seed,
    )
    write_jsonl(args.output_jsonl, rows)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
