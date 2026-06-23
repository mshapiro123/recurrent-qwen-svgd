"""Prepare MCQ SFT rows for conditional invariance over option order.

Inputs:
* an MCQ JSONL file in the same schema consumed by ``eval/eval_mcq.py``;
* an order-sensitivity diagnosis from ``eval/analyze_mcq_order_sensitivity.py``.

For rows where content losses are concentrated in order-sensitive cyclic
predictions, this emits permuted-option examples that teach the same semantic
answer under nuisance option-order changes.  The primary rows use
``score_target=option_text`` so the completion is invariant to the label
assigned by each permutation.  Optional label rows preserve the cyclic label
surface without making label position the main target.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import MCQExample, format_completion, format_prompt, normalize_answer, option_items  # noqa: E402
from eval.mcq_debias import cyclic_permutation_rows  # noqa: E402


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mcq_example(row: dict[str, Any]) -> MCQExample:
    choices = option_items(row)
    answer = normalize_answer(row.get("answer") or row.get("label") or row.get("target"), choices)
    return MCQExample(
        id=str(row["id"]),
        question=str(row["question"]),
        choices=choices,
        answer=answer,
    )


def selected_ids(diagnosis: dict[str, Any], *, include_wins: bool = False) -> tuple[set[str], dict[str, dict[str, Any]]]:
    rows = diagnosis.get("rows")
    if not isinstance(rows, list):
        rows = []
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("candidate_order_sensitive"):
            continue
        change = row.get("change")
        if change == "loss" or (include_wins and change == "win"):
            selected.append(row)
    ids = {str(row["id"]) for row in selected if row.get("id") is not None}
    return ids, {str(row["id"]): row for row in selected if row.get("id") is not None}


def sft_row(
    example: MCQExample,
    *,
    score_target: str,
    source_dataset: str,
    routing_type: str,
    target_loop_count: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    choices = dict(example.choices)
    answer = str(example.answer)
    return {
        "prompt": format_prompt(example, "with_options"),
        "completion": format_completion(answer, str(choices[answer]), score_target),
        "cot_tokens": 1,
        "source_dataset": source_dataset,
        "category": "mcq_conditional_invariance",
        "difficulty": "order_sensitive",
        "arc_id": str(example.id).split("::perm", maxsplit=1)[0],
        "answer": answer,
        "target_loop_count": int(target_loop_count),
        "routing_type": routing_type,
        **metadata,
    }


def build_rows(
    mcq_rows: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    *,
    target_loop_count: int = 1,
    include_wins: bool = False,
    semantic_repeat: int = 2,
    label_repeat: int = 1,
    rows_per_item: int | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids, diagnosis_by_id = selected_ids(diagnosis, include_wins=include_wins)
    mcq_by_id = {str(row["id"]): row for row in mcq_rows}
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    skipped = Counter()

    for row_id in sorted(ids):
        source = mcq_by_id.get(row_id)
        if source is None:
            skipped["missing_mcq"] += 1
            continue
        try:
            permutations = cyclic_permutation_rows([source])
        except Exception:
            skipped["bad_mcq"] += 1
            continue
        if rows_per_item is not None:
            permutations = permutations[: max(0, int(rows_per_item))]
        diagnosis_row = diagnosis_by_id.get(row_id, {})
        for permuted_row in permutations:
            try:
                example = mcq_example(permuted_row)
            except Exception:
                skipped["bad_permutation"] += 1
                continue
            metadata = {
                "conditional_invariance_source_id": row_id,
                "conditional_invariance_permutation_shift": permuted_row.get("permutation_shift"),
                "conditional_invariance_original_answer": permuted_row.get("original_answer"),
                "conditional_invariance_content_prediction": diagnosis_row.get("candidate_content_prediction"),
                "conditional_invariance_cyclic_prediction": diagnosis_row.get("candidate_cyclic_prediction"),
                "conditional_invariance_permutation_counts": diagnosis_row.get(
                    "candidate_permutation_prediction_counts"
                )
                or {},
            }
            for _ in range(max(0, int(semantic_repeat))):
                output.append(
                    sft_row(
                        example,
                        score_target="option_text",
                        source_dataset="mcq-conditional-invariance-semantic",
                        routing_type="conditional_invariance_semantic",
                        target_loop_count=target_loop_count,
                        metadata=metadata,
                    )
                )
            for _ in range(max(0, int(label_repeat))):
                output.append(
                    sft_row(
                        example,
                        score_target="label",
                        source_dataset="mcq-conditional-invariance-label",
                        routing_type="conditional_invariance_label",
                        target_loop_count=target_loop_count,
                        metadata=metadata,
                    )
                )

    rng.shuffle(output)
    summary = {
        "input_mcq_rows": len(mcq_rows),
        "diagnosis_benchmark": (diagnosis.get("summary") or {}).get("benchmark"),
        "selected_ids": len(ids),
        "output_rows": len(output),
        "include_wins": include_wins,
        "rows_per_item": rows_per_item,
        "semantic_repeat": semantic_repeat,
        "label_repeat": label_repeat,
        "target_loop_count": target_loop_count,
        "routing_type_counts": dict(Counter(str(row.get("routing_type")) for row in output)),
        "answer_counts": dict(Counter(str(row.get("answer")) for row in output)),
        "skipped": dict(skipped),
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcq_jsonl", required=True, type=Path)
    parser.add_argument("--diagnosis_json", required=True, type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--summary_json", type=Path)
    parser.add_argument("--target_loop_count", type=int, default=1)
    parser.add_argument("--include_wins", action="store_true")
    parser.add_argument("--rows_per_item", type=int)
    parser.add_argument("--semantic_repeat", type=int, default=2)
    parser.add_argument("--label_repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows, summary = build_rows(
        read_jsonl(args.mcq_jsonl),
        read_json(args.diagnosis_json),
        target_loop_count=args.target_loop_count,
        include_wins=args.include_wins,
        rows_per_item=args.rows_per_item,
        semantic_repeat=args.semantic_repeat,
        label_repeat=args.label_repeat,
        seed=args.seed,
    )
    write_jsonl(args.output_jsonl, rows)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
