"""Prepare MCQ SFT rows for content/cyclic surface alignment.

Inputs:
* an MCQ JSONL file in the same schema consumed by ``eval/eval_mcq.py``;
* a surface-mismatch diagnosis from ``eval/analyze_mcq_surface_mismatch.py``.

For rows where content scoring lost but cyclic aggregation stably recovered the
answer, this emits:
* a question-only / option-text row to lift the content surface;
* cyclic with-options / label rows to preserve the permutation-aggregated
  surface that already recovered the correct answer.

The output is ordinary causal SFT JSONL and can be fed to
``training/train_phase1_ponder.py``.
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


def sft_row(
    example: MCQExample,
    *,
    prompt_style: str,
    score_target: str,
    source_dataset: str,
    routing_type: str,
    target_loop_count: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    choices = dict(example.choices)
    answer = str(example.answer)
    row = {
        "prompt": format_prompt(example, prompt_style),
        "completion": format_completion(answer, str(choices[answer]), score_target),
        "cot_tokens": 1,
        "source_dataset": source_dataset,
        "category": "mcq_surface_alignment",
        "difficulty": "stable_cyclic_rescue",
        "arc_id": str(example.id).split("::perm", maxsplit=1)[0],
        "answer": answer,
        "target_loop_count": int(target_loop_count),
        "routing_type": routing_type,
    }
    if metadata:
        row.update(metadata)
    return row


def selected_ids(diagnosis: dict[str, Any], *, include_unrescued: bool = False) -> tuple[set[str], dict[str, dict[str, Any]]]:
    summary = diagnosis.get("summary") or {}
    examples = list(summary.get("stable_rescue_examples") or [])
    if include_unrescued:
        examples.extend(summary.get("unrescued_loss_examples") or [])
    ids = {str(row["id"]) for row in examples if row.get("id") is not None}
    return ids, {str(row["id"]): row for row in examples if row.get("id") is not None}


def build_rows(
    mcq_rows: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    *,
    target_loop_count: int = 1,
    include_unrescued: bool = False,
    cyclic_rows_per_item: int | None = None,
    content_repeat: int = 1,
    cyclic_repeat: int = 1,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids, diagnosis_by_id = selected_ids(diagnosis, include_unrescued=include_unrescued)
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
            example = mcq_example(source)
        except Exception:
            skipped["bad_mcq"] += 1
            continue
        metadata = {
            "surface_alignment_source_id": row_id,
            "surface_alignment_content_prediction": diagnosis_by_id.get(row_id, {}).get("candidate_content_prediction"),
            "surface_alignment_cyclic_prediction": diagnosis_by_id.get(row_id, {}).get("candidate_cyclic_prediction"),
            "surface_alignment_answer_rank": diagnosis_by_id.get(row_id, {}).get("candidate_content_answer_rank"),
        }
        for _ in range(max(1, int(content_repeat))):
            output.append(
                sft_row(
                    example,
                    prompt_style="question_only",
                    score_target="option_text",
                    source_dataset="mcq-surface-content-align",
                    routing_type="surface_content_align",
                    target_loop_count=target_loop_count,
                    metadata=metadata,
                )
            )

        permuted = cyclic_permutation_rows([source])
        if cyclic_rows_per_item is not None:
            permuted = permuted[: max(0, int(cyclic_rows_per_item))]
        for permuted_row in permuted:
            try:
                permuted_example = mcq_example(permuted_row)
            except Exception:
                skipped["bad_permutation"] += 1
                continue
            for _ in range(max(1, int(cyclic_repeat))):
                output.append(
                    sft_row(
                        permuted_example,
                        prompt_style="with_options",
                        score_target="label",
                        source_dataset="mcq-surface-cyclic-preserve",
                        routing_type="surface_cyclic_preserve",
                        target_loop_count=target_loop_count,
                        metadata={
                            **metadata,
                            "surface_alignment_original_id": row_id,
                            "surface_alignment_permutation_shift": permuted_row.get("permutation_shift"),
                        },
                    )
                )

    rng.shuffle(output)
    summary = {
        "input_mcq_rows": len(mcq_rows),
        "diagnosis_benchmark": (diagnosis.get("summary") or {}).get("benchmark"),
        "selected_ids": len(ids),
        "output_rows": len(output),
        "include_unrescued": include_unrescued,
        "cyclic_rows_per_item": cyclic_rows_per_item,
        "content_repeat": content_repeat,
        "cyclic_repeat": cyclic_repeat,
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
    parser.add_argument("--include_unrescued", action="store_true")
    parser.add_argument("--cyclic_rows_per_item", type=int)
    parser.add_argument("--content_repeat", type=int, default=1)
    parser.add_argument("--cyclic_repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows, summary = build_rows(
        read_jsonl(args.mcq_jsonl),
        read_json(args.diagnosis_json),
        target_loop_count=args.target_loop_count,
        include_unrescued=args.include_unrescued,
        cyclic_rows_per_item=args.cyclic_rows_per_item,
        content_repeat=args.content_repeat,
        cyclic_repeat=args.cyclic_repeat,
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
