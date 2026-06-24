"""Prepare MCQ rows for direct score-level surface alignment.

The earlier surface repair emitted causal SFT prompt/completion rows.  That is
useful but indirect: MCQ evaluation scores every option completion and chooses
the highest likelihood.  This builder keeps the original MCQ schema and adds
per-row scoring metadata so ``train_phase1_mcq_score_align.py`` can optimize
the correct option against distractors on the exact scoring surface that failed.
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

from eval.eval_mcq import normalize_answer, option_items  # noqa: E402
from eval.mcq_debias import cyclic_permutation_rows  # noqa: E402
from training.prepare_mcq_surface_alignment_jsonl import read_json, read_jsonl, selected_ids, write_jsonl  # noqa: E402


def normalize_mcq_row(
    row: dict[str, Any],
    *,
    prompt_style: str,
    score_target: str,
    routing_type: str,
    target_loop_count: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    choices = option_items(row)
    answer = normalize_answer(row.get("answer") or row.get("label") or row.get("target"), choices)
    out = {
        "id": str(row["id"]),
        "question": str(row["question"]),
        "choices": {str(label): str(text) for label, text in choices},
        "answer": answer,
        "prompt_style": prompt_style,
        "score_target": score_target,
        "source_dataset": "mcq-score-surface-alignment",
        "category": "mcq_score_alignment",
        "difficulty": "stable_cyclic_rescue",
        "target_loop_count": int(target_loop_count),
        "routing_type": routing_type,
    }
    if metadata:
        out.update(metadata)
    return out


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
        metadata = {
            "score_alignment_source_id": row_id,
            "score_alignment_content_prediction": diagnosis_by_id.get(row_id, {}).get(
                "candidate_content_prediction"
            ),
            "score_alignment_cyclic_prediction": diagnosis_by_id.get(row_id, {}).get(
                "candidate_cyclic_prediction"
            ),
            "score_alignment_answer_rank": diagnosis_by_id.get(row_id, {}).get(
                "candidate_content_answer_rank"
            ),
            "score_alignment_answer_margin": diagnosis_by_id.get(row_id, {}).get(
                "candidate_content_answer_margin"
            ),
        }
        for _ in range(max(1, int(content_repeat))):
            try:
                output.append(
                    normalize_mcq_row(
                        source,
                        prompt_style="question_only",
                        score_target="option_text",
                        routing_type="score_content_align",
                        target_loop_count=target_loop_count,
                        metadata=metadata,
                    )
                )
            except Exception:
                skipped["bad_mcq"] += 1

        permuted = cyclic_permutation_rows([source])
        if cyclic_rows_per_item is not None:
            permuted = permuted[: max(0, int(cyclic_rows_per_item))]
        for permuted_row in permuted:
            for _ in range(max(1, int(cyclic_repeat))):
                try:
                    output.append(
                        normalize_mcq_row(
                            permuted_row,
                            prompt_style="with_options",
                            score_target="label",
                            routing_type="score_cyclic_preserve",
                            target_loop_count=target_loop_count,
                            metadata={
                                **metadata,
                                "score_alignment_original_id": row_id,
                                "score_alignment_permutation_shift": permuted_row.get("permutation_shift"),
                            },
                        )
                    )
                except Exception:
                    skipped["bad_permutation"] += 1

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
        "score_target_counts": dict(Counter(str(row.get("score_target")) for row in output)),
        "prompt_style_counts": dict(Counter(str(row.get("prompt_style")) for row in output)),
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
