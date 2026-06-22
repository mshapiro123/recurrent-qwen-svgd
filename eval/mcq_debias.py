"""Utilities for diagnosing MCQ option-label selection bias.

These helpers are deliberately model-free. They prepare cyclic option
permutations and aggregate scored permutation rows back to the original option
content labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("A", "B", "C", "D", "E", "F")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choice_items(row: dict[str, Any]) -> list[tuple[str, str]]:
    choices = row.get("choices") or row.get("options")
    if isinstance(choices, dict):
        labels = [label for label in LABELS if label in choices]
        labels.extend(label for label in choices if label not in labels)
        return [(str(label), str(choices[label])) for label in labels]
    if isinstance(choices, list):
        if len(choices) > len(LABELS):
            raise ValueError(f"Too many choices: {len(choices)}")
        return list(zip(LABELS, [str(item) for item in choices]))
    raise ValueError("Row must contain choices/options as a dict or list.")


def rotate_mcq_row(row: dict[str, Any], shift: int) -> dict[str, Any]:
    items = choice_items(row)
    labels = [label for label, _text in items]
    n = len(labels)
    if n == 0:
        raise ValueError("Cannot rotate row with no choices.")
    shift = shift % n
    answer = str(row.get("answer") or row.get("label") or row.get("target"))
    choices: dict[str, str] = {}
    label_map: dict[str, str] = {}
    for new_idx, new_label in enumerate(labels):
        orig_idx = (new_idx - shift) % n
        orig_label, orig_text = items[orig_idx]
        choices[new_label] = orig_text
        label_map[new_label] = orig_label
    new_answer = next(new_label for new_label, orig_label in label_map.items() if orig_label == answer)
    return {
        "id": f"{row.get('id')}::perm{shift}",
        "original_id": str(row.get("id")),
        "permutation_shift": shift,
        "question": row["question"],
        "choices": choices,
        "answer": new_answer,
        "original_answer": answer,
        "label_map": label_map,
    }


def cyclic_permutation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    permuted: list[dict[str, Any]] = []
    for row in rows:
        for shift in range(len(choice_items(row))):
            permuted.append(rotate_mcq_row(row, shift))
    return permuted


def aggregate_permutation_scores(
    scored_rows: list[dict[str, Any]],
    permutation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Average permuted label scores in original content-label space."""

    scored_by_id = {str(row["id"]): row for row in scored_rows}
    groups: dict[str, dict[str, Any]] = {}
    for perm_row in permutation_rows:
        scored = scored_by_id.get(str(perm_row["id"]))
        if scored is None:
            continue
        original_id = str(perm_row["original_id"])
        group = groups.setdefault(
            original_id,
            {
                "id": original_id,
                "answer": str(perm_row["original_answer"]),
                "score_lists": {},
                "permutation_predictions": [],
            },
        )
        label_map = {str(k): str(v) for k, v in (perm_row.get("label_map") or {}).items()}
        for new_label, score in (scored.get("scores") or {}).items():
            original_label = label_map.get(str(new_label))
            if original_label is None:
                continue
            group["score_lists"].setdefault(original_label, []).append(float(score))
        prediction = scored.get("prediction")
        if prediction in label_map:
            group["permutation_predictions"].append(label_map[str(prediction)])

    aggregated: list[dict[str, Any]] = []
    for original_id, group in sorted(groups.items()):
        scores = {
            label: sum(values) / max(len(values), 1)
            for label, values in sorted(group["score_lists"].items())
        }
        if not scores:
            continue
        prediction = max(scores, key=scores.get)
        answer = str(group["answer"])
        aggregated.append(
            {
                "id": original_id,
                "aggregate": "permutation_mean",
                "prediction": prediction,
                "answer": answer,
                "hit": prediction == answer,
                "scores": scores,
                "num_permutations": min(len(values) for values in group["score_lists"].values()),
                "permutation_prediction_counts": dict(Counter(group["permutation_predictions"])),
            }
        )
    return aggregated


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for row in rows if row.get("hit"))
    total = len(rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / max(total, 1),
        "prediction_counts": dict(Counter(str(row.get("prediction")) for row in rows)),
        "answer_counts": dict(Counter(str(row.get("answer")) for row in rows)),
        "edge_minus_middle": edge_minus_middle(dict(Counter(str(row.get("prediction")) for row in rows))),
    }


def edge_minus_middle(counts: dict[str, int]) -> int:
    return int(counts.get("A", 0) + counts.get("D", 0) - counts.get("B", 0) - counts.get("C", 0))

