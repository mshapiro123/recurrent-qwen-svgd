"""Compare synthetic-depth active-label and final-answer readers.

This is a CPU-only diagnostic for cases where the active intermediate-label
matrix and the final-answer MCQ matrix appear to disagree.  It aligns rows by
``(id, forced_loop_count)`` and builds a diagonal cross-tab so we do not mistake
reader, option-set, or prompt-surface differences for a model capability result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["id"]), int(row["forced_loop_count"])): row for row in rows}


def _safe_choice_text(data_row: dict[str, Any] | None, label: Any) -> str | None:
    if not data_row:
        return None
    choices = data_row.get("choices")
    if not isinstance(choices, dict):
        return None
    return None if label is None else choices.get(str(label))


def _maybe_summary_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.exists() else None


def analyze(
    *,
    active_rows_path: str | Path,
    final_rows_path: str | Path,
    final_data_jsonl: str | Path,
    chain_data_jsonl: str | Path | None = None,
    active_summary_path: str | Path | None = None,
    final_summary_path: str | Path | None = None,
    example_limit: int = 5,
) -> dict[str, Any]:
    active_rows = read_jsonl(active_rows_path)
    final_rows = read_jsonl(final_rows_path)
    final_data = rows_by_id(read_jsonl(final_data_jsonl))
    chain_data = rows_by_id(read_jsonl(chain_data_jsonl)) if chain_data_jsonl else {}
    active_summary = read_json(active_summary_path) if _maybe_summary_path(active_summary_path) else {}
    final_summary = read_json(final_summary_path) if _maybe_summary_path(final_summary_path) else {}

    active_by_key = rows_by_key(active_rows)
    final_by_key = rows_by_key(final_rows)
    matched_keys = sorted(set(active_by_key) & set(final_by_key))

    depths = sorted({int(active_by_key[key]["depth"]) for key in matched_keys})
    by_depth: dict[str, Any] = {}
    total_active_right_final_wrong = 0
    total_diagonal_rows = 0

    for depth in depths:
        diagonal_keys = [
            key
            for key in matched_keys
            if int(active_by_key[key]["depth"]) == depth and int(active_by_key[key]["forced_loop_count"]) == depth
        ]
        table = Counter()
        active_right_final_wrong_examples: list[dict[str, Any]] = []
        active_right_final_wrong_in_final_choices = 0
        active_right_final_wrong_same_pred_symbol = 0
        active_right_final_wrong_same_target_symbol = 0

        for key in diagonal_keys:
            active = active_by_key[key]
            final = final_by_key[key]
            data_row = final_data.get(str(active["id"]))
            chain_row = chain_data.get(str(active["id"]))
            final_prediction_symbol = _safe_choice_text(data_row, final.get("prediction"))
            final_answer_symbol = _safe_choice_text(data_row, final.get("answer"))
            active_prediction = str(active.get("prediction"))
            active_target = None if active.get("target") is None else str(active.get("target"))
            active_hit = bool(active.get("hit"))
            final_hit = bool(final.get("hit"))
            table[(active_hit, final_hit)] += 1
            if active_hit and not final_hit:
                total_active_right_final_wrong += 1
                choices = (data_row or {}).get("choices") or {}
                in_final_choices = active_prediction in {str(value) for value in choices.values()}
                same_pred_symbol = active_prediction == final_prediction_symbol
                same_target_symbol = active_target == final_answer_symbol
                active_right_final_wrong_in_final_choices += int(in_final_choices)
                active_right_final_wrong_same_pred_symbol += int(same_pred_symbol)
                active_right_final_wrong_same_target_symbol += int(same_target_symbol)
                if len(active_right_final_wrong_examples) < example_limit:
                    active_right_final_wrong_examples.append(
                        {
                            "id": active.get("id"),
                            "depth": depth,
                            "active_prediction_symbol": active_prediction,
                            "active_target_symbol": active_target,
                            "final_prediction_label": final.get("prediction"),
                            "final_prediction_symbol": final_prediction_symbol,
                            "final_answer_label": final.get("answer"),
                            "final_answer_symbol": final_answer_symbol,
                            "active_prediction_in_final_choices": in_final_choices,
                            "active_prediction_equals_final_prediction_symbol": same_pred_symbol,
                            "active_target_equals_final_answer_symbol": same_target_symbol,
                            "final_choices": choices,
                            "chain_choices": (chain_row or {}).get("choices"),
                        }
                    )
        total_diagonal_rows += len(diagonal_keys)
        by_depth[str(depth)] = {
            "diagonal_rows": len(diagonal_keys),
            "table": {
                "both_right": table[(True, True)],
                "active_right_final_wrong": table[(True, False)],
                "active_wrong_final_right": table[(False, True)],
                "both_wrong": table[(False, False)],
            },
            "active_right_final_wrong": {
                "count": table[(True, False)],
                "active_prediction_in_final_choices": active_right_final_wrong_in_final_choices,
                "active_prediction_equals_final_prediction_symbol": active_right_final_wrong_same_pred_symbol,
                "active_target_equals_final_answer_symbol": active_right_final_wrong_same_target_symbol,
                "examples": active_right_final_wrong_examples,
            },
        }

    reader_mismatch = {
        "active_data_jsonl": active_summary.get("data_jsonl"),
        "final_data_jsonl": final_summary.get("data_jsonl"),
        "active_prediction_space": active_summary.get("prediction_space"),
        "final_score_target": final_summary.get("score_target"),
        "active_prompt_style": active_summary.get("prompt_style"),
        "final_prompt_style": final_summary.get("prompt_style", "with_options"),
        "different_data_jsonl": bool(
            active_summary.get("data_jsonl")
            and final_summary.get("data_jsonl")
            and active_summary.get("data_jsonl") != final_summary.get("data_jsonl")
        ),
        "different_reader_surface": bool(
            active_summary.get("prediction_space") and final_summary.get("score_target")
        ),
    }
    metric_suspended = (
        total_active_right_final_wrong > 0
        and (reader_mismatch["different_data_jsonl"] or reader_mismatch["different_reader_surface"])
    )
    return {
        "kind": "synthetic_depth_reader_alignment",
        "active_rows": len(active_rows),
        "final_rows": len(final_rows),
        "matched_rows": len(matched_keys),
        "total_diagonal_rows": total_diagonal_rows,
        "total_active_right_final_wrong": total_active_right_final_wrong,
        "reader_mismatch": reader_mismatch,
        "final_answer_metric_suspended": metric_suspended,
        "interpretation": (
            "Active-label and final-answer rows are not scored by the same reader/surface; "
            "do not interpret their diagonal disagreement as answer-collapse failure."
            if metric_suspended
            else "No reader/surface mismatch was detected from the provided summaries."
        ),
        "by_depth": by_depth,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active_rows", required=True)
    parser.add_argument("--final_rows", required=True)
    parser.add_argument("--final_data_jsonl", required=True)
    parser.add_argument("--chain_data_jsonl")
    parser.add_argument("--active_summary")
    parser.add_argument("--final_summary")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--example_limit", type=int, default=5)
    args = parser.parse_args()

    payload = analyze(
        active_rows_path=args.active_rows,
        final_rows_path=args.final_rows,
        final_data_jsonl=args.final_data_jsonl,
        chain_data_jsonl=args.chain_data_jsonl,
        active_summary_path=args.active_summary,
        final_summary_path=args.final_summary,
        example_limit=args.example_limit,
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
