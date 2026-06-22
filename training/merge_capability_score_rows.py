"""Merge per-model score JSONLs into capability-ladder scored rows.

This is a CPU-only adapter. It does not run models and does not judge
correctness beyond the model-result files it is given. It combines a task JSONL
with one or more result JSONLs from tools such as ``eval/eval_mcq.py`` into the
``model_results`` schema consumed by ``build_capability_ladder_curriculum.py``.

Example:

    python training/merge_capability_score_rows.py \
      --tasks_jsonl data/arc_train_mcq.jsonl \
      --result qwen_0_5b=outputs/base.jsonl \
      --result qwen_1_5b=outputs/qwen15.jsonl \
      --result qwen_3b=outputs/qwen3.jsonl \
      --output_jsonl data/curriculum/scored_capability_rows.jsonl \
      --assume_decontaminated

Rows are keyed by ``id``. Optional trace/solution/completion fields in result
rows are preserved; plain MCQ likelihood outputs remain score-only unless
``--prediction_as_solution`` is set deliberately.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CORRECT_FIELDS = ("correct", "hit", "is_correct", "matched", "success", "exact")
TEXT_FIELDS = ("trace", "solution", "completion", "text", "response")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object.")
        rows.append(row)
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_result_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--result must be KEY=PATH")
    key, raw_path = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("--result model key cannot be empty")
    path = Path(raw_path.strip())
    if not path.exists():
        raise argparse.ArgumentTypeError(f"result path does not exist: {path}")
    return key, path


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("id") or row.get("task_id") or row.get("uid") or row.get("name") or index)


def answer_value(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if isinstance(answer, dict):
        return normalize_text(answer.get("value"))
    return normalize_text(answer or row.get("label") or row.get("target") or row.get("answerKey"))


def choice_text_for_answer(row: dict[str, Any], answer: str) -> str:
    choices = row.get("choices") or row.get("options")
    if isinstance(choices, dict):
        return normalize_text(choices.get(answer))
    if isinstance(choices, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if answer in labels[: len(choices)]:
            return normalize_text(choices[labels.index(answer)])
    return ""


def task_statement(row: dict[str, Any]) -> str:
    question = normalize_text(row.get("question") or row.get("prompt") or row.get("statement"))
    choices = row.get("choices") or row.get("options")
    if isinstance(choices, dict):
        rendered = "\n".join(f"{label}. {text}" for label, text in choices.items())
        return f"{question}\n{rendered}".strip()
    if isinstance(choices, list):
        rendered = "\n".join(f"{chr(ord('A') + idx)}. {text}" for idx, text in enumerate(choices))
        return f"{question}\n{rendered}".strip()
    return question


def result_correct(row: dict[str, Any]) -> bool:
    for field in CORRECT_FIELDS:
        if field in row:
            return row.get(field) is True
    prediction = normalize_text(row.get("prediction") or row.get("predicted") or row.get("answer_pred"))
    answer = normalize_text(row.get("answer") or row.get("label") or row.get("target"))
    return bool(prediction and answer and prediction == answer)


def result_text(row: dict[str, Any], *, prediction_as_solution: bool) -> str:
    for field in TEXT_FIELDS:
        text = normalize_text(row.get(field))
        if text:
            return text
    if prediction_as_solution and normalize_text(row.get("prediction")):
        return f"ANSWER: {normalize_text(row.get('prediction'))}"
    return ""


def margin_for_scores(row: dict[str, Any]) -> float | None:
    scores = row.get("scores")
    answer = normalize_text(row.get("answer") or row.get("label") or row.get("target"))
    if not isinstance(scores, dict) or not answer or answer not in scores:
        return None
    try:
        answer_score = float(scores[answer])
        distractors = [float(value) for key, value in scores.items() if key != answer]
    except (TypeError, ValueError):
        return None
    if not distractors:
        return None
    return answer_score - max(distractors)


def result_to_model_payload(
    *,
    key: str,
    row: dict[str, Any],
    result_path: Path,
    prediction_as_solution: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "correct": result_correct(row),
        "prediction": row.get("prediction") or row.get("predicted") or row.get("answer_pred"),
        "answer": row.get("answer") or row.get("label") or row.get("target"),
        "model": row.get("model") or row.get("model_name") or key,
        "mode": row.get("mode"),
        "checkpoint": row.get("checkpoint"),
        "source_file": str(result_path),
    }
    for field in ("prompt_style", "score_target", "aggregate", "num_trajectories", "scores", "trajectory_scores"):
        if field in row:
            payload[field] = row[field]
    margin = margin_for_scores(row)
    if margin is not None:
        payload["answer_margin"] = margin
    text = result_text(row, prediction_as_solution=prediction_as_solution)
    if text:
        payload["solution"] = text
    if isinstance(row.get("answer_match"), dict):
        payload["answer_match"] = row["answer_match"]
    for field in ("steps", "reasoning_steps", "depth_steps", "natural"):
        if field in row:
            payload[field] = row[field]
    return {key: value for key, value in payload.items() if value is not None}


def index_results(path: Path, *, id_field: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(read_jsonl(path)):
        rid = str(row.get(id_field) or row_id(row, idx))
        if rid in indexed:
            raise ValueError(f"Duplicate result id {rid!r} in {path}")
        indexed[rid] = row
    return indexed


def merge_rows(
    tasks: list[dict[str, Any]],
    result_specs: list[tuple[str, Path]],
    *,
    verified_by: str,
    decontaminated: bool,
    id_field: str,
    prediction_as_solution: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_indexes = {
        key: index_results(path, id_field=id_field)
        for key, path in result_specs
    }
    rows: list[dict[str, Any]] = []
    missing: Counter[str] = Counter()
    correctness: dict[str, Counter[str]] = {key: Counter() for key, _path in result_specs}
    for idx, task in enumerate(tasks):
        rid = row_id(task, idx)
        answer = answer_value(task)
        if not answer:
            missing["task_answer"] += 1
            continue
        statement = task_statement(task)
        if not statement:
            missing["task_statement"] += 1
            continue
        model_results: dict[str, dict[str, Any]] = {}
        for key, path in result_specs:
            result = result_indexes[key].get(rid)
            if result is None:
                missing[f"{key}_result"] += 1
                continue
            payload = result_to_model_payload(
                key=key,
                row=result,
                result_path=path,
                prediction_as_solution=prediction_as_solution,
            )
            model_results[key] = payload
            correctness[key]["correct" if payload.get("correct") is True else "incorrect"] += 1
        if not model_results:
            missing["no_model_results"] += 1
            continue
        row = {
            "id": rid,
            "domain": task.get("domain") or task.get("category") or task.get("dataset") or "benchmark",
            "question": task.get("question") or task.get("prompt") or task.get("statement"),
            "statement": statement,
            "choices": task.get("choices") or task.get("options"),
            "answer": {
                "value": answer,
                "choice_text": choice_text_for_answer(task, answer),
                "verified_by": [verified_by],
            },
            "decontaminated": bool(task.get("decontaminated", decontaminated)),
            "source_dataset": task.get("source_dataset") or task.get("dataset_id") or task.get("dataset") or "capability_score_merge",
            "model_results": model_results,
        }
        rows.append({key: value for key, value in row.items() if value not in (None, "")})
    report = {
        "task_rows": len(tasks),
        "output_rows": len(rows),
        "result_files": {key: str(path) for key, path in result_specs},
        "correctness": {key: dict(counter) for key, counter in correctness.items()},
        "missing": dict(sorted(missing.items())),
        "verified_by": verified_by,
        "decontaminated_default": decontaminated,
        "prediction_as_solution": prediction_as_solution,
    }
    return rows, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks_jsonl", required=True)
    parser.add_argument("--result", action="append", type=parse_result_spec, required=True, help="Model result as KEY=PATH")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json")
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--verified_by", default="benchmark_ground_truth")
    parser.add_argument("--assume_decontaminated", action="store_true")
    parser.add_argument(
        "--prediction_as_solution",
        action="store_true",
        help="Use `ANSWER: <prediction>` as a minimal solution when result rows have no trace text.",
    )
    args = parser.parse_args(argv)

    tasks = read_jsonl(args.tasks_jsonl)
    rows, report = merge_rows(
        tasks,
        args.result,
        verified_by=args.verified_by,
        decontaminated=args.assume_decontaminated,
        id_field=args.id_field,
        prediction_as_solution=args.prediction_as_solution,
    )
    write_jsonl(args.output_jsonl, rows)
    report_path = Path(args.report_json) if args.report_json else Path(args.output_jsonl).with_suffix(".report.json")
    write_json(report_path, report)
    print(f"task_rows={report['task_rows']}")
    print(f"output_rows={report['output_rows']}")
    print(f"missing={report['missing']}")
    print(f"output_jsonl={args.output_jsonl}")
    print(f"report_json={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
