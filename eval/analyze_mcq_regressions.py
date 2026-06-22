"""Analyze paired MCQ wins and regressions between two label eval outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.prepare_arc_mcq import row_to_mcq


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def load_arc_data(config: str, split: str, seed: int, limit: int | None) -> dict[str, dict[str, Any]]:
    dataset = load_dataset("allenai/ai2_arc", config, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return {
        str(row.get("id") or idx): row_to_mcq(dict(row), idx, seed=seed, shuffle_choices=True)
        for idx, row in enumerate(dataset)
    }


def load_eval_data(path: Path | None, arc_config: str | None, split: str, seed: int, limit: int | None) -> dict[str, dict[str, Any]]:
    if path is not None:
        return rows_by_id(read_jsonl(path))
    if arc_config is None:
        return {}
    return load_arc_data(arc_config, split=split, seed=seed, limit=limit)


def score_margin(row: dict[str, Any]) -> float | None:
    scores = row.get("scores") or {}
    answer = row.get("answer")
    if answer not in scores:
        return None
    other_scores = [float(score) for label, score in scores.items() if label != answer]
    if not other_scores:
        return None
    return float(scores[answer]) - max(other_scores)


def correct_score(row: dict[str, Any]) -> float | None:
    scores = row.get("scores") or {}
    answer = row.get("answer")
    if answer not in scores:
        return None
    return float(scores[answer])


def text_features(question: str) -> dict[str, bool]:
    q = question.casefold()
    return {
        "has_number": bool(re.search(r"\d", question)),
        "has_math_word": any(word in q for word in ["sum", "difference", "product", "total", "average", "rate", "speed", "distance", "mass", "volume"]),
        "has_negation": any(word in q for word in [" not ", " except ", " least ", " never "]),
        "asks_why": "why" in q,
        "asks_which": "which" in q,
        "asks_best": any(word in q for word in ["best", "most likely", "main reason", "primary"]),
        "diagram_like": any(word in q for word in ["diagram", "picture", "graph", "table", "model", "shown"]),
    }


def paired_rows(
    base: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row_id in sorted(set(base) & set(candidate)):
        base_row = base[row_id]
        cand_row = candidate[row_id]
        data_row = data.get(row_id, {})
        question = str(data_row.get("question") or "")
        base_hit = bool(base_row.get("hit"))
        cand_hit = bool(cand_row.get("hit"))
        if cand_hit and not base_hit:
            change = "win"
        elif base_hit and not cand_hit:
            change = "loss"
        elif base_hit and cand_hit:
            change = "tie_correct"
        else:
            change = "tie_wrong"
        base_margin = score_margin(base_row)
        cand_margin = score_margin(cand_row)
        base_correct_score = correct_score(base_row)
        cand_correct_score = correct_score(cand_row)
        rows.append(
            {
                "id": row_id,
                "change": change,
                "answer": base_row.get("answer"),
                "base_prediction": base_row.get("prediction"),
                "candidate_prediction": cand_row.get("prediction"),
                "base_hit": base_hit,
                "candidate_hit": cand_hit,
                "base_margin": base_margin,
                "candidate_margin": cand_margin,
                "margin_delta": None if base_margin is None or cand_margin is None else cand_margin - base_margin,
                "correct_score_delta": None
                if base_correct_score is None or cand_correct_score is None
                else cand_correct_score - base_correct_score,
                "question": question,
                "question_len": len(question.split()),
                "features": text_features(question),
            }
        )
    return rows


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def summarize(rows: list[dict[str, Any]], benchmark: str) -> dict[str, Any]:
    counts = Counter(row["change"] for row in rows)
    base_correct = counts["loss"] + counts["tie_correct"]
    candidate_correct = counts["win"] + counts["tie_correct"]
    pred_counts = {
        "base": dict(Counter(str(row["base_prediction"]) for row in rows)),
        "candidate": dict(Counter(str(row["candidate_prediction"]) for row in rows)),
        "answer": dict(Counter(str(row["answer"]) for row in rows)),
    }
    by_feature = {}
    for feature in sorted(next(iter(rows), {"features": {}})["features"]):
        yes = [row for row in rows if row["features"].get(feature)]
        no = [row for row in rows if not row["features"].get(feature)]
        by_feature[feature] = {
            "yes": feature_summary(yes),
            "no": feature_summary(no),
        }
    return {
        "benchmark": benchmark,
        "paired_examples": len(rows),
        "base_correct": base_correct,
        "candidate_correct": candidate_correct,
        "delta": candidate_correct - base_correct,
        "changes": dict(counts),
        "mean_margin_delta": mean([row["margin_delta"] for row in rows]),
        "mean_correct_score_delta": mean([row["correct_score_delta"] for row in rows]),
        "mean_question_len": mean([float(row["question_len"]) for row in rows]),
        "prediction_counts": pred_counts,
        "features": by_feature,
        "loss_examples": example_rows([row for row in rows if row["change"] == "loss"], reverse=False),
        "win_examples": example_rows([row for row in rows if row["change"] == "win"], reverse=True),
    }


def feature_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["change"] for row in rows)
    base_correct = counts["loss"] + counts["tie_correct"]
    candidate_correct = counts["win"] + counts["tie_correct"]
    return {
        "n": len(rows),
        "delta": candidate_correct - base_correct,
        "wins": counts["win"],
        "losses": counts["loss"],
        "base_correct": base_correct,
        "candidate_correct": candidate_correct,
    }


def example_rows(rows: list[dict[str, Any]], reverse: bool) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> float:
        value = row.get("margin_delta")
        return float("-inf") if value is None else float(value)

    selected = sorted(rows, key=key, reverse=reverse)[:8]
    return [
        {
            "id": row["id"],
            "answer": row["answer"],
            "base_prediction": row["base_prediction"],
            "candidate_prediction": row["candidate_prediction"],
            "base_margin": row["base_margin"],
            "candidate_margin": row["candidate_margin"],
            "margin_delta": row["margin_delta"],
            "question": row["question"][:280],
        }
        for row in selected
    ]


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# MCQ Regression Diagnosis - {payload['run_id']}",
        "",
        f"- Candidate: `{payload['candidate_name']}`",
        f"- Benchmarks: `{', '.join(payload['benchmarks'])}`",
        "",
        "| Benchmark | Base | Candidate | Delta | W/L/Tie-correct/Tie-wrong | Mean margin delta |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for summary in payload["summaries"]:
        changes = summary["changes"]
        lines.append(
            "| {benchmark} | {base}/{n} | {cand}/{n} | {delta:+d} | {win}/{loss}/{tc}/{tw} | {margin} |".format(
                benchmark=summary["benchmark"],
                base=summary["base_correct"],
                cand=summary["candidate_correct"],
                n=summary["paired_examples"],
                delta=summary["delta"],
                win=changes.get("win", 0),
                loss=changes.get("loss", 0),
                tc=changes.get("tie_correct", 0),
                tw=changes.get("tie_wrong", 0),
                margin=format_float(summary["mean_margin_delta"]),
            )
        )
    lines.extend(["", "## Feature Buckets"])
    for summary in payload["summaries"]:
        lines.extend(["", f"### {summary['benchmark']}", "", "| Feature | n yes | delta yes | n no | delta no |", "|---|---:|---:|---:|---:|"])
        for feature, values in summary["features"].items():
            lines.append(
                f"| `{feature}` | {values['yes']['n']} | {values['yes']['delta']:+d} | {values['no']['n']} | {values['no']['delta']:+d} |"
            )
    lines.extend(["", "## Largest Regressions"])
    for summary in payload["summaries"]:
        lines.extend(["", f"### {summary['benchmark']}"])
        for row in summary["loss_examples"]:
            lines.append(
                f"- `{row['id']}` answer `{row['answer']}`, base `{row['base_prediction']}`, recurrent `{row['candidate_prediction']}`, "
                f"margin delta {format_float(row['margin_delta'])}: {row['question']}"
            )
    lines.extend(["", "## Largest Wins"])
    for summary in payload["summaries"]:
        lines.extend(["", f"### {summary['benchmark']}"])
        for row in summary["win_examples"]:
            lines.append(
                f"- `{row['id']}` answer `{row['answer']}`, base `{row['base_prediction']}`, recurrent `{row['candidate_prediction']}`, "
                f"margin delta {format_float(row['margin_delta'])}: {row['question']}"
            )
    lines.append("")
    return "\n".join(lines)


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--candidate_name", default="recurrent")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--benchmark",
        action="append",
        nargs=4,
        metavar=("NAME", "ARC_CONFIG", "BASE_JSONL", "CANDIDATE_JSONL"),
        required=True,
        help="Benchmark spec. Example: arc_easy ARC-Easy base.jsonl recurrent.jsonl",
    )
    args = parser.parse_args()

    summaries = []
    for name, arc_config, base_jsonl, candidate_jsonl in args.benchmark:
        data = load_eval_data(None, arc_config, split=args.split, seed=args.seed, limit=args.limit)
        rows = paired_rows(rows_by_id(read_jsonl(Path(base_jsonl))), rows_by_id(read_jsonl(Path(candidate_jsonl))), data)
        summaries.append(summarize(rows, benchmark=name))

    payload = {
        "run_id": args.run_id,
        "candidate_name": args.candidate_name,
        "benchmarks": [summary["benchmark"] for summary in summaries],
        "summaries": summaries,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(markdown_report(payload), encoding="utf-8")
    print(f"wrote_json={output_json}")
    print(f"wrote_md={output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
