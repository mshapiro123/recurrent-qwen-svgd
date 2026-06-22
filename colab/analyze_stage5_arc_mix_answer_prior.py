"""CPU-only answer-prior diagnosis for failed Stage 5 ARC-mix gates."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from eval.analyze_mcq_regressions import (
        load_eval_data,
        paired_rows,
        read_jsonl as raw_read_jsonl,
        rows_by_id,
        summarize,
    )
except ModuleNotFoundError:  # pragma: no cover - direct ``python colab/script.py`` execution
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from eval.analyze_mcq_regressions import (
        load_eval_data,
        paired_rows,
        read_jsonl as raw_read_jsonl,
        rows_by_id,
        summarize,
    )


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    def read_text() -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            rel = path_for_cli(path)
            try:
                return subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT, text=True)
            except subprocess.CalledProcessError:
                raise

    try:
        return raw_read_jsonl(path)
    except json.JSONDecodeError:
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except OSError:
        text = read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def exactly_one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        joined = ", ".join(path.name for path in paths) or "none"
        raise FileNotFoundError(f"Expected exactly one {description}, found {len(paths)}: {joined}")
    return paths[0]


def label_paths(run_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    base = exactly_one(sorted(run_dir.glob("*_base_label.jsonl")), "base label file")
    start = exactly_one(sorted(run_dir.glob("*_start_label.jsonl")), "start label file")
    best = payload.get("best_arm", {}).get("best_checkpoint", {})
    best_arc_path = str(best.get("arc_path") or "").strip()
    if not best_arc_path:
        raise ValueError("Summary does not include best_arm.best_checkpoint.arc_path")
    return {
        "base": base,
        "start": start,
        "best": resolve_path(best_arc_path),
    }


def comparison(
    name: str,
    base_path: Path,
    candidate_path: Path,
    data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = paired_rows(rows_by_id(read_jsonl(base_path)), rows_by_id(read_jsonl(candidate_path)), data)
    return {"name": name, "summary": summarize(rows, benchmark="ARC-Easy")}


def direct_bucket_delta(summary: dict[str, Any]) -> int | None:
    bucket = (summary.get("routing_buckets") or {}).get("base_confident_direct_proxy")
    if not isinstance(bucket, dict):
        return None
    return int(bucket.get("delta", 0))


def build_diagnosis(source_summary: Path) -> dict[str, Any]:
    payload = read_json(source_summary)
    run_dir = source_summary.parent
    paths = label_paths(run_dir, payload)
    arc_config = str(payload.get("arc_eval_config") or "ARC-Easy")
    arc_limit = payload.get("arc_eval_limit")
    data = load_eval_data(None, arc_config, split="validation", seed=0, limit=int(arc_limit) if arc_limit else None)
    comparisons = [
        comparison("base_vs_start", paths["base"], paths["start"], data),
        comparison("base_vs_best", paths["base"], paths["best"], data),
        comparison("start_vs_best", paths["start"], paths["best"], data),
    ]
    by_name = {item["name"]: item["summary"] for item in comparisons}
    base_vs_best = by_name["base_vs_best"]
    status = "answer_prior_preserved"
    if int(base_vs_best.get("delta", 0)) < 0 and (direct_bucket_delta(base_vs_best) or 0) < 0:
        status = "direct_answer_prior_not_preserved"
    return {
        "kind": "stage5_arc_mix_answer_prior_diagnosis",
        "status": status,
        "source_summary": path_for_cli(source_summary),
        "run_id": payload.get("run_id"),
        "paths": {key: path_for_cli(value) for key, value in paths.items()},
        "comparisons": comparisons,
        "next_step": (
            "Do not launch another A100 SFT run from this branch. The base-confident direct bucket is still below "
            "base; revise the objective toward direct-route/base-logit preservation or a hard max_loops=1 path."
            if status == "direct_answer_prior_not_preserved"
            else "Answer prior looks preserved; inspect remaining task-specific misses before further GPU work."
        ),
    }


def compact_row(item: dict[str, Any]) -> dict[str, Any]:
    summary = item["summary"]
    drift = summary["prediction_count_deltas"]
    return {
        "name": item["name"],
        "base_correct": summary["base_correct"],
        "candidate_correct": summary["candidate_correct"],
        "delta": summary["delta"],
        "wins": summary["changes"].get("win", 0),
        "losses": summary["changes"].get("loss", 0),
        "mean_margin_delta": summary["mean_margin_delta"],
        "max_prediction_shift": drift["max_abs_candidate_minus_base"],
        "direct_bucket_delta": direct_bucket_delta(summary),
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# ARC-Mix Answer-Prior Diagnosis - {payload.get('run_id')}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Next step: {payload['next_step']}",
        "",
        "| comparison | base | candidate | delta | wins | losses | margin delta | max pred shift | direct-bucket delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["comparisons"]:
        row = compact_row(item)
        lines.append(
            "| {name} | {base}/{n} | {cand}/{n} | {delta:+d} | {wins} | {losses} | {margin:.4f} | {shift} | {direct} |".format(
                name=f"`{row['name']}`",
                base=row["base_correct"],
                cand=row["candidate_correct"],
                n=item["summary"]["paired_examples"],
                delta=row["delta"],
                wins=row["wins"],
                losses=row["losses"],
                margin=float(row["mean_margin_delta"] or 0.0),
                shift=row["max_prediction_shift"],
                direct=row["direct_bucket_delta"],
            )
        )
    lines.extend(["", "## Label Priors"])
    for item in payload["comparisons"]:
        summary = item["summary"]
        counts = summary["prediction_counts"]
        drift = summary["prediction_count_deltas"]
        lines.extend(["", f"### `{item['name']}`", "", "| label | base | candidate | answer | candidate-base |", "|---|---:|---:|---:|---:|"])
        labels = sorted(set(counts["base"]) | set(counts["candidate"]) | set(counts["answer"]))
        for label in labels:
            lines.append(
                "| `{label}` | {base} | {candidate} | {answer} | {delta:+d} |".format(
                    label=label,
                    base=counts["base"].get(label, 0),
                    candidate=counts["candidate"].get(label, 0),
                    answer=counts["answer"].get(label, 0),
                    delta=drift["candidate_minus_base"].get(label, 0),
                )
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_summary = resolve_path(args.source_summary)
    payload = build_diagnosis(source_summary)
    output_json = resolve_path(args.output_json) if args.output_json else source_summary.parent / "answer_prior_diagnosis.json"
    output_md = resolve_path(args.output_md) if args.output_md else source_summary.parent / "answer_prior_diagnosis.md"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(markdown_report(payload), encoding="utf-8")
    print(f"status={payload['status']}")
    for item in payload["comparisons"]:
        print(compact_row(item))
    print(f"wrote_json={path_for_cli(output_json)}")
    print(f"wrote_md={path_for_cli(output_md)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
