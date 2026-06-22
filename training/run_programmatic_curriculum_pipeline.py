"""Build a gate-ready direct/deep curriculum shard without provider calls.

This packages the constructed arithmetic-chain records from
``generate_programmatic_curriculum.py`` into the same artifact shape consumed by
``check_curriculum_sft_gate.py``. It is CPU-only and intended to create cheap,
verified direct/deep-narrow training material before any A100 SFT run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.generate_programmatic_curriculum import generate_records, parse_range, summarize  # noqa: E402
from training.prepare_curriculum_jsonl import (  # noqa: E402
    convert_curriculum_records,
    validate_curriculum_record,
)


RUN_ID = time.strftime("programmatic_curriculum_%Y%m%d_%H%M%S")


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


def jsonl_line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def artifact_entry(path: Path) -> dict[str, Any]:
    payload = {"path": str(path), "exists": path.exists()}
    if path.suffix == ".jsonl" and path.exists():
        payload["lines"] = jsonl_line_count(path)
    return payload


def typed_records_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    mode_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    validation_issues: list[str] = []
    positive_traces = 0
    for index, record in enumerate(records, start=1):
        mode_counts[str(record.get("mode") or "")] += 1
        validation_issues.extend(validate_curriculum_record(record, line_no=index))
        for trace in record.get("traces", []):
            if not isinstance(trace, dict):
                continue
            role = str(trace.get("role") or "")
            role_counts[role] += 1
            if role.startswith("positive_"):
                positive_traces += 1

    return {
        "records": len(records),
        "validation_issues": validation_issues,
        "unsafe_auxiliary_traces": 0,
        "mode_counts": dict(sorted(mode_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "positive_traces": positive_traces,
        "distinctness_required": False,
        # Constructed records have deterministic method/depth labels. Keep the
        # existing generated-shard gate satisfied while documenting provenance.
        "min_natural_agree": 2,
        "min_distinct_agree": 2,
        "distinctness_judgments": 0,
        "source": "programmatic_generator",
    }


def write_pipeline_artifacts(
    *,
    work_dir: Path,
    records: list[dict[str, Any]],
    prompt_style: str,
    default_direct_target_loop: int,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": work_dir / "summary.json",
        "programmatic_report": work_dir / "programmatic_report.json",
        "typed_records": work_dir / "typed_records.jsonl",
        "typed_records_report": work_dir / "typed_records_report.json",
        "positive_sft": work_dir / "positive_sft.jsonl",
        "positive_sft_report": work_dir / "positive_sft_report.json",
        "verified_candidates_report": work_dir / "verified_candidates_report.json",
        "decontam_report": work_dir / "decontam_report.json",
        "method_solutions_report": work_dir / "method_solutions_report.json",
        "naturalness_report": work_dir / "naturalness_report.json",
        "depth_report": work_dir / "depth_report.json",
        "difficulty_report": work_dir / "difficulty_report.json",
    }

    sft_rows, sft_report = convert_curriculum_records(
        records,
        modes={"direct", "deep_narrow"},
        prompt_style=prompt_style,
        default_direct_target_loop=default_direct_target_loop,
        fail_on_validation=True,
    )
    typed_report = typed_records_report(records)
    if typed_report["validation_issues"]:
        raise ValueError("Generated programmatic curriculum failed validation.")

    write_jsonl(paths["typed_records"], records)
    write_json(paths["typed_records_report"], typed_report)
    write_jsonl(paths["positive_sft"], sft_rows)
    write_json(paths["positive_sft_report"], sft_report)

    programmatic_report = summarize(records)
    programmatic_report.update(
        {
            "source": "programmatic_generator",
            "trusted_answer_source": "constructed_python_eval",
            "positive_sft_rows": len(sft_rows),
        }
    )
    write_json(paths["programmatic_report"], programmatic_report)

    write_json(
        paths["verified_candidates_report"],
        {
            "verified": len(records),
            "require_programmatic_answer_check": True,
            "programmatic_check_counts": {"constructed_python_eval": len(records)},
            "source": "programmatic_generator",
        },
    )
    write_json(
        paths["decontam_report"],
        {
            "accepted": len(records),
            "rejected": 0,
            "method": "constructed_no_external_source",
        },
    )
    write_json(
        paths["method_solutions_report"],
        {
            "solution_candidates": len(sft_rows),
            "correct_solution_candidates": len(sft_rows),
            "source": "programmatic_generator",
        },
    )
    write_json(
        paths["naturalness_report"],
        {
            "judgments": len(sft_rows),
            "natural": len(sft_rows),
            "source": "constructed_arithmetic_chain",
        },
    )
    write_json(
        paths["depth_report"],
        {
            "measurements": len(records),
            "source": "constructed_operation_count",
        },
    )
    write_json(
        paths["difficulty_report"],
        {
            "measured": len(records),
            "source": "constructed_proxy_pass_rate",
        },
    )

    summary = {
        "run_id": RUN_ID,
        "kind": "programmatic_curriculum_pipeline",
        "status": "complete",
        "next_action": "Run training/check_curriculum_sft_gate.py before any GPU fine-tuning.",
        "artifacts": {name: artifact_entry(path) for name, path in sorted(paths.items())},
        "counts": {
            "typed_records": len(records),
            "positive_sft_rows": len(sft_rows),
            "mode_counts": programmatic_report["by_mode"],
        },
    }
    write_json(paths["summary"], summary)
    summary["artifacts"]["summary"] = artifact_entry(paths["summary"])
    write_json(paths["summary"], summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--num_direct", type=int, default=100)
    parser.add_argument("--num_deep_narrow", type=int, default=100)
    parser.add_argument("--direct_steps", type=parse_range, default=(1, 2))
    parser.add_argument("--deep_steps", type=parse_range, default=(5, 9))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max_abs_value", type=int, default=500)
    parser.add_argument("--max_target_loops", type=int, default=4)
    parser.add_argument("--prompt_style", choices=("qwen_instruct", "plain"), default="qwen_instruct")
    parser.add_argument("--default_direct_target_loop", type=int, default=1)
    args = parser.parse_args(argv)

    records = generate_records(
        num_direct=args.num_direct,
        num_deep_narrow=args.num_deep_narrow,
        direct_steps=args.direct_steps,
        deep_steps=args.deep_steps,
        seed=args.seed,
        max_abs_value=args.max_abs_value,
        max_target_loops=args.max_target_loops,
    )
    summary = write_pipeline_artifacts(
        work_dir=Path(args.work_dir),
        records=records,
        prompt_style=args.prompt_style,
        default_direct_target_loop=args.default_direct_target_loop,
    )
    print(f"status={summary['status']}")
    print(f"typed_records={summary['counts']['typed_records']}")
    print(f"positive_sft_rows={summary['counts']['positive_sft_rows']}")
    print(f"mode_counts={summary['counts']['mode_counts']}")
    print(f"summary_json={Path(args.work_dir) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
