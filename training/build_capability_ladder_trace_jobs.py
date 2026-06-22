"""Build strong-model trace jobs from capability-ladder scored rows.

This is the CPU-only bridge between the Qwen-scale capability probe and real
reasoning-trace SFT data. The probe can discover rows where larger Qwen models
solve examples the 0.5B model misses, but those rows are answer-only MCQ
artifacts. This script turns the verified rows into provider-neutral jobs for a
non-student strong model to produce loop-targeted traces.
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

from training.build_capability_ladder_curriculum import (  # noqa: E402
    answer_payload,
    answer_value,
    has_trusted_answer,
    model_correct,
    read_jsonl,
    tier_for_row,
)
from training.build_curriculum_generation_jobs import (  # noqa: E402
    make_job,
    validate_external_models,
    write_jsonl,
)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if (ROOT / path).exists():
        return ROOT / path
    if base is not None and (base / path).exists():
        return base / path
    return ROOT / path


def scored_jsonl_from_summary(summary_json: Path) -> Path:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{summary_json} is not a JSON object.")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    scored = str(artifacts.get("scored_capability_rows") or "").strip()
    if not scored:
        raise ValueError(f"{summary_json} does not contain artifacts.scored_capability_rows.")
    path = resolve_path(scored, base=summary_json.parent)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def answer_label_and_text(row: dict[str, Any]) -> tuple[str, str]:
    payload = answer_payload(row) or {}
    value = str(payload.get("value") or answer_value(row)).strip()
    choice_text = str(payload.get("choice_text") or "").strip()
    return value, choice_text


def prompt_capability_ladder_trace(
    *,
    statement: str,
    answer: str,
    answer_choice_text: str,
    target_loop_count: int,
    capability_tier: str,
) -> str:
    answer_line = f'ANSWER: {answer}'
    if answer_choice_text:
        answer_hint = f"{answer} ({answer_choice_text})"
    else:
        answer_hint = answer

    if target_loop_count <= 1:
        depth_instruction = (
            "Write a compact preservation trace. Use the shortest reliable reasoning "
            "needed to get the answer; do not add unnecessary multi-step deliberation."
        )
        loop_plan = "Target recurrent depth: 1 pass."
    elif target_loop_count == 2:
        depth_instruction = (
            "Write a moderately decomposed trace suitable for two latent reasoning "
            "passes: first identify the relevant facts and constraint, then solve and verify."
        )
        loop_plan = "Target recurrent depth: 2 passes."
    else:
        depth_instruction = (
            "Write a careful multi-step trace suitable for deeper recurrence: separate "
            "problem parsing, constraint/useful-fact extraction, solving, and verification."
        )
        loop_plan = f"Target recurrent depth: {target_loop_count} passes."

    return (
        "Create a verified training trace for a recurrent-depth reasoning model.\n"
        f"{loop_plan}\n"
        f"Capability tier: {capability_tier}.\n"
        f"{depth_instruction}\n\n"
        "Rules:\n"
        "- Use only the information in the problem.\n"
        "- Preserve the verified final answer exactly; do not choose a different option.\n"
        "- If this is multiple choice, explain why the selected option is supported and avoid label-position shortcuts.\n"
        "- End with the final answer on its own line, exactly as shown below.\n\n"
        f"Problem:\n{statement}\n\n"
        f"Verified final answer: {answer_hint}\n"
        f"Final line to use exactly:\n{answer_line}"
    )


def candidate_rows(
    rows: list[dict[str, Any]],
    *,
    base_key: str,
    mid_key: str,
    high_keys: list[str],
    high_target_loop: int,
    assume_decontaminated: bool,
    include_direct: bool,
    include_deep: bool,
    max_rows: int | None,
    max_per_tier: int | None,
) -> tuple[list[tuple[int, dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    selected: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    skipped: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        tier = tier_for_row(
            row,
            base_key=base_key,
            mid_key=mid_key,
            high_keys=high_keys,
            high_target_loop=high_target_loop,
        )
        if tier is None:
            skipped["unresolved_capability"] += 1
            continue
        target_loop = int(tier["target_loop_count"])
        if target_loop <= 1 and not include_direct:
            skipped["direct_excluded"] += 1
            continue
        if target_loop > 1 and not include_deep:
            skipped["deep_excluded"] += 1
            continue
        if not has_trusted_answer(row):
            skipped["untrusted_answer"] += 1
            continue
        if row.get("decontaminated") is not True and not assume_decontaminated:
            skipped["not_decontaminated"] += 1
            continue
        statement = str(row.get("statement") or row.get("prompt") or row.get("question") or "").strip()
        if not statement:
            skipped["missing_statement"] += 1
            continue
        tier_name = str(tier["tier"])
        if max_per_tier is not None and tier_counts[tier_name] >= max_per_tier:
            skipped["max_per_tier"] += 1
            continue
        selected.append((row_index, row, tier))
        tier_counts[tier_name] += 1
        if max_rows is not None and len(selected) >= max_rows:
            break
    report = {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "tier_counts": dict(sorted(tier_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "base_key": base_key,
        "mid_key": mid_key,
        "high_keys": high_keys,
        "high_target_loop": high_target_loop,
        "model_correct_counts": {
            base_key: sum(1 for row in rows if model_correct(row, base_key)),
            mid_key: sum(1 for row in rows if model_correct(row, mid_key)),
            **{key: sum(1 for row in rows if model_correct(row, key)) for key in high_keys},
        },
    }
    return selected, report


def build_trace_jobs(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    base_key: str,
    mid_key: str,
    high_keys: list[str],
    high_target_loop: int,
    assume_decontaminated: bool = False,
    include_direct: bool = True,
    include_deep: bool = True,
    max_rows: int | None = None,
    max_per_tier: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, report = candidate_rows(
        rows,
        base_key=base_key,
        mid_key=mid_key,
        high_keys=high_keys,
        high_target_loop=high_target_loop,
        assume_decontaminated=assume_decontaminated,
        include_direct=include_direct,
        include_deep=include_deep,
        max_rows=max_rows,
        max_per_tier=max_per_tier,
    )
    jobs: list[dict[str, Any]] = []
    for row_index, row, tier in selected:
        statement = str(row.get("statement") or row.get("prompt") or row.get("question") or "").strip()
        answer, choice_text = answer_label_and_text(row)
        target_loop = int(tier["target_loop_count"])
        for model in models:
            jobs.append(
                make_job(
                    index=len(jobs),
                    stage="capability_ladder_trace_solve",
                    role="capability_trace_solver",
                    model=model,
                    prompt=prompt_capability_ladder_trace(
                        statement=statement,
                        answer=answer,
                        answer_choice_text=choice_text,
                        target_loop_count=target_loop,
                        capability_tier=str(tier["tier"]),
                    ),
                    expects_json=False,
                    metadata={
                        "record_id": str(row.get("id") or row.get("task_id") or row.get("uid") or f"row-{row_index:06d}"),
                        "row_index": row_index,
                        "domain": row.get("domain") or row.get("category") or "capability_ladder",
                        "target_loop_count": target_loop,
                        "capability_tier": tier["tier"],
                        "mode": tier["mode"],
                        "role": tier["role"],
                        "solver_key": tier["solver_key"],
                        "verified_answer": answer,
                        "verified_answer_choice_text": choice_text,
                        "base_key": base_key,
                        "mid_key": mid_key,
                        "high_keys": high_keys,
                    },
                )
            )
    report = {
        "kind": "capability_ladder_trace_jobs",
        "status": "ready" if jobs else "empty",
        **report,
        "jobs": len(jobs),
        "by_model": dict(sorted(Counter(str(job["model"]) for job in jobs).items())),
        "by_target_loop": dict(sorted(Counter(str(job["metadata"]["target_loop_count"]) for job in jobs).items())),
    }
    return jobs, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scored_jsonl")
    source.add_argument("--summary_json")
    parser.add_argument("--models", required=True, help="Comma-separated non-student strong model ids.")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json")
    parser.add_argument("--base_key", default="qwen_0_5b")
    parser.add_argument("--mid_key", default="qwen_1_5b")
    parser.add_argument("--high_keys", default="qwen_3b,strong_solver")
    parser.add_argument("--high_target_loop", type=int, default=3)
    parser.add_argument("--assume_decontaminated", action="store_true")
    parser.add_argument("--include_direct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_deep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--max_per_tier", type=int)
    parser.add_argument("--allow_student_lineage", action="store_true")
    args = parser.parse_args(argv)

    models = parse_csv(args.models)
    if not models:
        raise ValueError("--models is required")
    validate_external_models(models, allow_student_lineage=args.allow_student_lineage)
    scored_jsonl = resolve_path(args.scored_jsonl) if args.scored_jsonl else scored_jsonl_from_summary(resolve_path(args.summary_json))
    rows = read_jsonl(scored_jsonl)
    jobs, report = build_trace_jobs(
        rows,
        models=models,
        base_key=args.base_key,
        mid_key=args.mid_key,
        high_keys=parse_csv(args.high_keys),
        high_target_loop=args.high_target_loop,
        assume_decontaminated=args.assume_decontaminated,
        include_direct=args.include_direct,
        include_deep=args.include_deep,
        max_rows=args.max_rows,
        max_per_tier=args.max_per_tier,
    )
    report["source"] = {"scored_jsonl": scored_jsonl.as_posix(), "summary_json": str(args.summary_json or "")}
    report["artifacts"] = {"jobs_jsonl": str(args.output_jsonl), "report_json": str(args.report_json or "")}
    write_jsonl(args.output_jsonl, jobs)
    if args.report_json:
        write_json(args.report_json, report)
    print(f"status={report['status']}")
    print(f"selected_rows={report['selected_rows']}")
    print(f"jobs={report['jobs']}")
    print(f"tier_counts={report['tier_counts']}")
    print(f"by_target_loop={report['by_target_loop']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
