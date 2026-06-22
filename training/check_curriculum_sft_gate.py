"""No-GPU gate for generated curriculum shards before recurrent SFT.

The strong-model curriculum pipeline writes several reports before producing
``positive_sft.jsonl``. This checker turns those reports into one explicit
go/no-go artifact so paid GPU training starts only after answer verification,
decontamination, role safety, and SFT export checks pass.
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

from training.prepare_curriculum_jsonl import validate_curriculum_record  # noqa: E402


RUN_ID = time.strftime("curriculum_sft_gate_%Y%m%d_%H%M%S")


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_no} in {path} is not a JSON object")
        rows.append(row)
    return rows


def line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def summary_path_from_args(args: argparse.Namespace) -> Path:
    if args.summary_json:
        return resolve_path(args.summary_json)
    return resolve_path(args.work_dir) / "summary.json"


def artifact_path(summary: dict[str, Any], name: str) -> Path | None:
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    item = artifacts.get(name)
    if not isinstance(item, dict) or not item.get("path"):
        return None
    return resolve_path(str(item["path"]))


def role_starts_positive(role: Any) -> bool:
    return str(role or "").startswith("positive_")


def add_issue(issues: list[dict[str, Any]], code: str, message: str, *, severity: str = "blocker") -> None:
    issues.append({"severity": severity, "code": code, "message": message})


def parse_min_mode_rows(values: list[str] | None) -> dict[str, int]:
    requirements: dict[str, int] = {}
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                mode, count = item.split("=", 1)
            elif ":" in item:
                mode, count = item.split(":", 1)
            else:
                raise argparse.ArgumentTypeError(
                    f"Invalid --min_mode_rows item {item!r}; expected mode=count."
                )
            mode = mode.strip()
            if mode not in {"direct", "deep_narrow", "wide", "both"}:
                raise argparse.ArgumentTypeError(f"Invalid curriculum mode in --min_mode_rows: {mode!r}")
            try:
                parsed_count = int(count)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid row count in --min_mode_rows item {item!r}; expected integer."
                ) from exc
            if parsed_count < 0:
                raise argparse.ArgumentTypeError(
                    f"Invalid row count in --min_mode_rows item {item!r}; expected non-negative integer."
                )
            requirements[mode] = parsed_count
    return requirements


def check_required_artifacts(
    summary: dict[str, Any],
    required: list[str],
    issues: list[dict[str, Any]],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in required:
        path = artifact_path(summary, name)
        if path is None:
            add_issue(issues, "missing_artifact_path", f"Summary has no artifact path for {name}.")
            continue
        paths[name] = path
        if not path.exists():
            add_issue(issues, "missing_artifact_file", f"Required artifact is missing: {name} -> {path}")
    return paths


def check_positive_sft_rows(
    rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    *,
    max_loop_target: int,
) -> dict[str, Any]:
    role_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    source_model_counts: Counter[str] = Counter()
    bad_rows = 0
    for index, row in enumerate(rows):
        role = str(row.get("trace_role") or "")
        mode = str(row.get("curriculum_mode") or row.get("routing_type") or "")
        source_model = str(row.get("source_model") or "").strip()
        role_counts[role] += 1
        mode_counts[mode] += 1
        if source_model:
            source_model_counts[source_model] += 1
        else:
            bad_rows += 1
            add_issue(
                issues,
                "missing_sft_source_model",
                f"positive_sft row {index} is missing source_model provenance.",
            )
        if not role_starts_positive(role):
            bad_rows += 1
            add_issue(issues, "non_positive_sft_row", f"positive_sft row {index} has non-positive trace_role={role!r}.")
        if not str(row.get("prompt") or "").strip():
            bad_rows += 1
            add_issue(issues, "missing_sft_prompt", f"positive_sft row {index} is missing prompt.")
        if not str(row.get("completion") or "").strip():
            bad_rows += 1
            add_issue(issues, "missing_sft_completion", f"positive_sft row {index} is missing completion.")
        if "target_loop_count" in row:
            target = row["target_loop_count"]
            if isinstance(target, bool) or not isinstance(target, int) or not (1 <= target <= max_loop_target):
                bad_rows += 1
                add_issue(
                    issues,
                    "invalid_target_loop_count",
                    f"positive_sft row {index} has invalid target_loop_count={target!r}.",
                )
    return {
        "rows": len(rows),
        "bad_rows": bad_rows,
        "role_counts": dict(sorted(role_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "source_model_counts": dict(sorted(source_model_counts.items())),
    }


def check_mode_row_requirements(
    mode_counts: dict[str, int],
    min_mode_rows: dict[str, int],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mode, required in sorted(min_mode_rows.items()):
        observed = int(mode_counts.get(mode, 0))
        passed = observed >= required
        results[mode] = {"required": required, "observed": observed, "passed": passed}
        if not passed:
            add_issue(
                issues,
                "too_few_mode_rows",
                f"positive_sft has {observed} {mode!r} rows < required {required}.",
            )
    return results


def check_typed_records(rows: list[dict[str, Any]], issues: list[dict[str, Any]]) -> dict[str, Any]:
    mode_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    invalid = 0
    positive_missing_answer_match = 0
    for index, row in enumerate(rows, start=1):
        mode_counts[str(row.get("mode") or "")] += 1
        for trace in row.get("traces") if isinstance(row.get("traces"), list) else []:
            if isinstance(trace, dict):
                role_counts[str(trace.get("role") or "")] += 1
                if role_starts_positive(trace.get("role")):
                    answer_match = trace.get("answer_match")
                    if not isinstance(answer_match, dict) or answer_match.get("matched") is not True:
                        positive_missing_answer_match += 1
                        add_issue(
                            issues,
                            "positive_trace_missing_answer_match",
                            (
                                f"typed record {index} has positive trace role={trace.get('role')!r} "
                                "without answer_match.matched=true."
                            ),
                        )
                    elif not str(answer_match.get("verified_answer_normalized") or "").strip():
                        positive_missing_answer_match += 1
                        add_issue(
                            issues,
                            "positive_trace_missing_verified_answer",
                            (
                                f"typed record {index} has positive trace role={trace.get('role')!r} "
                                "without answer_match.verified_answer_normalized."
                            ),
                        )
        row_issues = validate_curriculum_record(row, line_no=index)
        if row_issues:
            invalid += 1
            for issue in row_issues[:10]:
                add_issue(issues, "typed_record_validation", issue)
    return {
        "rows": len(rows),
        "invalid_rows": invalid,
        "positive_missing_answer_match": positive_missing_answer_match,
        "mode_counts": dict(sorted(mode_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
    }


def check_report_payloads(
    *,
    reports: dict[str, dict[str, Any]],
    min_positive_rows: int,
    require_programmatic_answer_check: bool,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    positive_report = reports["positive_sft_report"]
    typed_report = reports["typed_records_report"]
    verified_report = reports["verified_candidates_report"]
    decontam_report = reports["decontam_report"]
    method_report = reports["method_solutions_report"]
    naturalness_report = reports["naturalness_report"]
    depth_report = reports["depth_report"]
    difficulty_report = reports["difficulty_report"]

    if int(positive_report.get("exported_examples") or 0) < min_positive_rows:
        add_issue(
            issues,
            "too_few_positive_rows",
            f"positive_sft_report exported_examples={positive_report.get('exported_examples')} < {min_positive_rows}.",
        )
    if int(positive_report.get("invalid_records") or 0) != 0:
        add_issue(issues, "invalid_positive_sft_records", "positive_sft_report has invalid_records != 0.")
    if positive_report.get("issues"):
        add_issue(issues, "positive_sft_issues", "positive_sft_report contains validation issues.")
    exported_roles = positive_report.get("exported_role_counts")
    if not isinstance(exported_roles, dict) or not exported_roles:
        add_issue(issues, "missing_exported_positive_roles", "positive_sft_report has no exported positive roles.")
    else:
        for role in exported_roles:
            if not role_starts_positive(role):
                add_issue(issues, "exported_non_positive_role", f"Exported role is non-positive: {role!r}.")

    if int(typed_report.get("records") or 0) <= 0:
        add_issue(issues, "no_typed_records", "typed_records_report has no typed records.")
    if typed_report.get("validation_issues"):
        add_issue(issues, "typed_record_report_validation_issues", "typed_records_report contains validation issues.")
    if int(typed_report.get("unsafe_auxiliary_traces") or 0) != 0:
        add_issue(issues, "unsafe_auxiliary_traces", "typed_records_report contains unsafe auxiliary traces.")
    min_natural_agree = typed_report.get("min_natural_agree")
    if min_natural_agree is None or int(min_natural_agree or 0) < 2:
        add_issue(
            issues,
            "naturalness_agreement_too_low",
            (
                "typed_records_report min_natural_agree must be at least 2 for generated curriculum SFT; "
                f"observed {min_natural_agree!r}."
            ),
        )
    typed_mode_counts = typed_report.get("mode_counts")
    typed_modes = set(typed_mode_counts) if isinstance(typed_mode_counts, dict) else set()
    exported_wide_roles = (
        isinstance(exported_roles, dict)
        and any(str(role) == "positive_wide" for role in exported_roles)
    )
    wide_or_both_rows = bool({"wide", "both"} & typed_modes) or exported_wide_roles
    if wide_or_both_rows and typed_report.get("distinctness_required") is not True:
        add_issue(
            issues,
            "wide_rows_without_distinctness_gate",
            "wide/both generated curriculum rows require method-distinctness judgments before SFT.",
        )
    if wide_or_both_rows and int(typed_report.get("distinctness_judgments") or 0) <= 0:
        add_issue(
            issues,
            "wide_rows_without_distinctness_judgments",
            "wide/both generated curriculum rows require at least one collected distinctness judgment.",
        )
    min_distinct_agree = typed_report.get("min_distinct_agree")
    if typed_report.get("distinctness_required") is True and (
        min_distinct_agree is None or int(min_distinct_agree or 0) < 2
    ):
        add_issue(
            issues,
            "distinctness_agreement_too_low",
            (
                "typed_records_report min_distinct_agree must be at least 2 when distinctness is required; "
                f"observed {min_distinct_agree!r}."
            ),
        )

    if int(verified_report.get("verified") or 0) <= 0:
        add_issue(issues, "no_verified_candidates", "verified_candidates_report has no verified candidates.")
    if require_programmatic_answer_check and verified_report.get("require_programmatic_answer_check") is not True:
        add_issue(
            issues,
            "programmatic_check_not_required",
            "verified_candidates_report did not require programmatic answer checks.",
        )

    if int(decontam_report.get("accepted") or 0) <= 0:
        add_issue(issues, "no_decontaminated_candidates", "decontam_report has no accepted candidates.")
    if int(method_report.get("solution_candidates") or 0) <= 0:
        add_issue(issues, "no_method_solutions", "method_solutions_report has no correct method solutions.")
    if int(naturalness_report.get("judgments") or 0) <= 0:
        add_issue(issues, "no_naturalness_judgments", "naturalness_report has no judgments.")
    if int(depth_report.get("measurements") or 0) <= 0:
        add_issue(issues, "no_depth_measurements", "depth_report has no measurements.")
    if int(difficulty_report.get("measured") or 0) <= 0:
        add_issue(issues, "no_difficulty_measurements", "difficulty_report has no measured candidates.")

    return {
        "exported_examples": positive_report.get("exported_examples"),
        "typed_records": typed_report.get("records"),
        "verified": verified_report.get("verified"),
        "decontaminated": decontam_report.get("accepted"),
        "method_solutions": method_report.get("solution_candidates"),
        "naturalness_judgments": naturalness_report.get("judgments"),
        "min_natural_agree": typed_report.get("min_natural_agree"),
        "min_distinct_agree": typed_report.get("min_distinct_agree"),
        "distinctness_required": typed_report.get("distinctness_required"),
        "depth_measurements": depth_report.get("measurements"),
        "difficulty_measured": difficulty_report.get("measured"),
    }


def build_gate_payload(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = summary_path_from_args(args)
    issues: list[dict[str, Any]] = []
    if not summary_path.exists():
        add_issue(issues, "missing_summary", f"Summary JSON is missing: {summary_path}")
        return {
            "run_id": RUN_ID,
            "kind": "curriculum_sft_gate",
            "summary_json": str(summary_path),
            "go": False,
            "status": "missing_summary",
            "issues": issues,
        }

    summary = read_json(summary_path)
    if summary.get("status") != "complete":
        add_issue(issues, "pipeline_not_complete", f"Pipeline status is {summary.get('status')!r}, not 'complete'.")

    required = [
        "summary",
        "typed_records",
        "typed_records_report",
        "positive_sft",
        "positive_sft_report",
        "verified_candidates_report",
        "decontam_report",
        "method_solutions_report",
        "naturalness_report",
        "depth_report",
        "difficulty_report",
    ]
    paths = check_required_artifacts(summary, required, issues)

    reports: dict[str, dict[str, Any]] = {}
    for name in [
        "typed_records_report",
        "positive_sft_report",
        "verified_candidates_report",
        "decontam_report",
        "method_solutions_report",
        "naturalness_report",
        "depth_report",
        "difficulty_report",
    ]:
        path = paths.get(name)
        if path and path.exists():
            reports[name] = read_json(path)

    report_summary: dict[str, Any] = {}
    if len(reports) == 8:
        report_summary = check_report_payloads(
            reports=reports,
            min_positive_rows=args.min_positive_rows,
            require_programmatic_answer_check=args.require_programmatic_answer_check,
            issues=issues,
        )

    typed_summary: dict[str, Any] = {}
    typed_path = paths.get("typed_records")
    if typed_path and typed_path.exists():
        typed_summary = check_typed_records(read_jsonl(typed_path), issues)

    sft_summary: dict[str, Any] = {}
    sft_path = paths.get("positive_sft")
    if sft_path and sft_path.exists():
        rows = read_jsonl(sft_path)
        if len(rows) < args.min_positive_rows:
            add_issue(
                issues,
                "too_few_positive_sft_lines",
                f"positive_sft has {len(rows)} rows < {args.min_positive_rows}.",
            )
        sft_summary = check_positive_sft_rows(rows, issues, max_loop_target=args.max_loop_target)
        sft_summary["mode_requirements"] = check_mode_row_requirements(
            sft_summary.get("mode_counts", {}),
            args.min_mode_rows,
            issues,
        )

    go = not any(issue.get("severity") == "blocker" for issue in issues)
    return {
        "run_id": RUN_ID,
        "kind": "curriculum_sft_gate",
        "summary_json": str(summary_path),
        "work_dir": str(resolve_path(args.work_dir)) if args.work_dir else None,
        "go": go,
        "status": "go_train_recurrent_sft" if go else "no_go",
        "issues": issues,
        "checks": {
            "reports": report_summary,
            "typed_records": typed_summary,
            "positive_sft": sft_summary,
        },
        "artifacts": {name: str(path) for name, path in sorted(paths.items())},
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Curriculum SFT Gate - {payload['run_id']}",
        "",
        f"- Summary: `{payload['summary_json']}`",
        f"- Go: `{payload['go']}`",
        f"- Status: `{payload['status']}`",
        "",
        "## Report Checks",
        "",
        "```json",
        json.dumps(payload.get("checks", {}), indent=2),
        "```",
        "",
        "## Issues",
        "",
    ]
    if payload.get("issues"):
        for issue in payload["issues"]:
            lines.append(f"- `{issue['code']}`: {issue['message']}")
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(payload: dict[str, Any], *, output_json: str | None, output_md: str | None) -> None:
    if output_json:
        path = resolve_path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if output_md:
        write_markdown(resolve_path(output_md), payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_dir", default="data/curriculum/run_001")
    parser.add_argument("--summary_json")
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    parser.add_argument("--min_positive_rows", type=int, default=1)
    parser.add_argument(
        "--min_mode_rows",
        action="append",
        default=[],
        help=(
            "Optional mode coverage requirements over positive_sft rows. "
            "Use mode=count, e.g. direct=64,deep_narrow=64. May be repeated."
        ),
    )
    parser.add_argument("--max_loop_target", type=int, default=8)
    parser.add_argument("--require_programmatic_answer_check", action="store_true", default=True)
    parser.add_argument(
        "--allow_cross_model_only_answers",
        action="store_false",
        dest="require_programmatic_answer_check",
        help="Allow generated shards whose verified answers were only cross-model agreed, without a cheap deterministic check.",
    )
    parser.add_argument("--fail_on_no_go", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.min_mode_rows = parse_min_mode_rows(args.min_mode_rows)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_gate_payload(args)
    write_outputs(payload, output_json=args.output_json, output_md=args.output_md)
    print(f"status={payload['status']}")
    print(f"go={payload['go']}")
    print(f"issues={len(payload.get('issues', []))}")
    if args.output_json:
        print(f"output_json={resolve_path(args.output_json)}")
    if args.output_md:
        print(f"output_md={resolve_path(args.output_md)}")
    return 1 if args.fail_on_no_go and not payload["go"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
