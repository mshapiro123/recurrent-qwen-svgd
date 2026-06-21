"""Validate the sourced same-size ARC-AGI baseline registry.

This gate is intentionally conservative. The ARC-AGI same-size comparison is
only claim-ready when the registry contains sourced, non-placeholder baseline
scores in the configured model-size band.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_BASELINE_REGISTRY_RUN_ID") or time.strftime(
    "stage5_arc_agi_baseline_registry_%Y%m%d_%H%M%S"
)
DEFAULT_BASELINE_REGISTRY = ROOT / "config" / "arc_agi_same_size_baselines.json"
EXAMPLE_BASELINE_REGISTRY = ROOT / "config" / "arc_agi_same_size_baselines.example.json"
SUPPORTED_METRICS = {"selected_accuracy", "best_of_k_accuracy", "first_accuracy"}
SUPPORTED_EVIDENCE_TYPES = {"official_leaderboard", "paper", "model_card", "repository", "reproduced_eval"}
PLACEHOLDER_TOKENS = {
    "example",
    "placeholder",
    "replace_with",
    "todo",
    "tbd",
    "dummy",
    "fake",
    "unit-test",
}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def path_for_cli(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def baseline_registry_next_step(passed: bool) -> str:
    if passed:
        return "Run colab/build_stage5_arc_agi_sota_comparison.py against this registry."
    return (
        "Copy config/arc_agi_same_size_baselines.example.json to "
        "config/arc_agi_same_size_baselines.json, then replace every placeholder "
        "with sourced same-size ARC-AGI baseline scores before making a SOTA claim."
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def text_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def is_placeholder_text(value: Any) -> bool:
    text = text_value(value).casefold()
    return not text or any(token in text for token in PLACEHOLDER_TOKENS)


def is_authoritative_source(value: Any) -> bool:
    text = text_value(value)
    lower = text.casefold()
    if is_placeholder_text(text):
        return False
    if "example.com" in lower:
        return False
    return lower.startswith(("https://", "http://", "doi:", "arxiv:"))


def local_artifact_path(value: Any) -> Path | None:
    text = text_value(value)
    if is_placeholder_text(text):
        return None
    path = resolve_path(text)
    if not path.exists() or not path.is_file():
        return None
    if path.suffix.lower() != ".json":
        return None
    try:
        path.relative_to(ROOT)
    except ValueError:
        return None
    return path


def has_reproduced_eval_evidence(row: dict[str, Any]) -> bool:
    if text_value(row.get("evidence_type")) != "reproduced_eval":
        return False
    artifact = local_artifact_path(row.get("source_artifact") or row.get("artifact") or row.get("summary_json"))
    command = row.get("reproduction_command") or row.get("command")
    commit = row.get("git_commit") or row.get("commit")
    return artifact is not None and not is_placeholder_text(command) and not is_placeholder_text(commit)


def is_valid_date(value: Any) -> bool:
    text = text_value(value)
    if not DATE_RE.match(text):
        return False
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def issue(path: str, message: str, *, severity: str = "error") -> dict[str, str]:
    return {"path": path, "severity": severity, "message": message}


def criterion(name: str, passed: bool, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "reason": reason, "evidence": evidence}


def validate_registry_payload(payload: Any, *, source_path: Path | None = None) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    valid_baselines: list[dict[str, Any]] = []

    if not isinstance(payload, dict):
        issues.append(issue("$", "Registry JSON must be an object."))
        return build_payload(
            source_path=source_path,
            payload={},
            valid_baselines=valid_baselines,
            issues=issues,
            metric=None,
            arc_version=None,
            arc_split=None,
            same_size_band={},
        )

    benchmark = text_value(payload.get("benchmark"))
    if is_placeholder_text(benchmark):
        issues.append(issue("$.benchmark", "Benchmark must be present and non-placeholder."))

    arc_version = text_value(payload.get("arc_version"))
    if is_placeholder_text(arc_version):
        issues.append(issue("$.arc_version", "arc_version must identify the ARC-AGI version used by these baselines."))

    arc_split = text_value(payload.get("arc_split") or payload.get("eval_split"))
    if is_placeholder_text(arc_split):
        issues.append(issue("$.arc_split", "arc_split or eval_split must identify the evaluated ARC-AGI split."))

    metric = text_value(payload.get("metric"))
    if metric not in SUPPORTED_METRICS:
        issues.append(
            issue(
                "$.metric",
                f"Metric must be one of {sorted(SUPPORTED_METRICS)}.",
            )
        )

    band = payload.get("same_size_band")
    band = band if isinstance(band, dict) else {}
    min_params = band.get("min_params_b")
    max_params = band.get("max_params_b")
    band_valid = is_finite_number(min_params) and is_finite_number(max_params) and float(min_params) < float(max_params)
    if not band_valid:
        issues.append(
            issue(
                "$.same_size_band",
                "same_size_band must contain numeric min_params_b < max_params_b.",
            )
        )

    baselines = payload.get("baselines")
    if not isinstance(baselines, list) or not baselines:
        issues.append(issue("$.baselines", "baselines must be a non-empty list."))
        baselines = []

    for idx, row in enumerate(baselines):
        row_path = f"$.baselines[{idx}]"
        if not isinstance(row, dict):
            issues.append(issue(row_path, "Baseline row must be an object."))
            continue
        row_issues_start = len(issues)

        name = text_value(row.get("name"))
        if is_placeholder_text(name):
            issues.append(issue(f"{row_path}.name", "Baseline name must be present and non-placeholder."))

        params_b = row.get("params_b")
        if not is_finite_number(params_b):
            issues.append(issue(f"{row_path}.params_b", "params_b must be a finite number."))
        elif band_valid and not (float(min_params) <= float(params_b) <= float(max_params)):
            issues.append(
                issue(
                    f"{row_path}.params_b",
                    "params_b must fall inside same_size_band for this comparison.",
                )
            )

        row_metric = text_value(row.get("metric")) or metric
        if is_placeholder_text(row.get("metric")):
            issues.append(issue(f"{row_path}.metric", "Baseline row must explicitly identify the scored metric."))
        if row_metric != metric:
            issues.append(issue(f"{row_path}.metric", "Baseline metric must match registry metric."))

        row_arc_version = text_value(row.get("arc_version"))
        if is_placeholder_text(row_arc_version):
            issues.append(issue(f"{row_path}.arc_version", "Baseline row must explicitly identify ARC-AGI version."))
        elif row_arc_version != arc_version:
            issues.append(issue(f"{row_path}.arc_version", "Baseline ARC-AGI version must match registry arc_version."))

        row_arc_split = text_value(row.get("arc_split") or row.get("eval_split"))
        if is_placeholder_text(row_arc_split):
            issues.append(issue(f"{row_path}.arc_split", "Baseline row must explicitly identify ARC-AGI split."))
        elif row_arc_split != arc_split:
            issues.append(issue(f"{row_path}.arc_split", "Baseline ARC-AGI split must match registry arc_split."))

        evidence_type = text_value(row.get("evidence_type"))
        if evidence_type not in SUPPORTED_EVIDENCE_TYPES:
            issues.append(
                issue(
                    f"{row_path}.evidence_type",
                    f"evidence_type must be one of {sorted(SUPPORTED_EVIDENCE_TYPES)}.",
                )
            )

        accuracy = row.get("accuracy", row.get(metric))
        if not is_finite_number(accuracy) or not (0.0 <= float(accuracy) <= 1.0):
            issues.append(issue(f"{row_path}.accuracy", "accuracy must be a finite number between 0 and 1."))

        source = row.get("source") or row.get("source_url") or row.get("citation_url")
        reproduced_eval = evidence_type == "reproduced_eval"
        has_external_source = is_authoritative_source(source)
        has_local_reproduction = has_reproduced_eval_evidence(row)
        if not has_external_source and not has_local_reproduction:
            if reproduced_eval:
                issues.append(
                    issue(
                        f"{row_path}.source_artifact",
                        "reproduced_eval rows need either an authoritative source URL/DOI/arXiv reference or an existing local JSON source_artifact plus reproduction_command and git_commit.",
                    )
                )
            else:
                issues.append(
                    issue(
                        f"{row_path}.source",
                        "source must be an authoritative URL, DOI, or arXiv reference, not a placeholder.",
                    )
                )

        date_value = row.get("accessed_date") or row.get("as_of_date")
        if not is_valid_date(date_value):
            issues.append(
                issue(
                    f"{row_path}.accessed_date",
                    "Each baseline needs accessed_date or as_of_date in YYYY-MM-DD format.",
                )
            )

        if len(issues) == row_issues_start:
            valid_baselines.append(
                {
                    **row,
                    "name": name,
                    "params_b": float(params_b),
                    "metric": metric,
                    "arc_version": row_arc_version,
                    "arc_split": row_arc_split,
                    "evidence_type": evidence_type,
                    "accuracy": float(accuracy),
                    "source": text_value(source),
                    "source_artifact": path_for_cli(
                        local_artifact_path(row.get("source_artifact") or row.get("artifact") or row.get("summary_json"))
                    ),
                }
            )

    return build_payload(
        source_path=source_path,
        payload=payload,
        valid_baselines=valid_baselines,
        issues=issues,
        metric=metric,
        arc_version=arc_version,
        arc_split=arc_split,
        same_size_band=band,
    )


def build_payload(
    *,
    source_path: Path | None,
    payload: dict[str, Any],
    valid_baselines: list[dict[str, Any]],
    issues: list[dict[str, str]],
    metric: str | None,
    arc_version: str | None,
    arc_split: str | None,
    same_size_band: dict[str, Any],
) -> dict[str, Any]:
    errors = [row for row in issues if row["severity"] == "error"]
    passed = not errors and bool(valid_baselines)
    best = max(valid_baselines, key=lambda row: float(row["accuracy"])) if valid_baselines else None
    criteria = [
        criterion(
            "registry_present",
            source_path is not None and source_path.exists(),
            "Registry file is present." if source_path is not None and source_path.exists() else "Registry file is missing.",
            {"path": path_for_cli(source_path)},
        ),
        criterion(
            "registry_schema_valid",
            not errors,
            "Registry schema and baseline rows are valid." if not errors else "Registry has blocking validation issues.",
            {"issues": errors},
        ),
        criterion(
            "arc_agi_metadata_present",
            bool(arc_version and arc_split),
            (
                "Registry identifies the ARC-AGI version and split."
                if arc_version and arc_split
                else "Registry must identify ARC-AGI version and split for claim-safe comparisons."
            ),
            {"arc_version": arc_version, "arc_split": arc_split},
        ),
        criterion(
            "has_valid_same_size_baseline",
            bool(valid_baselines),
            "At least one valid same-size baseline is available."
            if valid_baselines
            else "No valid same-size baseline rows are available.",
            {"valid_baseline_count": len(valid_baselines)},
        ),
    ]
    return {
        "run_id": RUN_ID,
        "gate": "stage5_arc_agi_baseline_registry",
        "kind": "arc_agi_baseline_registry",
        "status": "passed" if passed else "needs_baseline_registry",
        "passed": passed,
        "path": path_for_cli(source_path),
        "benchmark": payload.get("benchmark"),
        "metric": metric,
        "arc_version": arc_version,
        "arc_split": arc_split,
        "same_size_band": same_size_band,
        "baseline_count": len(payload.get("baselines", [])) if isinstance(payload.get("baselines"), list) else 0,
        "valid_baseline_count": len(valid_baselines),
        "valid_baselines": valid_baselines,
        "best_baseline": best,
        "issues": issues,
        "criteria": criteria,
        "next_step": baseline_registry_next_step(passed),
    }


def validate_baseline_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return build_payload(
            source_path=path,
            payload={},
            valid_baselines=[],
            issues=[issue("$", "Baseline registry file is missing.")],
            metric=None,
            arc_version=None,
            arc_split=None,
            same_size_band={},
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return build_payload(
            source_path=path,
            payload={},
            valid_baselines=[],
            issues=[issue("$", f"Could not read registry JSON: {exc}")],
            metric=None,
            arc_version=None,
            arc_split=None,
            same_size_band={},
        )
    return validate_registry_payload(payload, source_path=path)


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    write_json(output_json, payload)
    lines = [
        f"# Stage 5 ARC-AGI Baseline Registry - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Registry: `{payload.get('path')}`",
        f"- Example schema: `{path_for_cli(EXAMPLE_BASELINE_REGISTRY)}`",
        f"- ARC version/split: `{payload.get('arc_version')}` / `{payload.get('arc_split')}`",
        f"- Metric: `{payload.get('metric')}`",
        f"- Valid baselines: `{payload.get('valid_baseline_count')}` / `{payload.get('baseline_count')}`",
        f"- Best baseline: `{(payload.get('best_baseline') or {}).get('name')}` at `{(payload.get('best_baseline') or {}).get('accuracy')}`",
        f"- Next step: {payload['next_step']}",
        "",
        "## Valid Baselines",
        "",
    ]
    valid_baselines = payload.get("valid_baselines") or []
    if valid_baselines:
        for row in valid_baselines:
            source_ref = row.get("source") or row.get("source_artifact")
            lines.append(
                f"- `{row.get('name')}`: accuracy `{row.get('accuracy')}`, params `{row.get('params_b')}`B, "
                f"ARC `{row.get('arc_version')}`/`{row.get('arc_split')}`, evidence `{row.get('evidence_type')}`, "
                f"source `{source_ref}`"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Issues",
            "",
        ]
    )
    if payload["issues"]:
        for row in payload["issues"]:
            lines.append(f"- `{row['severity']}` `{row['path']}`: {row['message']}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Requirement",
            "",
            "A same-size ARC-AGI SOTA comparison requires sourced, non-placeholder "
            "baseline scores inside the configured parameter band for the same "
            "ARC-AGI version and split. Each row must also identify whether the "
            "score came from an official leaderboard, paper, model card, "
            "repository, or reproduced eval. Placeholder, mixed-split, or "
            "unsourced values keep the SOTA gate closed.",
            "",
            "`reproduced_eval` rows may use a local `source_artifact` instead "
            "of an external source only when the artifact exists in this repo "
            "and the row also supplies `reproduction_command` and `git_commit`.",
            "",
            "Required row fields: `name`, `params_b`, `arc_version`, `arc_split`, "
            "`metric`, `accuracy`, `evidence_type`, `source`, and "
            "`accessed_date` or `as_of_date`; for `reproduced_eval`, use either "
            "`source` or the audited `source_artifact` bundle above.",
            "",
            f"Accepted evidence types: `{', '.join(sorted(SUPPORTED_EVIDENCE_TYPES))}`.",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_registry_json", default=str(DEFAULT_BASELINE_REGISTRY))
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    args = parser.parse_args()

    registry_path = resolve_path(args.baseline_registry_json)
    output_json = (
        resolve_path(args.output_json)
        if args.output_json
        else ROOT / "outputs" / "stage5" / RUN_ID / "summary.json"
    )
    output_md = resolve_path(args.output_md) if args.output_md else output_json.with_suffix(".md")
    payload = validate_baseline_registry(registry_path)
    write_report(payload, output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
