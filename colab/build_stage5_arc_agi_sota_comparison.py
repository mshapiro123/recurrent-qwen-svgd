"""Build an ARC-AGI same-size baseline comparison artifact.

This is the authoritative comparison artifact consumed by
``build_stage5_claim_packet.py``. It intentionally does not scrape or invent
leaderboard numbers. Provide a baseline registry JSON with sourced same-size
model scores, then compare the selected recurrent ARC-AGI summary against the
best applicable baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_SOTA_COMPARISON_RUN_ID") or time.strftime(
    "stage5_arc_agi_sota_comparison_%Y%m%d_%H%M%S"
)

DEFAULT_BASELINE_REGISTRY = ROOT / "config" / "arc_agi_same_size_baselines.json"
DEFAULT_CANDIDATE_LABEL = os.environ.get("STAGE5_ARC_AGI_SOTA_CANDIDATE_LABEL", "phase1_arc_agi_tuned")
METRIC = os.environ.get("STAGE5_ARC_AGI_SOTA_METRIC", "selected_accuracy")
MIN_EXAMPLES = int(os.environ.get("STAGE5_ARC_AGI_SOTA_MIN_EXAMPLES", "100"))
MIN_MARGIN = float(os.environ.get("STAGE5_ARC_AGI_SOTA_MIN_MARGIN", "0.0"))


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_candidate_summary() -> Path | None:
    candidates: list[Path] = []
    root = ROOT / "outputs" / "stage5"
    if not root.exists():
        return None
    for path in root.glob("**/summary.json"):
        payload = safe_read_json(path)
        if not payload:
            continue
        if "phase1_arc_agi_tuned" in payload or (payload.get("summary") or {}).get("examples_with_targets"):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def summary_block(payload: dict[str, Any], *, label: str) -> dict[str, Any] | None:
    summary = payload.get("summary")
    if isinstance(summary, dict) and "examples_with_targets" in summary:
        return summary
    block = payload.get(label)
    if isinstance(block, dict):
        nested = block.get("summary")
        if isinstance(nested, dict):
            return nested
        if "examples_with_targets" in block:
            return block
    return None


def accuracy_from_summary(summary: dict[str, Any], metric: str) -> float | None:
    if metric in summary:
        return float(summary[metric])
    examples = int(summary.get("examples_with_targets", 0) or summary.get("examples", 0) or 0)
    if examples <= 0:
        return None
    if metric == "selected_accuracy":
        return int(summary.get("selected_exact", 0) or 0) / examples
    if metric == "best_of_k_accuracy":
        return int(summary.get("best_of_k_exact", 0) or 0) / examples
    if metric == "first_accuracy":
        return int(summary.get("first_exact", 0) or 0) / examples
    return None


def candidate_evidence(path: Path | None, *, label: str, metric: str) -> dict[str, Any]:
    if not path:
        return {"present": False}
    payload = safe_read_json(path)
    if not payload:
        return {"present": False, "path": path_for_cli(path)}
    summary = summary_block(payload, label=label)
    if not summary:
        return {"present": False, "path": path_for_cli(path), "reason": f"missing summary block {label!r}"}
    accuracy = accuracy_from_summary(summary, metric)
    return {
        "present": accuracy is not None,
        "path": path_for_cli(path),
        "run_id": str(payload.get("run_id") or path.parent.name),
        "label": label,
        "metric": metric,
        "accuracy": accuracy,
        "examples": int(summary.get("examples_with_targets", 0) or summary.get("examples", 0) or 0),
        "selected_exact": int(summary.get("selected_exact", 0) or 0),
        "best_of_k_exact": int(summary.get("best_of_k_exact", 0) or 0),
        "summary": summary,
    }


def load_baselines(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"present": False, "path": path_for_cli(path or DEFAULT_BASELINE_REGISTRY), "baselines": []}
    payload = read_json(path)
    baselines = payload.get("baselines")
    return {
        "present": isinstance(baselines, list) and bool(baselines),
        "path": path_for_cli(path),
        "benchmark": payload.get("benchmark"),
        "metric": payload.get("metric"),
        "same_size_band": payload.get("same_size_band") or {},
        "baselines": baselines if isinstance(baselines, list) else [],
        "source_notes": payload.get("source_notes"),
    }


def baseline_accuracy(row: dict[str, Any], metric: str) -> float | None:
    if row.get("metric") and row.get("metric") != metric:
        return None
    value = row.get("accuracy", row.get(metric))
    if value is None:
        return None
    return float(value)


def best_baseline(registry: dict[str, Any], metric: str) -> dict[str, Any] | None:
    scored: list[dict[str, Any]] = []
    for row in registry.get("baselines", []):
        if not isinstance(row, dict):
            continue
        accuracy = baseline_accuracy(row, metric)
        if accuracy is None:
            continue
        scored.append({**row, "accuracy": accuracy})
    if not scored:
        return None
    return max(scored, key=lambda row: float(row["accuracy"]))


def criterion(name: str, passed: bool, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "reason": reason, "evidence": evidence}


def build_sota_comparison(
    *,
    candidate_summary: Path | None,
    baseline_registry: Path | None,
    candidate_label: str,
    metric: str,
    min_examples: int,
    min_margin: float,
) -> dict[str, Any]:
    candidate = candidate_evidence(candidate_summary, label=candidate_label, metric=metric)
    registry = load_baselines(baseline_registry)
    best = best_baseline(registry, metric) if registry.get("present") else None
    candidate_present = bool(candidate.get("present"))
    candidate_examples = int(candidate.get("examples", 0) or 0)
    candidate_accuracy = candidate.get("accuracy")
    best_accuracy = float(best["accuracy"]) if best else None
    delta = (float(candidate_accuracy) - best_accuracy) if candidate_accuracy is not None and best_accuracy is not None else None

    criteria = [
        criterion(
            "candidate_summary_present",
            candidate_present,
            "Candidate ARC-AGI summary is present." if candidate_present else "Candidate ARC-AGI summary is missing.",
            candidate,
        ),
        criterion(
            "candidate_examples_sufficient",
            candidate_present and candidate_examples >= min_examples,
            (
                "Candidate ARC-AGI summary has enough examples."
                if candidate_present and candidate_examples >= min_examples
                else "Candidate ARC-AGI summary has too few examples for a SOTA comparison."
            ),
            candidate,
        ),
        criterion(
            "baseline_registry_present",
            bool(registry.get("present")) and best is not None,
            (
                "Same-size baseline registry has at least one score for the requested metric."
                if bool(registry.get("present")) and best is not None
                else "Same-size baseline registry is missing or has no score for the requested metric."
            ),
            registry,
        ),
        criterion(
            "candidate_meets_or_exceeds_best_baseline",
            delta is not None and delta >= min_margin,
            (
                "Candidate meets or exceeds the best same-size baseline by the required margin."
                if delta is not None and delta >= min_margin
                else "Candidate does not meet the best same-size baseline by the required margin."
            ),
            {"candidate": candidate, "best_baseline": best, "delta_accuracy": delta, "min_margin": min_margin},
        ),
    ]

    if not candidate_present:
        status = "needs_candidate_summary"
        next_step = "Run an ARC-AGI evaluation summary for the recurrent candidate."
    elif candidate_examples < min_examples:
        status = "needs_more_examples"
        next_step = "Evaluate the recurrent candidate on a larger ARC-AGI split before comparing to baselines."
    elif not registry.get("present") or best is None:
        status = "needs_baseline_registry"
        next_step = "Provide config/arc_agi_same_size_baselines.json with sourced same-size baseline scores."
    elif delta is not None and delta >= min_margin:
        status = "passed"
        next_step = "Rebuild the Stage 5 claim packet; it can now include this ARC-AGI SOTA comparison evidence."
    else:
        status = "failed"
        next_step = "Continue recurrent training or selector work before making a SOTA ARC-AGI claim."

    return {
        "run_id": RUN_ID,
        "gate": "stage5_arc_agi_sota_comparison",
        "kind": "arc_agi_sota_comparison",
        "status": status,
        "passed": status == "passed",
        "metric": metric,
        "candidate_label": candidate_label,
        "min_examples": min_examples,
        "min_margin": min_margin,
        "candidate": candidate,
        "baseline_registry": registry,
        "best_baseline": best,
        "delta_accuracy_vs_best_baseline": delta,
        "criteria": criteria,
        "next_step": next_step,
    }


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    best = payload.get("best_baseline") or {}
    candidate = payload.get("candidate") or {}
    lines = [
        f"# Stage 5 ARC-AGI Same-Size Comparison - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Metric: `{payload['metric']}`",
        f"- Candidate: `{candidate.get('accuracy')}` over `{candidate.get('examples')}` examples",
        f"- Best baseline: `{best.get('name')}` at `{best.get('accuracy')}`",
        f"- Delta accuracy: `{payload['delta_accuracy_vs_best_baseline']}`",
        f"- Next step: {payload['next_step']}",
        "",
        "## Criteria",
        "",
    ]
    for row in payload["criteria"]:
        lines.append(f"- `{row['name']}` passed `{row['passed']}`: {row['reason']}")
    lines.extend(
        [
            "",
            "## Baseline Registry Requirement",
            "",
            "This artifact is only authoritative if the baseline registry contains sourced same-size scores. "
            "Do not use this file for SOTA claims when status is `needs_baseline_registry`.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_summary_json")
    parser.add_argument("--baseline_registry_json")
    parser.add_argument("--candidate_label", default=DEFAULT_CANDIDATE_LABEL)
    parser.add_argument("--metric", default=METRIC)
    parser.add_argument("--min_examples", type=int, default=MIN_EXAMPLES)
    parser.add_argument("--min_margin", type=float, default=MIN_MARGIN)
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    args = parser.parse_args()

    candidate_summary = resolve_path(args.candidate_summary_json) if args.candidate_summary_json else latest_candidate_summary()
    baseline_registry = resolve_path(args.baseline_registry_json) if args.baseline_registry_json else DEFAULT_BASELINE_REGISTRY
    output_json = resolve_path(args.output_json) if args.output_json else ROOT / "outputs" / "stage5" / RUN_ID / "summary.json"
    output_md = resolve_path(args.output_md) if args.output_md else output_json.with_suffix(".md")
    payload = build_sota_comparison(
        candidate_summary=candidate_summary,
        baseline_registry=baseline_registry,
        candidate_label=args.candidate_label,
        metric=args.metric,
        min_examples=args.min_examples,
        min_margin=args.min_margin,
    )
    write_report(payload, output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
