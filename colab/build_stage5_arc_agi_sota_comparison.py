"""Build an ARC-AGI same-size baseline comparison artifact.

This is the authoritative comparison artifact consumed by
``build_stage5_claim_packet.py``. It intentionally does not scrape or invent
leaderboard numbers. Provide a baseline registry JSON with sourced same-size
model scores, then compare the selected recurrent ARC-AGI summary against the
best applicable baseline. Local ``reproduced_eval`` rows are accepted as
reproduced-control evidence, but only public-source rows establish SOTA scope.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
try:
    from colab.validate_arc_agi_baseline_registry import validate_baseline_registry
except ModuleNotFoundError:  # pragma: no cover - used when run as a script from colab/
    sys.path.insert(0, str(ROOT))
    from colab.validate_arc_agi_baseline_registry import validate_baseline_registry

RUN_ID = os.environ.get("STAGE5_ARC_AGI_SOTA_COMPARISON_RUN_ID") or time.strftime(
    "stage5_arc_agi_sota_comparison_%Y%m%d_%H%M%S"
)

DEFAULT_BASELINE_REGISTRY = ROOT / "config" / "arc_agi_same_size_baselines.json"
DEFAULT_CANDIDATE_LABEL = os.environ.get("STAGE5_ARC_AGI_SOTA_CANDIDATE_LABEL", "auto")
METRIC = os.environ.get("STAGE5_ARC_AGI_SOTA_METRIC", "selected_accuracy")
MIN_EXAMPLES = int(os.environ.get("STAGE5_ARC_AGI_SOTA_MIN_EXAMPLES", "100"))
MIN_MARGIN = float(os.environ.get("STAGE5_ARC_AGI_SOTA_MIN_MARGIN", "0.0"))
PUBLIC_SOTA_EVIDENCE_TYPES = {"official_leaderboard", "paper", "model_card", "repository"}


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


def candidate_discovery_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def discovery_has_arc_metadata(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get("arc_version") and (metadata.get("arc_split") or metadata.get("eval_split")))


def discovery_has_params(metadata: dict[str, Any]) -> bool:
    for key in ("params_b", "candidate_params_b", "model_params_b", "base_model_params_b", "parameter_count_b"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return True
    return False


def discovery_summary_examples(block: Any) -> int:
    if not isinstance(block, dict):
        return 0
    summary = block.get("summary")
    if isinstance(summary, dict):
        block = summary
    return int(block.get("examples_with_targets", 0) or block.get("examples", 0) or 0)


def candidate_discovery_score(payload: dict[str, Any]) -> tuple[int, int] | None:
    """Rank default SOTA candidates by claim-readiness, not file recency alone."""

    label_scores = [
        (30, discovery_summary_examples(payload.get("recovered"))),
        (20, discovery_summary_examples(payload.get("phase1_arc_agi_tuned"))),
        (10, discovery_summary_examples(payload.get("summary"))),
    ]
    label_score, examples = max(label_scores, key=lambda item: (item[0] if item[1] > 0 else -1, item[1]))
    if examples <= 0:
        return None

    metadata = candidate_discovery_metadata(payload)
    score = label_score
    if discovery_has_arc_metadata(metadata):
        score += 100
    if discovery_has_params(metadata):
        score += 50
    return score, examples


def latest_candidate_summary() -> Path | None:
    candidates: list[tuple[int, int, float, Path]] = []
    root = ROOT / "outputs" / "stage5"
    if not root.exists():
        return None
    for path in root.glob("**/summary.json"):
        payload = safe_read_json(path)
        if not payload:
            continue
        score = candidate_discovery_score(payload)
        if score is not None:
            candidates.append((*score, path.stat().st_mtime, path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1], item[2]), reverse=True)[0][3]


def candidate_label_order(label: str) -> list[str]:
    if label != "auto":
        return [label]
    return ["recovered", "phase1_arc_agi_tuned", "summary"]


def summary_block(payload: dict[str, Any], *, label: str) -> tuple[str, dict[str, Any]] | None:
    summary = payload.get("summary")
    for candidate_label in candidate_label_order(label):
        if candidate_label == "summary" and isinstance(summary, dict) and "examples_with_targets" in summary:
            return candidate_label, summary
        block = payload.get(candidate_label)
        if isinstance(block, dict):
            nested = block.get("summary")
            if isinstance(nested, dict):
                return candidate_label, nested
            if "examples_with_targets" in block:
                return candidate_label, block
    return None


def candidate_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def candidate_arc_split(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("arc_split") or metadata.get("eval_split")
    return str(value) if value else None


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def candidate_params_b(metadata: dict[str, Any]) -> float | None:
    for key in ("params_b", "candidate_params_b", "model_params_b", "base_model_params_b", "parameter_count_b"):
        value = finite_float(metadata.get(key))
        if value is not None:
            return value
    return None


def has_arc_agi_metadata(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get("arc_version") and candidate_arc_split(metadata))


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
    resolved = summary_block(payload, label=label)
    if not resolved:
        return {"present": False, "path": path_for_cli(path), "reason": f"missing summary block {label!r}"}
    resolved_label, summary = resolved
    accuracy = accuracy_from_summary(summary, metric)
    metadata = candidate_metadata(payload)
    return {
        "present": accuracy is not None,
        "path": path_for_cli(path),
        "run_id": str(payload.get("run_id") or path.parent.name),
        "requested_label": label,
        "label": resolved_label,
        "metric": metric,
        "accuracy": accuracy,
        "examples": int(summary.get("examples_with_targets", 0) or summary.get("examples", 0) or 0),
        "selected_exact": int(summary.get("selected_exact", 0) or 0),
        "best_of_k_exact": int(summary.get("best_of_k_exact", 0) or 0),
        "arc_version": metadata.get("arc_version"),
        "arc_split": candidate_arc_split(metadata),
        "params_b": candidate_params_b(metadata),
        "metadata": metadata,
        "summary": summary,
    }


def load_baselines(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        validation = validate_baseline_registry(path or DEFAULT_BASELINE_REGISTRY)
        return {
            "present": False,
            "valid": False,
            "path": path_for_cli(path or DEFAULT_BASELINE_REGISTRY),
            "baselines": [],
            "validation": validation,
        }
    validation = validate_baseline_registry(path)
    baselines = validation.get("valid_baselines")
    return {
        "present": bool(validation.get("passed")) and isinstance(baselines, list) and bool(baselines),
        "valid": bool(validation.get("passed")),
        "path": path_for_cli(path),
        "benchmark": validation.get("benchmark"),
        "metric": validation.get("metric"),
        "arc_version": validation.get("arc_version"),
        "arc_split": validation.get("arc_split"),
        "same_size_band": validation.get("same_size_band") or {},
        "baselines": baselines if isinstance(baselines, list) else [],
        "validation": validation,
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


def is_public_sota_baseline(row: dict[str, Any]) -> bool:
    return normalized_text(row.get("evidence_type")) in PUBLIC_SOTA_EVIDENCE_TYPES and bool(row.get("source"))


def best_public_sota_baseline(registry: dict[str, Any], metric: str) -> dict[str, Any] | None:
    scored: list[dict[str, Any]] = []
    for row in registry.get("baselines", []):
        if not isinstance(row, dict) or not is_public_sota_baseline(row):
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


def normalized_text(value: Any) -> str:
    return str(value).strip().casefold() if value is not None else ""


def registry_matches_candidate_arc(candidate: dict[str, Any], registry: dict[str, Any]) -> bool:
    return (
        bool(candidate.get("arc_version") and candidate.get("arc_split"))
        and normalized_text(candidate.get("arc_version")) == normalized_text(registry.get("arc_version"))
        and normalized_text(candidate.get("arc_split")) == normalized_text(registry.get("arc_split"))
    )


def registry_band_contains_candidate(candidate: dict[str, Any], registry: dict[str, Any]) -> bool:
    params_b = finite_float(candidate.get("params_b"))
    band = registry.get("same_size_band")
    if params_b is None or not isinstance(band, dict):
        return False
    min_params = finite_float(band.get("min_params_b"))
    max_params = finite_float(band.get("max_params_b"))
    return min_params is not None and max_params is not None and min_params <= params_b <= max_params


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
    best_registry = best_baseline(registry, metric) if registry.get("present") else None
    best_public = best_public_sota_baseline(registry, metric) if registry.get("present") else None
    comparison_scope = "public_sota" if best_public else "reproduced_control"
    best = best_public or best_registry
    candidate_present = bool(candidate.get("present"))
    candidate_examples = int(candidate.get("examples", 0) or 0)
    candidate_has_arc_metadata = has_arc_agi_metadata(candidate.get("metadata") or {})
    candidate_has_params = finite_float(candidate.get("params_b")) is not None
    registry_candidate_match = registry_matches_candidate_arc(candidate, registry)
    candidate_in_same_size_band = registry_band_contains_candidate(candidate, registry)
    candidate_accuracy = candidate.get("accuracy")
    best_accuracy = float(best["accuracy"]) if best else None
    delta = (float(candidate_accuracy) - best_accuracy) if candidate_accuracy is not None and best_accuracy is not None else None
    best_registry_accuracy = float(best_registry["accuracy"]) if best_registry else None
    registry_delta = (
        float(candidate_accuracy) - best_registry_accuracy
        if candidate_accuracy is not None and best_registry_accuracy is not None
        else None
    )
    public_delta = delta if best_public else None
    if delta is None or delta < min_margin:
        comparison_reason = "Candidate does not meet the best same-size baseline by the required margin."
    elif comparison_scope == "public_sota":
        comparison_reason = "Candidate meets or exceeds the best same-size public SOTA baseline by the required margin."
    else:
        comparison_reason = "Candidate meets or exceeds the best reproduced same-size control by the required margin."

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
            "candidate_arc_agi_metadata_present",
            candidate_present and candidate_has_arc_metadata,
            (
                "Candidate summary has ARC-AGI version and split metadata."
                if candidate_present and candidate_has_arc_metadata
                else "Candidate summary must be a main ARC-AGI run summary with arc_version and eval/arc split metadata."
            ),
            candidate,
        ),
        criterion(
            "candidate_params_present",
            candidate_present and candidate_has_params,
            (
                "Candidate summary identifies the model parameter count."
                if candidate_present and candidate_has_params
                else "Candidate summary must include params_b or equivalent model-size metadata."
            ),
            candidate,
        ),
        criterion(
            "baseline_registry_present",
            bool(registry.get("present")) and best_registry is not None,
            (
                "Same-size baseline registry is valid and has at least one score for the requested metric."
                if bool(registry.get("present")) and best_registry is not None
                else "Same-size baseline registry is missing, invalid, or has no score for the requested metric."
            ),
            registry,
        ),
        criterion(
            "public_sota_baseline_present",
            bool(registry.get("present")) and best_public is not None,
            (
                "Same-size registry has at least one public-source SOTA baseline."
                if bool(registry.get("present")) and best_public is not None
                else "Registry only has reproduced/local controls; add public-source same-size baselines before SOTA claims."
            ),
            {"best_public_baseline": best_public, "baseline_registry": registry},
        ),
        criterion(
            "baseline_registry_matches_candidate_arc",
            bool(registry.get("present")) and registry_candidate_match,
            (
                "Baseline registry ARC-AGI version/split matches the candidate summary."
                if bool(registry.get("present")) and registry_candidate_match
                else "Baseline registry must match the candidate ARC-AGI version and split."
            ),
            {"candidate": candidate, "baseline_registry": registry},
        ),
        criterion(
            "candidate_inside_same_size_band",
            bool(registry.get("present")) and candidate_in_same_size_band,
            (
                "Candidate parameter count falls inside the same-size baseline band."
                if bool(registry.get("present")) and candidate_in_same_size_band
                else "Candidate parameter count must fall inside the registry same-size band."
            ),
            {"candidate": candidate, "baseline_registry": registry},
        ),
        criterion(
            "candidate_meets_or_exceeds_best_baseline",
            delta is not None and delta >= min_margin,
            comparison_reason,
            {
                "candidate": candidate,
                "best_baseline": best,
                "best_registry_baseline": best_registry,
                "best_public_baseline": best_public,
                "delta_accuracy": delta,
                "comparison_scope": comparison_scope,
                "min_margin": min_margin,
            },
        ),
    ]

    if not candidate_present:
        status = "needs_candidate_summary"
        next_step = "Run an ARC-AGI evaluation summary for the recurrent candidate."
    elif not candidate_has_arc_metadata:
        status = "needs_arc_agi_candidate_metadata"
        next_step = (
            "Use a main ARC-AGI run summary with metadata.arc_version and metadata.arc_split/eval_split; "
            "do not compare ad hoc eval summaries to same-size baselines."
        )
    elif not candidate_has_params:
        status = "needs_candidate_size_metadata"
        next_step = (
            "Add candidate model-size metadata such as metadata.params_b before comparing to same-size baselines."
        )
    elif candidate_examples < min_examples:
        status = "needs_more_examples"
        next_step = "Evaluate the recurrent candidate on a larger ARC-AGI split before comparing to baselines."
    elif not registry.get("present") or best_registry is None:
        status = "needs_baseline_registry"
        next_step = (
            "Provide a validator-passing config/arc_agi_same_size_baselines.json with sourced same-size "
            "baseline scores."
        )
    elif not registry_candidate_match:
        status = "needs_matching_baseline_registry"
        next_step = (
            "Use a baseline registry whose arc_version and arc_split match the candidate ARC-AGI run summary."
        )
    elif not candidate_in_same_size_band:
        status = "needs_candidate_size_match"
        next_step = "Use a same-size baseline registry whose parameter band contains the recurrent candidate."
    elif best_public is None:
        if delta is not None and delta >= min_margin:
            status = "passed_reproduced_control"
            next_step = (
                "Candidate beat the reproduced same-size control, but this is not SOTA evidence. Add public-source "
                "same-size baselines before rebuilding the claim packet as a SOTA-facing artifact."
            )
        else:
            status = "failed_reproduced_control"
            next_step = (
                "Candidate does not beat the reproduced same-size control; continue recurrent recovery or selector work "
                "before spending effort on SOTA-facing baselines."
            )
    elif public_delta is not None and public_delta >= min_margin:
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
        "control_passed": status in {"passed", "passed_reproduced_control"} or (
            registry_delta is not None and registry_delta >= min_margin
        ),
        "comparison_scope": comparison_scope,
        "metric": metric,
        "candidate_label": candidate_label,
        "min_examples": min_examples,
        "min_margin": min_margin,
        "candidate": candidate,
        "baseline_registry": registry,
        "candidate_registry_arc_match": registry_candidate_match,
        "candidate_in_same_size_band": candidate_in_same_size_band,
        "best_baseline": best,
        "best_registry_baseline": best_registry,
        "best_public_baseline": best_public,
        "delta_accuracy_vs_best_baseline": delta,
        "delta_accuracy_vs_best_registry_baseline": registry_delta,
        "delta_accuracy_vs_best_public_baseline": public_delta,
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
        f"- Comparison scope: `{payload.get('comparison_scope')}`",
        f"- Metric: `{payload['metric']}`",
        f"- Candidate: `{candidate.get('accuracy')}` over `{candidate.get('examples')}` examples",
        f"- Candidate ARC version/split: `{candidate.get('arc_version')}` / `{candidate.get('arc_split')}`",
        f"- Candidate params (B): `{candidate.get('params_b')}`",
        f"- Baseline ARC version/split: `{payload.get('baseline_registry', {}).get('arc_version')}` / `{payload.get('baseline_registry', {}).get('arc_split')}`",
        f"- Best baseline: `{best.get('name')}` at `{best.get('accuracy')}`",
        f"- Best baseline params (B): `{best.get('params_b')}`",
        f"- Best baseline evidence: `{best.get('evidence_type')}` from `{best.get('source')}`",
        f"- Delta accuracy: `{payload['delta_accuracy_vs_best_baseline']}`",
        f"- Delta vs best public baseline: `{payload.get('delta_accuracy_vs_best_public_baseline')}`",
        f"- Delta vs best registry baseline: `{payload.get('delta_accuracy_vs_best_registry_baseline')}`",
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
            "Each accepted baseline row must explicitly match the ARC-AGI version/split and identify its evidence type. "
            "Rows with `evidence_type: reproduced_eval` are useful reproduced controls, but do not establish public "
            "SOTA scope by themselves. Do not use this file for SOTA claims unless status is `passed` and "
            "`comparison_scope` is `public_sota`.",
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
