"""Assess Stage 5 benchmark/release readiness from saved artifacts.

This is a no-GPU audit for the Stage 5E gate. It does not claim SOTA. It checks
whether the current recurrent artifact has enough evidence to move into broader
benchmarks / release packaging:

* recovered recurrent is competitive with base on a sufficiently large ARC-AGI
  slice;
* same-recipe dense-vs-recurrent architecture evidence exists;
* a Hugging Face export artifact exists with checkpoint metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_RELEASE_GATE_RUN_ID") or time.strftime(
    "stage5_release_gate_%Y%m%d_%H%M%S"
)


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


def summary_metrics(payload_or_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not payload_or_summary:
        return {}
    summary = payload_or_summary.get("summary")
    if isinstance(summary, dict):
        return summary
    return payload_or_summary


def metric(summary: dict[str, Any] | None, key: str) -> int:
    if not summary:
        return 0
    return int(summary.get(key, 0) or 0)


def is_recovered_benchmark(payload: dict[str, Any]) -> bool:
    benchmark = payload.get("recovered_benchmark") or payload
    return {"base", "phase1_start", "recovered"} <= set(benchmark)


def is_recipe_control(payload: dict[str, Any]) -> bool:
    return payload.get("gate") == "stage5_same_recipe_architecture"


def is_recipe_selector_conversion(payload: dict[str, Any]) -> bool:
    return (
        payload.get("gate") == "stage5_same_recipe_selector_conversion"
        or payload.get("kind") == "recipe_selector_conversion"
    )


def is_hf_export(payload: dict[str, Any]) -> bool:
    return bool(payload.get("export_dir") and payload.get("checkpoint") and isinstance(payload.get("metadata"), dict))


def latest_matching(paths: list[Path], predicate: Callable[[dict[str, Any]], bool]) -> Path | None:
    matches: list[Path] = []
    for path in paths:
        payload = safe_read_json(path)
        if payload and predicate(payload):
            matches.append(path)
    if not matches:
        return None
    return sorted(matches, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def summary_files(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.glob("**/summary.json"))
    return sorted(set(files), key=lambda path: str(path))


def recovered_benchmark_evidence(path: Path | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    if not path or not payload:
        return {"present": False}
    benchmark = payload.get("recovered_benchmark") or payload
    base = summary_metrics(benchmark.get("base"))
    recovered = summary_metrics(benchmark.get("recovered"))
    deltas = benchmark.get("deltas") or {}
    recovered_vs_base = deltas.get("recovered_vs_base") or {}
    selected_delta = int(
        recovered_vs_base.get(
            "selected_exact_delta",
            metric(recovered, "selected_exact") - metric(base, "selected_exact"),
        )
        or 0
    )
    best_delta = int(
        recovered_vs_base.get(
            "best_of_k_exact_delta",
            metric(recovered, "best_of_k_exact") - metric(base, "best_of_k_exact"),
        )
        or 0
    )
    examples = max(metric(base, "examples_with_targets"), metric(recovered, "examples_with_targets"))
    return {
        "present": True,
        "path": path_for_cli(path),
        "run_id": str(benchmark.get("run_id") or payload.get("run_id") or path.parent.name),
        "examples": examples,
        "selected_delta_recovered_vs_base": selected_delta,
        "best_of_k_delta_recovered_vs_base": best_delta,
        "base": base,
        "recovered": recovered,
    }


def architecture_evidence(path: Path | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    if not path or not payload:
        return {"present": False}
    decision = payload.get("decision_evidence") or {}
    aggregate = decision.get("aggregate") or {}
    hard = decision.get("hard") or {}
    aggregate_best = decision.get("aggregate_best_of_k") or {}
    hard_best = decision.get("hard_best_of_k") or {}
    return {
        "present": True,
        "path": path_for_cli(path),
        "run_id": str(payload.get("run_id") or path.parent.name),
        "status": payload.get("status"),
        "passed": bool(payload.get("passed", False)),
        "reason": payload.get("reason"),
        "next_step": payload.get("next_step"),
        "aggregate_selected_delta": int(aggregate.get("delta_exact", 0) or 0),
        "hard_selected_delta": int(hard.get("delta_exact", 0) or 0),
        "aggregate_best_of_k_delta": int(aggregate_best.get("delta_exact", 0) or 0),
        "hard_best_of_k_delta": int(hard_best.get("delta_exact", 0) or 0),
    }


def selector_conversion_evidence(path: Path | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    if not path or not payload:
        return {"present": False}
    best = payload.get("best_selector") or {}
    return {
        "present": True,
        "path": path_for_cli(path),
        "run_id": str(payload.get("run_id") or path.parent.name),
        "status": payload.get("status"),
        "passed": bool(payload.get("passed", False)),
        "reason": payload.get("reason"),
        "next_step": payload.get("next_step"),
        "passing_selectors": payload.get("passing_selectors") or [],
        "best_selector": best,
    }


def hf_export_evidence(path: Path | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    if not path or not payload:
        return {"present": False}
    metadata = payload.get("metadata") or {}
    return {
        "present": True,
        "path": path_for_cli(path),
        "run_id": str(payload.get("run_id") or path.parent.name),
        "export_dir": payload.get("export_dir"),
        "checkpoint": payload.get("checkpoint"),
        "hf_repo_id": payload.get("hf_repo_id"),
        "uploaded": bool(payload.get("uploaded", False)),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "architecture_evidence_status": (metadata.get("architecture_evidence") or {}).get("status"),
    }


def criterion(name: str, passed: bool, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "reason": reason, "evidence": evidence}


def evidence_source(evidence: dict[str, Any]) -> str | None:
    if evidence.get("path"):
        return str(evidence["path"])
    sources = []
    for key in ("architecture", "selector_conversion"):
        nested = evidence.get(key)
        if isinstance(nested, dict) and nested.get("path"):
            sources.append(f"{key}:{nested['path']}")
    return ", ".join(sources) if sources else None


def assess_release_gate(
    *,
    benchmark_summary: Path | None,
    recipe_control_summary: Path | None,
    hf_export_summary: Path | None,
    min_arc_examples: int,
    selector_conversion_summary: Path | None = None,
) -> dict[str, Any]:
    benchmark_payload = safe_read_json(benchmark_summary) if benchmark_summary else None
    recipe_payload = safe_read_json(recipe_control_summary) if recipe_control_summary else None
    selector_conversion_payload = safe_read_json(selector_conversion_summary) if selector_conversion_summary else None
    export_payload = safe_read_json(hf_export_summary) if hf_export_summary else None

    benchmark = recovered_benchmark_evidence(benchmark_summary, benchmark_payload)
    architecture = architecture_evidence(recipe_control_summary, recipe_payload)
    selector_conversion = selector_conversion_evidence(selector_conversion_summary, selector_conversion_payload)
    export = hf_export_evidence(hf_export_summary, export_payload)

    benchmark_present = bool(benchmark.get("present"))
    benchmark_large_enough = int(benchmark.get("examples", 0) or 0) >= min_arc_examples
    benchmark_nonnegative = (
        int(benchmark.get("selected_delta_recovered_vs_base", -10**9) or 0) >= 0
        and int(benchmark.get("best_of_k_delta_recovered_vs_base", -10**9) or 0) >= 0
    )
    architecture_present = bool(architecture.get("present"))
    architecture_status = str(architecture.get("status") or "missing")
    architecture_passed = bool(architecture.get("passed", False))
    selector_conversion_passed = bool(selector_conversion.get("passed", False))
    architecture_or_conversion_passed = architecture_passed or selector_conversion_passed
    export_present = bool(export.get("present"))
    export_has_hash = bool(export.get("checkpoint_sha256"))

    criteria = [
        criterion(
            "arc_benchmark_confirmation",
            benchmark_present and benchmark_large_enough and benchmark_nonnegative,
            (
                "Recovered recurrent is non-negative versus base on a sufficiently large ARC-AGI slice."
                if benchmark_present and benchmark_large_enough and benchmark_nonnegative
                else "Need a larger or non-negative recovered-vs-base ARC-AGI benchmark before broader claims."
            ),
            benchmark,
        ),
        criterion(
            "same_recipe_architecture_or_selector_conversion",
            architecture_or_conversion_passed,
            (
                "Same-recipe recurrent-vs-dense architecture gate passed."
                if architecture_passed
                else (
                    "Same-recipe selector conversion gate passed against the dense control."
                    if selector_conversion_passed
                    else f"Same-recipe architecture gate is {architecture_status!r}, and selector conversion has not passed."
                )
            ),
            {"architecture": architecture, "selector_conversion": selector_conversion},
        ),
        criterion(
            "hf_export_artifact",
            export_present and export_has_hash,
            (
                "HF export artifact exists with checkpoint hash metadata."
                if export_present and export_has_hash
                else "Need an HF export artifact with checkpoint metadata."
            ),
            export,
        ),
    ]

    if not benchmark_present or not benchmark_large_enough:
        status = "needs_benchmark_confirmation"
        next_step = "Run or replicate recovered-vs-base ARC-AGI benchmark on a larger held-out slice."
    elif not benchmark_nonnegative:
        status = "needs_recovery_training"
        next_step = "Continue deterministic recurrent recovery before release or broader benchmark claims."
    elif not architecture_present and not selector_conversion_passed:
        status = "needs_architecture_evidence"
        next_step = "Run dense-control plus matched recurrent SFT, then assess same-recipe architecture lift."
    elif architecture_status == "needs_selector_conversion" and not selector_conversion_passed:
        status = "needs_selector_conversion"
        next_step = (
            "Run selector/verifier rescoring and the same-recipe selector-conversion gate before judging "
            "architecture lift."
        )
    elif not architecture_or_conversion_passed:
        status = "needs_architecture_recovery"
        next_step = "Improve recurrent training or recipe before using the artifact as architecture evidence."
    elif not export_present or not export_has_hash:
        status = "needs_hf_export"
        next_step = "Export the recovered recurrent adapter with architecture-gate evidence packaged."
    else:
        status = "ready_for_broader_benchmarks"
        next_step = "Proceed to broader benchmark suite and compare against same-size baselines with the packaged artifact."

    return {
        "run_id": RUN_ID,
        "gate": "stage5_release_benchmark_readiness",
        "status": status,
        "passed": status == "ready_for_broader_benchmarks",
        "next_step": next_step,
        "min_arc_examples": min_arc_examples,
        "criteria": criteria,
        "benchmark_summary": benchmark.get("path"),
        "recipe_control_summary": architecture.get("path"),
        "selector_conversion_summary": selector_conversion.get("path"),
        "hf_export_summary": export.get("path"),
    }


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Release / Benchmark Gate - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Minimum ARC examples: `{payload['min_arc_examples']}`",
        f"- Next step: {payload['next_step']}",
        "",
        "## Criteria",
        "",
        "| Criterion | Passed | Reason | Source |",
        "|---|---:|---|---|",
    ]
    for row in payload["criteria"]:
        evidence = row.get("evidence") or {}
        lines.append(
            f"| `{row['name']}` | `{row['passed']}` | {row['reason']} | `{evidence_source(evidence)}` |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_summary")
    parser.add_argument("--recipe_control_summary")
    parser.add_argument("--selector_conversion_summary")
    parser.add_argument("--hf_export_summary")
    parser.add_argument("--stage5_root", default="outputs/stage5")
    parser.add_argument("--hf_exports_root", default="outputs/hf_exports")
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    parser.add_argument("--min_arc_examples", type=int, default=int(os.environ.get("STAGE5_RELEASE_MIN_ARC_EXAMPLES", "100")))
    args = parser.parse_args()

    stage5_root = resolve_path(args.stage5_root)
    hf_root = resolve_path(args.hf_exports_root)
    stage5_files = summary_files(stage5_root)
    hf_files = summary_files(hf_root)

    benchmark_summary = (
        resolve_path(args.benchmark_summary)
        if args.benchmark_summary
        else latest_matching(stage5_files, is_recovered_benchmark)
    )
    recipe_control_summary = (
        resolve_path(args.recipe_control_summary)
        if args.recipe_control_summary
        else latest_matching(stage5_files, is_recipe_control)
    )
    selector_conversion_summary = (
        resolve_path(args.selector_conversion_summary)
        if args.selector_conversion_summary
        else latest_matching(stage5_files, is_recipe_selector_conversion)
    )
    hf_export_summary = (
        resolve_path(args.hf_export_summary)
        if args.hf_export_summary
        else latest_matching(hf_files, is_hf_export)
    )
    output_dir = ROOT / "outputs" / "stage5" / RUN_ID
    output_json = resolve_path(args.output_json) if args.output_json else output_dir / "summary.json"
    output_md = resolve_path(args.output_md) if args.output_md else output_dir / "summary.md"

    payload = assess_release_gate(
        benchmark_summary=benchmark_summary,
        recipe_control_summary=recipe_control_summary,
        selector_conversion_summary=selector_conversion_summary,
        hf_export_summary=hf_export_summary,
        min_arc_examples=args.min_arc_examples,
    )
    write_report(payload, output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
