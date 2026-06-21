"""Build a conservative Stage 5 claim/readiness packet from saved artifacts.

This is a no-GPU synthesis step. It does not make benchmark claims by itself;
it gathers the release gate, broader benchmark gate, same-recipe architecture
gate, HF export metadata, and any future authoritative ARC-AGI comparison into
one auditable packet.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_CLAIM_PACKET_RUN_ID") or time.strftime("stage5_claim_packet_%Y%m%d_%H%M%S")


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


def summary_files(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.glob("**/summary.json"))
    return sorted(set(files), key=lambda path: str(path))


def latest_matching(paths: list[Path], predicate: Callable[[dict[str, Any]], bool]) -> Path | None:
    matches: list[Path] = []
    for path in paths:
        payload = safe_read_json(path)
        if payload and predicate(payload):
            matches.append(path)
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def is_release_gate(payload: dict[str, Any]) -> bool:
    return payload.get("gate") == "stage5_release_benchmark_readiness"


def is_broader_benchmark_gate(payload: dict[str, Any]) -> bool:
    return payload.get("gate") == "stage5_broader_benchmark_suite"


def is_recipe_control_gate(payload: dict[str, Any]) -> bool:
    return payload.get("gate") == "stage5_same_recipe_architecture"


def is_hf_export(payload: dict[str, Any]) -> bool:
    return bool(payload.get("export_dir") and payload.get("checkpoint") and isinstance(payload.get("metadata"), dict))


def is_authoritative_arc_agi_comparison(payload: dict[str, Any]) -> bool:
    return payload.get("gate") == "stage5_arc_agi_sota_comparison" or payload.get("kind") == "arc_agi_sota_comparison"


def artifact(path: Path | None) -> dict[str, Any]:
    if not path:
        return {"present": False}
    payload = safe_read_json(path) or {}
    return {
        "present": True,
        "path": path_for_cli(path),
        "run_id": str(payload.get("run_id") or path.parent.name),
        "status": payload.get("status"),
        "passed": bool(payload.get("passed", False)),
        "next_step": payload.get("next_step"),
        "checkpoint": payload.get("checkpoint"),
        "hf_repo_id": payload.get("hf_repo_id"),
        "summary": payload,
    }


def criterion(name: str, passed: bool, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "reason": reason, "evidence": evidence}


def build_claim_packet(
    *,
    release_gate_summary: Path | None,
    broader_benchmark_summary: Path | None,
    recipe_control_summary: Path | None,
    hf_export_summary: Path | None,
    arc_agi_comparison_summary: Path | None,
) -> dict[str, Any]:
    release_gate = artifact(release_gate_summary)
    broader_benchmark = artifact(broader_benchmark_summary)
    recipe_control = artifact(recipe_control_summary)
    hf_export = artifact(hf_export_summary)
    arc_agi = artifact(arc_agi_comparison_summary)

    release_passed = bool(release_gate.get("passed"))
    broader_passed = bool(broader_benchmark.get("passed"))
    recipe_passed = bool(recipe_control.get("passed"))
    export_has_hash = bool((hf_export.get("summary") or {}).get("metadata", {}).get("checkpoint_sha256"))
    arc_agi_present = bool(arc_agi.get("present"))
    arc_agi_passed = bool(arc_agi.get("passed"))

    criteria = [
        criterion(
            "release_gate_passed",
            release_passed,
            "Stage 5 release/benchmark gate passed." if release_passed else "Stage 5 release/benchmark gate has not passed.",
            release_gate,
        ),
        criterion(
            "broader_benchmark_gate_passed",
            broader_passed,
            (
                "Broader ARC-Challenge/GPQA-lite paired benchmark gate passed."
                if broader_passed
                else "Broader paired benchmark gate has not passed."
            ),
            broader_benchmark,
        ),
        criterion(
            "same_recipe_architecture_passed",
            recipe_passed,
            (
                "Same-recipe recurrent-vs-dense architecture gate passed."
                if recipe_passed
                else "Same-recipe architecture gate has not passed."
            ),
            recipe_control,
        ),
        criterion(
            "hf_export_hash_present",
            bool(hf_export.get("present")) and export_has_hash,
            (
                "HF export artifact exists with checkpoint hash metadata."
                if bool(hf_export.get("present")) and export_has_hash
                else "HF export artifact with checkpoint hash metadata is missing."
            ),
            hf_export,
        ),
        criterion(
            "authoritative_arc_agi_sota_comparison",
            arc_agi_present and arc_agi_passed,
            (
                "Authoritative ARC-AGI same-size SOTA comparison passed."
                if arc_agi_present and arc_agi_passed
                else "No authoritative ARC-AGI same-size SOTA comparison proves a SOTA claim."
            ),
            arc_agi,
        ),
    ]

    if not release_passed:
        status = "needs_release_gate"
        claim_level = "not_ready"
        next_step = "Run or satisfy the Stage 5 release/benchmark gate."
    elif not broader_passed:
        status = "needs_broader_benchmark_gate"
        claim_level = "not_ready"
        next_step = "Run the broader benchmark suite and paired assessment gate."
    elif not recipe_passed:
        status = "needs_architecture_evidence"
        claim_level = "not_ready"
        next_step = "Repair same-recipe recurrent-vs-dense architecture evidence before architecture claims."
    elif not bool(hf_export.get("present")) or not export_has_hash:
        status = "needs_hf_export"
        claim_level = "not_ready"
        next_step = "Export the recurrent adapter with checkpoint hash metadata."
    elif not arc_agi_present or not arc_agi_passed:
        status = "ready_for_release_candidate_not_sota"
        claim_level = "release_candidate"
        next_step = "Write the release-candidate report, but do not claim SOTA until authoritative ARC-AGI comparison evidence exists."
    else:
        status = "sota_claim_ready"
        claim_level = "sota_candidate"
        next_step = "Draft the SOTA claim with the ARC-AGI comparison evidence and limitations."

    return {
        "run_id": RUN_ID,
        "gate": "stage5_claim_readiness",
        "status": status,
        "passed": status in {"ready_for_release_candidate_not_sota", "sota_claim_ready"},
        "claim_level": claim_level,
        "next_step": next_step,
        "artifacts": {
            "release_gate": release_gate,
            "broader_benchmark_gate": broader_benchmark,
            "same_recipe_architecture": recipe_control,
            "hf_export": hf_export,
            "authoritative_arc_agi_comparison": arc_agi,
        },
        "criteria": criteria,
    }


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Claim Readiness Packet - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Claim level: `{payload['claim_level']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Next step: {payload['next_step']}",
        "",
        "## Criteria",
        "",
    ]
    for row in payload["criteria"]:
        evidence = row.get("evidence") or {}
        lines.append(
            f"- `{row['name']}` passed `{row['passed']}`: {row['reason']} "
            f"(source `{evidence.get('path')}`)"
        )
    lines.extend(
        [
            "",
            "## Claim Guardrail",
            "",
            "This packet separates release-candidate evidence from a SOTA ARC-AGI claim. "
            "A SOTA claim requires an authoritative ARC-AGI comparison artifact against similarly sized models.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release_gate_summary")
    parser.add_argument("--broader_benchmark_summary")
    parser.add_argument("--recipe_control_summary")
    parser.add_argument("--hf_export_summary")
    parser.add_argument("--arc_agi_comparison_summary")
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    args = parser.parse_args()

    stage5_files = summary_files(ROOT / "outputs" / "stage5")
    hf_files = summary_files(ROOT / "outputs" / "hf_exports")
    release_gate = resolve_path(args.release_gate_summary) if args.release_gate_summary else latest_matching(stage5_files, is_release_gate)
    broader = (
        resolve_path(args.broader_benchmark_summary)
        if args.broader_benchmark_summary
        else latest_matching(stage5_files, is_broader_benchmark_gate)
    )
    recipe = (
        resolve_path(args.recipe_control_summary)
        if args.recipe_control_summary
        else latest_matching(stage5_files, is_recipe_control_gate)
    )
    hf_export = resolve_path(args.hf_export_summary) if args.hf_export_summary else latest_matching(hf_files, is_hf_export)
    arc_agi = (
        resolve_path(args.arc_agi_comparison_summary)
        if args.arc_agi_comparison_summary
        else latest_matching(stage5_files, is_authoritative_arc_agi_comparison)
    )

    output_json = resolve_path(args.output_json) if args.output_json else ROOT / "outputs" / "stage5" / RUN_ID / "summary.json"
    output_md = resolve_path(args.output_md) if args.output_md else output_json.with_suffix(".md")
    payload = build_claim_packet(
        release_gate_summary=release_gate,
        broader_benchmark_summary=broader,
        recipe_control_summary=recipe,
        hf_export_summary=hf_export,
        arc_agi_comparison_summary=arc_agi,
    )
    write_report(payload, output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
