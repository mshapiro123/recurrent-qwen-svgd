"""Build a conservative Stage 5 claim/readiness packet from saved artifacts.

This is a no-GPU synthesis step. It does not make benchmark claims by itself;
it gathers the release gate, broader benchmark gate, replicated Gate 1 selector
evidence, Gate 2 particle-mechanism evidence, same-recipe architecture gate, HF
export metadata, and any future authoritative ARC-AGI comparison into one
auditable packet.
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


def is_recipe_selector_conversion_gate(payload: dict[str, Any]) -> bool:
    return (
        payload.get("gate") == "stage5_same_recipe_selector_conversion"
        or payload.get("kind") == "recipe_selector_conversion"
    )


def is_selector_replication_gate(payload: dict[str, Any]) -> bool:
    return (
        payload.get("gate") == "stage5_selector_replication"
        or payload.get("kind") == "selector_replication"
    )


def is_particle_mechanism_gate(payload: dict[str, Any]) -> bool:
    return payload.get("gate") == "stage5_gate2_particle_mechanism"


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


def normalized_ref(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("\\", "/")
    if not text:
        return None
    while text.startswith("./"):
        text = text[2:]
    root = str(ROOT).replace("\\", "/")
    if text.startswith(root + "/"):
        text = text[len(root) + 1 :]
    return text


def first_ref(*values: Any) -> str | None:
    for value in values:
        ref = normalized_ref(value)
        if ref:
            return ref
    return None


def hf_export_checkpoint_ref(hf_export: dict[str, Any]) -> str | None:
    summary = hf_export.get("summary") or {}
    metadata = summary.get("metadata") or {}
    return first_ref(summary.get("checkpoint"), metadata.get("checkpoint_source_path"))


def hf_export_source_summary_ref(hf_export: dict[str, Any]) -> str | None:
    metadata = (hf_export.get("summary") or {}).get("metadata") or {}
    source = metadata.get("source") or {}
    return first_ref(source.get("summary_path"))


def arc_agi_candidate(arc_agi: dict[str, Any]) -> dict[str, Any]:
    candidate = (arc_agi.get("summary") or {}).get("candidate")
    return candidate if isinstance(candidate, dict) else {}


def arc_agi_candidate_checkpoint_ref(arc_agi: dict[str, Any]) -> str | None:
    candidate = arc_agi_candidate(arc_agi)
    metadata = candidate.get("metadata") or {}
    return first_ref(
        metadata.get("recovered_checkpoint"),
        metadata.get("selected_checkpoint"),
        metadata.get("tuned_checkpoint"),
        metadata.get("final_checkpoint"),
        metadata.get("checkpoint"),
        metadata.get("checkpoint_path"),
    )


def arc_agi_candidate_summary_ref(arc_agi: dict[str, Any]) -> str | None:
    return first_ref(arc_agi_candidate(arc_agi).get("path"))


def sota_export_linkage(hf_export: dict[str, Any], arc_agi: dict[str, Any]) -> dict[str, Any]:
    """Verify the SOTA comparison is for the exported recurrent artifact."""

    hf_checkpoint = hf_export_checkpoint_ref(hf_export)
    hf_source_summary = hf_export_source_summary_ref(hf_export)
    candidate_checkpoint = arc_agi_candidate_checkpoint_ref(arc_agi)
    candidate_summary = arc_agi_candidate_summary_ref(arc_agi)

    evidence = {
        "hf_checkpoint": hf_checkpoint,
        "candidate_checkpoint": candidate_checkpoint,
        "hf_source_summary": hf_source_summary,
        "candidate_summary": candidate_summary,
        "matched_on": None,
        "verified": False,
    }

    if not (hf_export.get("present") and arc_agi.get("present") and arc_agi.get("passed")):
        return {
            **evidence,
            "passed": False,
            "reason": "SOTA export linkage requires both an HF export and a passed ARC-AGI comparison.",
        }

    if hf_checkpoint and candidate_checkpoint:
        passed = hf_checkpoint == candidate_checkpoint
        return {
            **evidence,
            "matched_on": "checkpoint",
            "verified": True,
            "passed": passed,
            "reason": (
                "HF export checkpoint matches the ARC-AGI candidate checkpoint."
                if passed
                else "HF export checkpoint does not match the ARC-AGI candidate checkpoint."
            ),
        }

    if hf_source_summary and candidate_summary:
        passed = hf_source_summary == candidate_summary
        return {
            **evidence,
            "matched_on": "source_summary",
            "verified": True,
            "passed": passed,
            "reason": (
                "HF export source summary matches the ARC-AGI candidate summary."
                if passed
                else "HF export source summary does not match the ARC-AGI candidate summary."
            ),
        }

    return {
        **evidence,
        "passed": False,
        "reason": "Could not verify that the HF export and ARC-AGI SOTA comparison refer to the same artifact.",
    }


def evidence_source(evidence: dict[str, Any]) -> str | None:
    if evidence.get("path"):
        return str(evidence["path"])
    sources = []
    for key in ("architecture", "selector_conversion"):
        nested = evidence.get(key)
        if isinstance(nested, dict) and nested.get("path"):
            sources.append(f"{key}:{nested['path']}")
    return ", ".join(sources) if sources else None


def build_claim_packet(
    *,
    release_gate_summary: Path | None,
    broader_benchmark_summary: Path | None,
    recipe_control_summary: Path | None,
    hf_export_summary: Path | None,
    arc_agi_comparison_summary: Path | None,
    selector_conversion_summary: Path | None = None,
    selector_replication_summary: Path | None = None,
    particle_mechanism_summary: Path | None = None,
) -> dict[str, Any]:
    release_gate = artifact(release_gate_summary)
    broader_benchmark = artifact(broader_benchmark_summary)
    recipe_control = artifact(recipe_control_summary)
    selector_conversion = artifact(selector_conversion_summary)
    selector_replication = artifact(selector_replication_summary)
    particle_mechanism = artifact(particle_mechanism_summary)
    hf_export = artifact(hf_export_summary)
    arc_agi = artifact(arc_agi_comparison_summary)

    release_passed = bool(release_gate.get("passed"))
    broader_passed = bool(broader_benchmark.get("passed"))
    selector_replication_passed = bool(selector_replication.get("passed"))
    particle_mechanism_passed = bool(particle_mechanism.get("passed"))
    recipe_passed = bool(recipe_control.get("passed"))
    selector_conversion_passed = bool(selector_conversion.get("passed"))
    architecture_or_conversion_passed = recipe_passed or selector_conversion_passed
    export_has_hash = bool((hf_export.get("summary") or {}).get("metadata", {}).get("checkpoint_sha256"))
    arc_agi_present = bool(arc_agi.get("present"))
    arc_agi_passed = bool(arc_agi.get("passed"))
    sota_linkage = sota_export_linkage(hf_export, arc_agi)
    sota_linkage_passed = bool(sota_linkage.get("passed"))

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
            "same_recipe_architecture_or_selector_conversion_passed",
            architecture_or_conversion_passed,
            (
                "Same-recipe recurrent-vs-dense architecture gate passed."
                if recipe_passed
                else (
                    "Same-recipe selector-conversion gate passed."
                    if selector_conversion_passed
                    else "Same-recipe architecture or selector-conversion evidence has not passed."
                )
            ),
            {"architecture": recipe_control, "selector_conversion": selector_conversion},
        ),
        criterion(
            "selector_replication_passed",
            selector_replication_passed,
            (
                "Gate 1 selector/TTA evidence replicated on a second slice."
                if selector_replication_passed
                else "Replicated Gate 1 selector/TTA evidence is missing or has not passed."
            ),
            selector_replication,
        ),
        criterion(
            "particle_mechanism_gate_passed",
            particle_mechanism_passed,
            (
                "Gate 2 recurrent-particle mechanism evidence passed."
                if particle_mechanism_passed
                else "Gate 2 recurrent-particle mechanism evidence is missing or has not passed."
            ),
            particle_mechanism,
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
        criterion(
            "sota_candidate_matches_hf_export",
            sota_linkage_passed,
            str(sota_linkage.get("reason")),
            sota_linkage,
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
    elif not selector_replication_passed:
        status = "needs_selector_replication"
        claim_level = "not_ready"
        next_step = "Replicate the Gate 1 selector/TTA setting on a distinct stratified slice."
    elif not particle_mechanism_passed:
        status = "needs_particle_mechanism_gate"
        claim_level = "not_ready"
        next_step = "Pass the Gate 2 recurrent-particle mechanism assessment before architecture-facing claims."
    elif not architecture_or_conversion_passed:
        status = "needs_architecture_evidence"
        claim_level = "not_ready"
        next_step = "Repair same-recipe recurrent-vs-dense architecture or selector-conversion evidence before architecture claims."
    elif not bool(hf_export.get("present")) or not export_has_hash:
        status = "needs_hf_export"
        claim_level = "not_ready"
        next_step = "Export the recurrent adapter with checkpoint hash metadata."
    elif not arc_agi_present or not arc_agi_passed:
        status = "ready_for_release_candidate_not_sota"
        claim_level = "release_candidate"
        next_step = "Write the release-candidate report, but do not claim SOTA until authoritative ARC-AGI comparison evidence exists."
    elif not sota_linkage_passed:
        status = "ready_for_release_candidate_needs_sota_export_linkage"
        claim_level = "release_candidate"
        next_step = (
            "Do not claim SOTA until the HF export and ARC-AGI comparison are linked to the same checkpoint "
            "or source summary."
        )
    else:
        status = "sota_claim_ready"
        claim_level = "sota_candidate"
        next_step = "Draft the SOTA claim with the ARC-AGI comparison evidence and limitations."

    return {
        "run_id": RUN_ID,
        "gate": "stage5_claim_readiness",
        "status": status,
        "passed": status
        in {
            "ready_for_release_candidate_not_sota",
            "ready_for_release_candidate_needs_sota_export_linkage",
            "sota_claim_ready",
        },
        "claim_level": claim_level,
        "next_step": next_step,
        "artifacts": {
            "release_gate": release_gate,
            "broader_benchmark_gate": broader_benchmark,
            "same_recipe_architecture": recipe_control,
            "same_recipe_selector_conversion": selector_conversion,
            "selector_replication": selector_replication,
            "particle_mechanism_gate": particle_mechanism,
            "hf_export": hf_export,
            "authoritative_arc_agi_comparison": arc_agi,
            "sota_export_linkage": sota_linkage,
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
            f"(source `{evidence_source(evidence)}`)"
        )
    lines.extend(
        [
            "",
            "## Claim Guardrail",
            "",
            "This packet separates release-candidate evidence from a SOTA ARC-AGI claim. "
            "A SOTA claim requires an authoritative ARC-AGI comparison artifact against similarly sized models, "
            "linked to the same checkpoint or source summary as the exported HF artifact.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release_gate_summary")
    parser.add_argument("--broader_benchmark_summary")
    parser.add_argument("--recipe_control_summary")
    parser.add_argument("--selector_conversion_summary")
    parser.add_argument("--selector_replication_summary")
    parser.add_argument("--particle_mechanism_summary")
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
    selector_conversion = (
        resolve_path(args.selector_conversion_summary)
        if args.selector_conversion_summary
        else latest_matching(stage5_files, is_recipe_selector_conversion_gate)
    )
    selector_replication = (
        resolve_path(args.selector_replication_summary)
        if args.selector_replication_summary
        else latest_matching(stage5_files, is_selector_replication_gate)
    )
    particle_mechanism = (
        resolve_path(args.particle_mechanism_summary)
        if args.particle_mechanism_summary
        else latest_matching(stage5_files, is_particle_mechanism_gate)
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
        selector_conversion_summary=selector_conversion,
        selector_replication_summary=selector_replication,
        particle_mechanism_summary=particle_mechanism,
    )
    write_report(payload, output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
