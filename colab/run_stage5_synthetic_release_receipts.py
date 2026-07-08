"""Consolidate Stage 5 synthetic-depth receipts into one release-status artifact."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.stage5_chain_consolidation_utils import ROOT, path_for_cli, publish_run, read_json, write_json  # noqa: E402


DEFAULTS = {
    "support6_receipts": "outputs/stage5/stage5_support6_replication_receipts_20260708_003055/summary.json",
    "support6_dosed": "outputs/stage5/stage5_support6_dosed_seed_resolution_latest/summary.json",
    "support8_same_reader": "outputs/stage5/stage5_same_reader_final_symbol_20260707_021010/summary.json",
    "n24_summary": "outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json",
    "scorer_equivalence": "outputs/stage5/stage5_scorer_equivalence_20260708_003132/summary.json",
    "regression_battery": "outputs/stage5/stage5_regression_battery_loop1_current/summary.json",
}


def maybe_read(path: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    candidate = ROOT / Path(path)
    if not candidate.exists():
        return None, f"missing: {path}"
    return json.loads(candidate.read_text(encoding="utf-8")), None


def find_latest(pattern: str) -> str | None:
    matches = sorted((ROOT / "outputs" / "stage5").glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        return None
    latest = matches[-1]
    summary = latest if latest.name == "summary.json" else latest / "summary.json"
    return path_for_cli(summary)


def recursive_key_hits(payload: Any, needle: str) -> list[str]:
    hits: list[str] = []

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if needle.lower() in str(key).lower():
                    hits.append(path)
                walk(child, path)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                walk(child, f"{prefix}[{idx}]")

    walk(payload, "")
    return hits


def compact_support6(payload: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    if payload is None:
        return {"status": "missing", "error": error}
    return {
        "status": payload.get("status"),
        "decision": payload.get("decision"),
        "frontier_policy": payload.get("frontier_policy"),
        "runs": [
            {
                "label": item.get("label"),
                "run_id": item.get("run_id"),
                "canonical_frontier": item.get("canonical_frontier"),
                "canonical_frontier_pass": item.get("canonical_frontier_pass"),
                "selected_correct": item.get("selected_correct"),
            }
            for item in payload.get("runs", [])
        ],
    }


def compact_dosed(payload: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    if payload is None:
        return {"status": "pending", "error": error}
    failed_count = len(payload.get("failed_replicates") or [])
    completed_count = len(payload.get("results") or [])
    status = payload.get("status")
    if failed_count and completed_count < failed_count:
        status = "dosed_seed_resolution_running"
    return {
        "status": status,
        "resolution_summary": payload.get("resolution_summary"),
        "failed_replicate_count": failed_count,
        "completed_result_count": completed_count,
        "all_expected_completed": (not failed_count) or completed_count >= failed_count,
        "decision": payload.get("decision"),
        "results": [
            {
                "label": item.get("label"),
                "seed": item.get("seed"),
                "pre_frontier": (item.get("pre_dose") or {}).get("canonical_frontier"),
                "post_frontier": (item.get("post_dose") or {}).get("canonical_frontier"),
                "post_pass": (item.get("post_dose") or {}).get("canonical_frontier_pass"),
            }
            for item in payload.get("results", [])
        ],
    }


def compact_n24(payload: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    if payload is None:
        return {"status": "missing", "error": error}
    evals = payload.get("checkpoint_evals") or []
    final_eval = evals[-1] if evals else {}
    score = final_eval.get("score") or {}
    canary_hits = recursive_key_hits(payload, "canary")
    return {
        "status": payload.get("status"),
        "run_id": payload.get("run_id"),
        "final_checkpoint": payload.get("final_checkpoint"),
        "final_checkpoint_drive_backup": payload.get("final_checkpoint_drive_backup"),
        "final_step": final_eval.get("step"),
        "final_verdict": score.get("verdict"),
        "overall_pass": score.get("overall_pass"),
        "selected_correct": score.get("selected_correct"),
        "nonregression_pass": score.get("nonregression_pass"),
        "skipped_checkpoints": payload.get("skipped_checkpoints") or [],
        "canary_policy": payload.get("canary_policy"),
        "canary_fields_present": canary_hits,
        "archive_note": (
            "Step 4000 was unavailable in the resume artifact; final step 6000 and ramp step 2000 are preserved."
            if payload.get("skipped_checkpoints")
            else "No skipped checkpoint note in summary."
        ),
    }


def compact_scorer_equiv(payload: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    if payload is None:
        return {"status": "missing", "error": error}
    equiv = payload.get("equivalence") or {}
    return {
        "status": payload.get("status"),
        "run_id": payload.get("run_id"),
        "pass": equiv.get("pass"),
        "records_checked": equiv.get("records_checked"),
        "mismatch_count": equiv.get("mismatch_count"),
        "loop_counts": equiv.get("loop_counts"),
    }


def compact_same_reader(payload: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    if payload is None:
        return {"status": "missing", "error": error}
    same = payload.get("same_reader_final") or {}
    return {
        "status": payload.get("status"),
        "run_id": payload.get("run_id"),
        "source_summary": payload.get("source_summary"),
        "max_loops": payload.get("max_loops"),
        "same_reader_total": same.get("same_reader_total"),
        "metric_policy": same.get("metric_policy") or payload.get("metric_policy"),
        "release_gate": payload.get("release_gate"),
    }


def compact_regression(payload: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    if payload is None:
        return {"status": "missing", "error": error}
    policy = payload.get("policy") or {}
    return {
        "status": payload.get("status"),
        "run_id": payload.get("run_id"),
        "source_summaries": payload.get("source_summaries"),
        "assessments": [
            {
                "label": item.get("label"),
                "status": item.get("status"),
                "assessment_summary": item.get("assessment_summary"),
            }
            for item in payload.get("assessments", [])
        ],
        "tier1_canary_status": policy.get("tier1_canary_status"),
        "hellaswag_winogrande_lambada_status": policy.get("hellaswag_winogrande_lambada_status"),
        "policy": policy,
    }


def release_status(payload: dict[str, Any]) -> str:
    blockers = payload.get("blockers") or []
    pending = payload.get("pending_followups") or []
    if blockers:
        return "release_receipts_blocked"
    if pending:
        return "release_receipts_need_followup"
    return "release_receipts_complete"


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Stage 5 Synthetic Release Receipts - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Support-6 replication: `{payload['receipts']['support6_replication'].get('status')}`",
        f"- Support-6 dosed resolution: `{payload['receipts']['support6_dosed_resolution'].get('status')}`",
        f"- Scorer equivalence: `{payload['receipts']['scorer_equivalence'].get('status')}`",
        f"- N24 final verdict: `{payload['receipts']['n24'].get('final_verdict')}`",
        f"- Same-reader support-8 status: `{payload['receipts']['support8_same_reader'].get('status')}`",
        f"- Regression battery status: `{payload['receipts']['regression_battery'].get('status')}`",
        "",
        "## Blockers",
    ]
    lines.extend(f"- {item}" for item in payload.get("blockers", []) or ["None"])
    lines.extend(["", "## Pending Followups"])
    lines.extend(f"- {item}" for item in payload.get("pending_followups", []) or ["None"])
    lines.extend(["", "## Notes"])
    lines.extend(f"- {item}" for item in payload.get("notes", []))
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_RELEASE_RECEIPTS_RUN_ID") or time.strftime(
        "stage5_synthetic_release_receipts_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    support6_dosed_default = find_latest("stage5_support6_dosed_seed_resolution_*/summary.json")
    sources = {
        "support6_receipts": os.environ.get("STAGE5_RELEASE_SUPPORT6_RECEIPTS", DEFAULTS["support6_receipts"]),
        "support6_dosed": os.environ.get(
            "STAGE5_RELEASE_SUPPORT6_DOSED", support6_dosed_default or DEFAULTS["support6_dosed"]
        ),
        "support8_same_reader": os.environ.get(
            "STAGE5_RELEASE_SUPPORT8_SAME_READER", DEFAULTS["support8_same_reader"]
        ),
        "n24_summary": os.environ.get("STAGE5_RELEASE_N24_SUMMARY", DEFAULTS["n24_summary"]),
        "scorer_equivalence": os.environ.get("STAGE5_RELEASE_SCORER_EQUIV", DEFAULTS["scorer_equivalence"]),
        "regression_battery": os.environ.get("STAGE5_RELEASE_REGRESSION_BATTERY", DEFAULTS["regression_battery"]),
    }
    raw = {name: maybe_read(path) for name, path in sources.items()}
    receipts = {
        "support6_replication": compact_support6(*raw["support6_receipts"]),
        "support6_dosed_resolution": compact_dosed(*raw["support6_dosed"]),
        "support8_same_reader": compact_same_reader(*raw["support8_same_reader"]),
        "n24": compact_n24(*raw["n24_summary"]),
        "scorer_equivalence": compact_scorer_equiv(*raw["scorer_equivalence"]),
        "regression_battery": compact_regression(*raw["regression_battery"]),
    }
    blockers: list[str] = []
    pending: list[str] = []
    notes = [
        "Canonical synthetic frontier is bar_crossing_frontier at accuracy bar 0.71.",
        "MCQ option-text final tables remain suspended for release claims; same-reader final-symbol scoring is the live final metric.",
    ]
    support6_replication_needs_dose = (
        receipts["support6_replication"].get("status") == "replication_needs_dosed_seed_resolution"
    )
    support6_dosed_status = receipts["support6_dosed_resolution"].get("status")
    if support6_replication_needs_dose and support6_dosed_status in {"pending", "missing"}:
        pending.append("Run support6_dosed_seed_resolution before treating support-6 replication as robustness evidence.")
    elif support6_replication_needs_dose and support6_dosed_status == "dosed_seed_resolution_running":
        pending.append("Wait for support6_dosed_seed_resolution to finish before treating support-6 replication as robustness evidence.")
    if support6_dosed_status in {"pending", "missing"}:
        pending.append("Support-6 dosed seed resolution receipt is not present yet.")
    elif support6_dosed_status != "dosed_seed_resolution_pass":
        blockers.append("Support-6 dosed seed resolution did not pass.")
    if receipts["scorer_equivalence"].get("pass") is not True:
        blockers.append("Fast active-label scorer has not been proven equivalent to the slow scorer.")
    if receipts["n24"].get("final_verdict") != "strong_four_point_law":
        blockers.append("N24 final synthetic rung did not pass the strong four-point law gate.")
    if receipts["n24"].get("skipped_checkpoints"):
        pending.append("N24 checkpoint step 4000 is missing; archive note is recorded but the interval checkpoint is unavailable.")
    canary = receipts["n24"].get("canary_policy") or {}
    if not canary.get("provided_external_deltas"):
        pending.append("N24 run recorded the canary policy, but no external Tier-1 canary deltas were provided.")
    if receipts["regression_battery"].get("tier1_canary_status") == "pending":
        pending.append("Natural-text NLL canary is still not wired into the regression battery.")
    if receipts["regression_battery"].get("hellaswag_winogrande_lambada_status") == "pending":
        pending.append("HellaSwag/Winogrande/LAMBADA regression extensions remain pending.")
    if receipts["support8_same_reader"].get("status") != "finished":
        pending.append("Support-8 same-reader final-symbol receipt is missing.")
    pending.append("N24 same-reader final-symbol scoring should be run before any final-symbol release claim.")
    pending.append("Full lineage regression battery across base/recovery/scaled/support6/support8/N24 is not complete yet.")

    payload = {
        "kind": "stage5_synthetic_release_receipts",
        "run_id": run_id,
        "status": "started",
        "sources": sources,
        "receipts": receipts,
        "blockers": blockers,
        "pending_followups": pending,
        "notes": notes,
    }
    payload["status"] = release_status(payload)
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    if os.environ.get("STAGE5_RELEASE_RECEIPTS_PUBLISH", "1").strip().lower() in {"1", "true", "yes", "y", "on"}:
        publish_run(run_dir, message=f"Record Stage 5 synthetic release receipts {run_id} [skip ci]")
    else:
        print("Skipping publish because STAGE5_RELEASE_RECEIPTS_PUBLISH=0", flush=True)
    print(json.dumps({"run_id": run_id, "status": payload["status"], "summary": path_for_cli(run_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
