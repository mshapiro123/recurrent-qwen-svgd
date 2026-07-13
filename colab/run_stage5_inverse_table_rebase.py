"""Continue the successful inverse-table control through deterministic caps 3 and 4."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_inverse_composition_staircase import (  # noqa: E402
    PHASE1_STEP_ENVELOPE,
    _guardrail_receipt,
    _prepare_guardrail_data,
    _publish,
    _restore_stage_checkpoint,
    _run_diagonal,
    _run_stage,
    build_matched_data,
    read_json,
    read_jsonl,
    write_json,
)
from colab.run_stage5_natural_surface_transfer import restore_checkpoint  # noqa: E402
from colab.stage5_chain_consolidation_utils import path_for_cli  # noqa: E402
from colab.stage5_n24_rung import tier1_canary_verdict  # noqa: E402


SOURCE_SUMMARY = (
    ROOT
    / "outputs/stage5/stage5_inverse_composition_staircase_20260713/summary.json"
)
SOURCE_CAP2_SHA256 = "bc1de1cd7d2a7acf30b9217c8d7054d805888c341b942ff0dab7691b4f995b01"
REBASE_CAPS = (3, 4)


def assert_source_cap2(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("kind") != "stage5_inverse_composition_staircase":
        raise RuntimeError("Rebase source is not an inverse-composition staircase receipt")
    stages = list(((payload.get("arms") or {}).get("C") or {}).get("stages") or [])
    stage = next((item for item in stages if int(item.get("cap", 0)) == 2), None)
    if stage is None:
        raise RuntimeError("Rebase source has no control cap-2 stage")
    checks = {
        "gate": bool((stage.get("gate") or {}).get("passed")),
        "guardrail": bool((stage.get("synthetic_guardrail") or {}).get("passed")),
        "checkpoint_sha": stage.get("checkpoint_sha256") == SOURCE_CAP2_SHA256,
        "drive_backup": bool(stage.get("checkpoint_drive_backup")),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Control cap-2 source is not the locked green checkpoint: {checks}")
    return stage


def rebase_decision(stages: list[dict[str, Any]], *, natural_canary_green: bool = True) -> str:
    by_cap = {int(stage.get("cap", 0)): stage for stage in stages}
    if not all(cap in by_cap for cap in REBASE_CAPS):
        return "rebase_incomplete"
    green = all(
        bool((by_cap[cap].get("gate") or {}).get("passed"))
        and bool((by_cap[cap].get("synthetic_guardrail") or {}).get("passed"))
        for cap in REBASE_CAPS
    )
    if green and natural_canary_green:
        return "rebase_caps3_4_green_pending_review"
    return "rebase_blocked"


def _write_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", payload)
    lines = [
        f"# Inverse-Table Rebase - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source cap-2 SHA256: `{payload['source_cap2']['checkpoint_sha256']}`",
        f"- Decision: `{payload.get('decision', 'pending')}`",
        f"- Phase G-alpha: `{payload.get('phase_g_alpha_status')}`",
        "",
        "| Cap | Correct | Passed | Weighted labels to bar | Synthetic guardrail |",
        "|---:|---:|---|---:|---|",
    ]
    for stage in payload.get("stages", []):
        gate = stage.get("gate") or {}
        dose = stage.get("dose_to_bar") or {}
        guardrail = stage.get("synthetic_guardrail") or {}
        lines.append(
            f"| {stage['cap']} | {gate.get('correct', 0)}/{gate.get('total', 0)} | "
            f"{gate.get('passed', False)} | {dose.get('newest_weighted_active_labels')} | "
            f"{guardrail.get('passed', False)} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _restore_source_cap2(run_dir: Path, stage: dict[str, Any]) -> Path:
    checkpoint, receipt = restore_checkpoint(
        [stage.get("checkpoint_drive_backup"), stage.get("checkpoint")],
        run_dir / "restored" / "locked_inverse_table_cap2.pt",
        label="inverse_table_rebase_source_cap2",
    )
    if receipt["selected_checkpoint_sha256"] != SOURCE_CAP2_SHA256:
        raise RuntimeError("Restored inverse-table cap-2 checkpoint SHA mismatch")
    print(f"[assert-ok] inverse_table_cap2_sha256={SOURCE_CAP2_SHA256}", flush=True)
    return checkpoint


def main() -> int:
    run_id = os.environ.get("STAGE5_REBASE_RUN_ID") or time.strftime(
        "stage5_inverse_table_rebase_caps3_4_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs/stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(os.environ.get("STAGE5_REBASE_SOURCE_SUMMARY", str(SOURCE_SUMMARY)))
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source_payload = read_json(source_path)
    source_stage = assert_source_cap2(source_payload)
    source_checkpoint = _restore_source_cap2(run_dir, source_stage)

    summary_path = run_dir / "summary.json"
    payload = read_json(summary_path) if summary_path.exists() else {
        "kind": "stage5_inverse_table_rebase",
        "run_id": run_id,
        "status": "started",
        "source_summary": path_for_cli(source_path),
        "source_cap2": {
            "checkpoint_sha256": SOURCE_CAP2_SHA256,
            "gate": source_stage["gate"],
            "synthetic_guardrail": source_stage["synthetic_guardrail"],
        },
        "optimizer": "adamw",
        "caps": list(REBASE_CAPS),
        "stages": [],
        "phase_g_alpha_status": "closed_pending_rebased_deterministic_gate",
    }

    if "datasets" not in payload:
        payload["datasets"] = build_matched_data(run_dir / "data")
        guardrails = _prepare_guardrail_data(run_dir)
        payload["guardrail_data"] = {key: path_for_cli(value) for key, value in guardrails.items()}
        payload["status"] = "data_ready"
        _write_summary(run_dir, payload)
        _publish(run_dir, f"Record inverse-table rebase data {run_id} [skip ci]")
    guardrail_paths = {key: ROOT / value for key, value in payload["guardrail_data"].items()}
    train_rows = read_jsonl(run_dir / "data" / "train_C_inverse_table.jsonl")
    test_jsonl = run_dir / "data" / "test_C_inverse_table.jsonl"

    current = source_checkpoint
    if payload["stages"]:
        current = _restore_stage_checkpoint(run_dir, "C_rebase", payload["stages"][-1])
    spent = sum(int(stage.get("optimizer_steps_spent", 0)) for stage in payload["stages"])

    for cap in REBASE_CAPS:
        existing = next((stage for stage in payload["stages"] if int(stage["cap"]) == cap), None)
        if existing is not None:
            current = _restore_stage_checkpoint(run_dir, "C_rebase", existing)
            if not bool((existing.get("gate") or {}).get("passed")):
                break
            continue
        stage = _run_stage(
            run_dir,
            arm_name="C_rebase",
            cap=cap,
            checkpoint=current,
            train_rows=train_rows,
            test_jsonl=test_jsonl,
            phase_remaining=PHASE1_STEP_ENVELOPE - spent,
            seed=81_001 + cap * 100 + 1,
        )
        if not stage.get("checkpoint_drive_backup"):
            stage["synthetic_guardrail"] = {"passed": False, "status": "no_checkpoint"}
        else:
            current = _restore_stage_checkpoint(run_dir, "C_rebase", stage)
            stage["synthetic_guardrail"] = _guardrail_receipt(
                run_dir,
                label=f"C_rebase_cap{cap}_synthetic",
                checkpoint=current,
                data_jsonl=guardrail_paths["synthetic"],
            )
        if not bool((stage.get("synthetic_guardrail") or {}).get("passed")):
            stage["status"] = "blocked_synthetic_guardrail"
            stage["gate"]["passed"] = False
        payload["stages"].append(stage)
        spent += int(stage.get("optimizer_steps_spent", 0))
        payload["decision"] = rebase_decision(payload["stages"])
        payload["status"] = f"cap{cap}_{stage['status']}"
        _write_summary(run_dir, payload)
        _publish(run_dir, f"Record inverse-table rebase cap {cap} {run_id} [skip ci]")
        if not bool((stage.get("gate") or {}).get("passed")):
            break

    stage_green = rebase_decision(payload["stages"]) == "rebase_caps3_4_green_pending_review"
    if stage_green and (payload.get("natural_canary") or {}).get("verdict"):
        natural_green = payload["natural_canary"]["verdict"]["status"] != "red_hard_stop"
        print("inverse_table_rebase_natural_canary=already_completed", flush=True)
    elif stage_green:
        natural = _run_diagonal(
            run_dir,
            label="C_rebase_caps3_4_natural_canary",
            checkpoint=current,
            data_jsonl=guardrail_paths["natural"],
            max_depth=8,
            value_prefix="name:",
        )
        baseline_accuracy = float(source_payload["tier1_canary_baseline"]["accuracy"])
        delta = float(natural["accuracy"]) - baseline_accuracy
        verdict = tier1_canary_verdict(accuracy_delta=delta, ppl_relative_delta=None)
        payload["natural_canary"] = {
            "baseline_accuracy": baseline_accuracy,
            "candidate_accuracy": float(natural["accuracy"]),
            "accuracy_delta": delta,
            "verdict": verdict,
            "summary": path_for_cli(run_dir / "guardrails/C_rebase_caps3_4_natural_canary/summary.json"),
        }
        natural_green = verdict["status"] != "red_hard_stop"
    else:
        natural_green = False
        payload["natural_canary"] = {"status": "not_run_stage_gate_failed"}

    payload["decision"] = rebase_decision(payload["stages"], natural_canary_green=natural_green)
    payload["status"] = payload["decision"]
    _write_summary(run_dir, payload)
    _publish(run_dir, f"Finish inverse-table rebase {run_id} [skip ci]")
    return 0 if payload["decision"] == "rebase_caps3_4_green_pending_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
