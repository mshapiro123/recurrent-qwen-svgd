"""Run the zero-shot inverse-rendered non-injective validity gate for G-alpha."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.run_stage5_natural_surface_transfer import restore_checkpoint, run
from colab.run_stage5_inverse_composition_staircase import _publish
from colab.stage5_chain_consolidation_utils import path_for_cli
from training.abductive_injective_task import (
    PhaseGFrozenEvalConfig,
    build_phase_g_frozen_rows,
    row_manifest,
    validate_phase_g_frozen_rows,
    validate_inverse_relation_rows,
    with_inverse_relation_prompt,
    write_jsonl,
)


ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
SOURCE_SUMMARY = ROOT / "outputs/stage5/stage5_inverse_table_rebase_caps3_4_20260713/summary.json"
DATA_ROOT = ROOT / "data/phase_g_alpha_inverse_rendered"
CANONICAL_DATA_ROOT = ROOT / "data/phase_g_alpha"
SOURCE_CAP3_SHA256 = "83767ebff2c2a13a2f15fe8266f605fb8485985c3289c1f1720cd70c122a9ac5"
POOLED_REQUIRED = 288
POOLED_TOTAL = 384
DEPTH_REQUIRED = 58
DEPTH_TOTAL = 96
RETENTION_REQUIRED = 30
RETENTION_TOTAL = 32


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_inverse_rendered_split(
    config: PhaseGFrozenEvalConfig,
    *,
    split: str,
    source_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical = source_rows if source_rows is not None else build_phase_g_frozen_rows(config, split=split)
    source_validation = validate_phase_g_frozen_rows(
        canonical,
        rows_per_stratum=config.rows_per_stratum,
    )
    if source_validation["status"] != "passed":
        raise RuntimeError(f"Canonical {split} validation failed: {source_validation['errors'][:5]}")
    rows = [with_inverse_relation_prompt(row) for row in canonical]
    validation = validate_inverse_relation_rows(rows, rows_per_stratum=config.rows_per_stratum)
    if validation["status"] != "passed":
        raise RuntimeError(f"Inverse-rendered {split} validation failed: {validation['errors'][:5]}")
    return rows, {
        "split": split,
        "manifest": row_manifest(rows),
        "source_manifest": row_manifest(canonical),
        "source_validation": source_validation,
        "validation": validation,
        "rendering": "inverse_relation_given",
    }


def assess_deterministic_validity(summary: dict[str, Any]) -> dict[str, Any]:
    overall = summary.get("overall", {})
    if "greedy_chain_valid" not in overall:
        raise ValueError("Deterministic validity summary is missing greedy_chain_valid")
    pooled_correct = int(overall["greedy_chain_valid"])
    pooled_total = int(overall.get("rows", 0))
    by_depth: dict[str, Any] = {}
    for depth in (1, 2, 3, 4):
        row = summary.get("by_depth", {}).get(str(depth), {})
        if "greedy_chain_valid" not in row:
            raise ValueError(f"Depth {depth} summary is missing greedy_chain_valid")
        correct = int(row["greedy_chain_valid"])
        total = int(row.get("rows", 0))
        by_depth[str(depth)] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
            "required_correct": DEPTH_REQUIRED,
            "required_total": DEPTH_TOTAL,
            "passed": total == DEPTH_TOTAL and correct >= DEPTH_REQUIRED,
        }
    return {
        "pooled": {
            "correct": pooled_correct,
            "total": pooled_total,
            "accuracy": pooled_correct / pooled_total if pooled_total else 0.0,
            "required_correct": POOLED_REQUIRED,
            "required_total": POOLED_TOTAL,
            "passed": pooled_total == POOLED_TOTAL and pooled_correct >= POOLED_REQUIRED,
        },
        "by_depth": by_depth,
        "pass": (
            pooled_total == POOLED_TOTAL
            and pooled_correct >= POOLED_REQUIRED
            and all(row["passed"] for row in by_depth.values())
        ),
    }


def _source_cap3(source: dict[str, Any]) -> dict[str, Any]:
    matches = [stage for stage in source.get("stages", []) if int(stage.get("cap", -1)) == 3]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cap-3 source stage, found {len(matches)}")
    stage = matches[0]
    if stage.get("checkpoint_sha256") != SOURCE_CAP3_SHA256:
        raise RuntimeError(
            f"C cap-3 SHA mismatch: {stage.get('checkpoint_sha256')} != {SOURCE_CAP3_SHA256}"
        )
    if int(stage.get("gate", {}).get("correct", 0)) < 46:
        raise RuntimeError("C cap-3 source did not pass its task bar")
    return stage


def _run_eval(run_dir: Path, *, label: str, checkpoint: Path, data_path: Path) -> dict[str, Any]:
    eval_dir = run_dir / "eval" / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    drive_eval_dir = Path(
        os.environ.get(
            "STAGE5_INVERSE_RENDERED_DRIVE_ROOT",
            f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{run_dir.name}/inverse_rendered",
        )
    ) / label
    run(
        [
            sys.executable,
            "eval/eval_abductive_coverage.py",
            "--data_jsonl",
            path_for_cli(data_path),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(eval_dir / "rows.jsonl"),
            "--output_summary",
            path_for_cli(eval_dir / "summary.json"),
            "--sample_counts",
            "1",
            "--temperature",
            "0.7",
            "--bridge_projection_mode",
            "split",
            "--dtype",
            os.environ.get("STAGE5_INVERSE_RENDERED_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--progress_every",
            "1",
            "--resume",
            "--progress_path",
            path_for_cli(eval_dir / "progress.json"),
            "--resume_source_jsonl",
            path_for_cli(drive_eval_dir / "rows.jsonl"),
            "--backup_output_jsonl",
            path_for_cli(drive_eval_dir / "rows.jsonl"),
            "--backup_progress_path",
            path_for_cli(drive_eval_dir / "progress.json"),
            "--backup_summary",
            path_for_cli(drive_eval_dir / "summary.json"),
        ],
        cwd=ROOT,
    )
    return read_json(eval_dir / "summary.json")


def main() -> int:
    run_id = os.environ.get("STAGE5_INVERSE_RENDERED_RUN_ID") or time.strftime(
        "stage5_inverse_rendered_width_gate_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs/stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_progress_path = run_dir / "progress.json"
    write_json(
        run_progress_path,
        {
            "kind": "stage5_inverse_rendered_width_gate_progress",
            "run_id": run_id,
            "status": "starting",
        },
    )
    config = PhaseGFrozenEvalConfig(rows_per_stratum=128, seed=7_194_203)
    canonical_manifest = read_json(CANONICAL_DATA_ROOT / "manifest.json")
    data_receipts: dict[str, Any] = {}
    for split in ("calibration", "test"):
        canonical_path = CANONICAL_DATA_ROOT / f"{split}_n24.jsonl"
        canonical_rows = jsonl_rows(canonical_path)
        expected_manifest = canonical_manifest["splits"][split]["manifest"]
        observed_manifest = row_manifest(canonical_rows)
        if observed_manifest != expected_manifest:
            raise RuntimeError(
                f"Canonical Phase-G {split} manifest mismatch; frozen rows changed"
            )
        rows, receipt = prepare_inverse_rendered_split(
            config,
            split=split,
            source_rows=canonical_rows,
        )
        data_path = DATA_ROOT / f"{split}_n24.jsonl"
        write_jsonl(data_path, rows)
        data_receipts[split] = {
            **receipt,
            "path": path_for_cli(data_path),
            "source_path": path_for_cli(canonical_path),
        }
    if set(row["id"] for row in jsonl_rows(DATA_ROOT / "calibration_n24.jsonl")) & set(
        row["id"] for row in jsonl_rows(DATA_ROOT / "test_n24.jsonl")
    ):
        raise RuntimeError("Calibration and test row IDs overlap")
    write_json(DATA_ROOT / "manifest.json", {"kind": "phase_g_inverse_rendered_frozen_set", "splits": data_receipts})

    source_path = Path(os.environ.get("STAGE5_INVERSE_RENDERED_SOURCE_SUMMARY", str(SOURCE_SUMMARY)))
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source = read_json(source_path)
    stage = _source_cap3(source)
    checkpoint, restore_receipt = restore_checkpoint(
        [stage.get("checkpoint_drive_backup"), stage.get("checkpoint")],
        run_dir / "restored" / "C_cap3.pt",
        label="inverse_rendered_C_cap3",
    )
    if restore_receipt["selected_checkpoint_sha256"] != SOURCE_CAP3_SHA256:
        raise RuntimeError("Restored C cap-3 checkpoint SHA mismatch")
    write_json(
        run_progress_path,
        {
            "kind": "stage5_inverse_rendered_width_gate_progress",
            "run_id": run_id,
            "status": "calibration_running",
            "checkpoint": path_for_cli(checkpoint),
            "checkpoint_sha256": SOURCE_CAP3_SHA256,
            "calibration_progress": path_for_cli(run_dir / "eval" / "calibration" / "progress.json"),
        },
    )

    calibration_summary = _run_eval(
        run_dir,
        label="calibration",
        checkpoint=checkpoint,
        data_path=DATA_ROOT / "calibration_n24.jsonl",
    )
    calibration_gate = assess_deterministic_validity(calibration_summary)
    source_guardrail = stage.get("synthetic_guardrail", {})
    retention_gate = {
        "active_diagonal_min": float(source_guardrail.get("active_diagonal_min", 0.0)),
        "required_correct_per_32": RETENTION_REQUIRED,
        "total_per_depth": RETENTION_TOTAL,
        "passed": bool(source_guardrail.get("passed", False)),
        "source_summary": source_guardrail.get("summary"),
    }
    test_summary = None
    test_gate = None
    if calibration_gate["pass"] and retention_gate["passed"]:
        write_json(
            run_progress_path,
            {
                "kind": "stage5_inverse_rendered_width_gate_progress",
                "run_id": run_id,
                "status": "test_running",
                "checkpoint": path_for_cli(checkpoint),
                "calibration_gate": calibration_gate,
                "test_progress": path_for_cli(run_dir / "eval" / "test" / "progress.json"),
            },
        )
        test_summary = _run_eval(
            run_dir,
            label="test",
            checkpoint=checkpoint,
            data_path=DATA_ROOT / "test_n24.jsonl",
        )
        test_gate = assess_deterministic_validity(test_summary)

    if not calibration_gate["pass"]:
        status = "blocked_calibration_validity"
    elif not retention_gate["passed"]:
        status = "calibration_pass_retention_blocked"
    elif not bool(test_gate and test_gate["pass"]):
        status = "blocked_test_validity"
    else:
        status = "deterministic_gate_green"
    payload = {
        "kind": "stage5_inverse_rendered_width_gate",
        "run_id": run_id,
        "status": status,
        "phase_g_alpha_status": "open_for_k1_parity" if status == "deterministic_gate_green" else "closed",
        "source_summary": path_for_cli(source_path),
        "source_checkpoint_sha256": SOURCE_CAP3_SHA256,
        "restore_receipt": restore_receipt,
        "data": data_receipts,
        "calibration": {"summary": calibration_summary, "gate": calibration_gate},
        "retention": retention_gate,
        "test": {"summary": test_summary, "gate": test_gate, "opened": test_summary is not None},
        "claim_scope": "guided branching over an explicitly rendered inverse relation",
    }
    write_json(run_dir / "summary.json", payload)
    write_json(
        run_progress_path,
        {
            "kind": "stage5_inverse_rendered_width_gate_progress",
            "run_id": run_id,
            "status": "finished",
            "final_status": status,
            "summary": path_for_cli(run_dir / "summary.json"),
        },
    )
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Inverse-Rendered Width Gate - {run_id}",
                "",
                f"- Status: `{status}`",
                f"- Calibration validity: `{calibration_gate['pooled']['correct']}/{calibration_gate['pooled']['total']}`",
                f"- Retention gate: `{retention_gate['passed']}`",
                f"- Test split opened: `{test_summary is not None}`",
                "- Phase G-alpha remains closed unless validity and retention are both green.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "-f", path_for_cli(DATA_ROOT)],
        cwd=ROOT,
        check=True,
    )
    _publish(run_dir, f"Record inverse-rendered width gate {run_id} [skip ci]")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "deterministic_gate_green" else 2


def jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
