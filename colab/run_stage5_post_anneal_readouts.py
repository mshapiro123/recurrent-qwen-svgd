"""Run post-anneal synthetic-depth readouts from the annealed checkpoint.

This target is intentionally eval-only.  It starts from the chain-anneal summary,
restores the final checkpoint through the existing child readout scripts, and
publishes one combined summary that keeps reader-alignment, extrapolation, and
probe evidence in the same place.
"""

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

from colab.stage5_chain_consolidation_utils import ROOT, path_for_cli, publish_run, read_json, write_json


DEFAULT_SOURCE_SUMMARY = "outputs/stage5/stage5_chain_anneal_20260703_160250/summary.json"


def run(cmd: list[str | os.PathLike[str]], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=ROOT,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout)
    return proc


def maybe_read_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def run_reader_alignment(source_summary: dict[str, Any], output_path: Path) -> dict[str, Any] | None:
    final_eval = source_summary.get("final_eval") or {}
    active = source_summary.get("final_active_eval") or final_eval.get("active") or {}
    final = source_summary.get("final_matrix_eval") or final_eval.get("final") or {}
    active_rows = active.get("active_rows")
    final_rows = final.get("matrix_rows")
    filters = source_summary.get("filters") or {}
    final_data = final.get("data_jsonl") or (filters.get("heldout_final_eval") or {}).get("path")
    chain_data = active.get("data_jsonl") or (filters.get("heldout_active_eval") or {}).get("path")
    active_summary = active.get("active_summary")
    final_summary = final.get("matrix_summary")
    required = [active_rows, final_rows, final_data, chain_data, active_summary, final_summary]
    if not all(required):
        return None
    run(
        [
            sys.executable,
            "eval/analyze_synthetic_reader_alignment.py",
            "--active_rows",
            str(active_rows),
            "--final_rows",
            str(final_rows),
            "--final_data_jsonl",
            str(final_data),
            "--chain_data_jsonl",
            str(chain_data),
            "--active_summary",
            str(active_summary),
            "--final_summary",
            str(final_summary),
            "--output_json",
            path_for_cli(output_path),
        ]
    )
    return read_json(output_path)


def compact_extrapolation(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    active = payload.get("active_eval") or {}
    return {
        "run_id": payload.get("run_id"),
        "summary": f"outputs/stage5/{payload.get('run_id')}/summary.json",
        "checkpoint": payload.get("checkpoint"),
        "status": payload.get("status"),
        "control_read": payload.get("control_read"),
        "active_diagonal": active.get("active_diagonal"),
        "active_total": active.get("active_total"),
        "above_diagonal": active.get("above_diagonal"),
        "extrapolation_read": payload.get("extrapolation_read"),
        "artifact_check_pass": (payload.get("artifact_check") or {}).get("pass"),
    }


def compact_probe(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    probe = payload.get("probe") or {}
    return {
        "run_id": payload.get("run_id"),
        "summary": f"outputs/stage5/{payload.get('run_id')}/summary.json",
        "checkpoint": payload.get("checkpoint"),
        "status": payload.get("status"),
        "probe_diagonal": payload.get("probe_diagonal"),
        "loop_index_probe": probe.get("loop_index_probe"),
        "depth_stratified_diagonal": probe.get("depth_stratified_diagonal"),
        "loop_index_deflation_curve": probe.get("loop_index_deflation_curve"),
        "state_envelope": probe.get("state_envelope"),
        "feature_transform_probes": probe.get("feature_transform_probes"),
        "router_leak_exclusion": probe.get("router_leak_exclusion"),
    }


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    reader = payload.get("reader_alignment") or {}
    extrap = payload.get("post_anneal_extrapolation") or {}
    probe = payload.get("post_anneal_probe") or {}
    lines = [
        f"# Post-Anneal Synthetic Readouts - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Reader metric suspended: `{reader.get('final_answer_metric_suspended')}`",
        f"- Active-right/final-wrong diagonal rows: `{reader.get('total_active_right_final_wrong')}`",
        f"- Post-anneal active diagonal: `{extrap.get('active_diagonal')}`",
        f"- Post-anneal extrapolation read: `{extrap.get('extrapolation_read')}`",
        f"- Probe diagonal: `{probe.get('probe_diagonal')}`",
        f"- Loop-index probe: `{probe.get('loop_index_probe')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_POST_ANNEAL_READOUT_RUN_ID") or time.strftime(
        "stage5_post_anneal_readouts_%Y%m%d_%H%M%S"
    )
    source_summary = os.environ.get("STAGE5_POST_ANNEAL_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY)
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_payload = read_json(source_summary)
    payload: dict[str, Any] = {
        "kind": "stage5_post_anneal_synthetic_readouts",
        "run_id": run_id,
        "status": "started",
        "source_summary": source_summary,
        "source_run_id": source_payload.get("run_id"),
        "source_final_checkpoint": source_payload.get("final_checkpoint"),
        "source_final_checkpoint_drive_backup": source_payload.get("final_checkpoint_drive_backup"),
    }
    write_json(run_dir / "summary.json", payload)

    reader_alignment_path = run_dir / "reader_alignment" / "final_checkpoint_reader_alignment.json"
    reader_alignment = run_reader_alignment(source_payload, reader_alignment_path)

    extrap_run_id = os.environ.get("STAGE5_POST_ANNEAL_EXTRAP_RUN_ID", f"{run_id}_depth_extrapolation")
    run(
        [sys.executable, "colab/run_stage5_depth_extrapolation_eval.py"],
        env={
            "STAGE5_EXTRAP_RUN_ID": extrap_run_id,
            "STAGE5_EXTRAP_CHECKPOINT": source_summary,
            "STAGE5_EXTRAP_N_SYMBOLS": os.environ.get("STAGE5_POST_ANNEAL_N_SYMBOLS", "16"),
            "STAGE5_EXTRAP_DEPTHS": os.environ.get("STAGE5_POST_ANNEAL_EXTRAP_DEPTHS", "1,2,3,4,5,6"),
            "STAGE5_EXTRAP_ROWS_PER_DEPTH": os.environ.get("STAGE5_POST_ANNEAL_ROWS_PER_DEPTH", "64"),
            "STAGE5_EXTRAP_MAX_LOOPS": os.environ.get("STAGE5_POST_ANNEAL_MAX_LOOPS", "6"),
            "STAGE5_EXTRAP_DTYPE": os.environ.get("STAGE5_POST_ANNEAL_DTYPE", "bfloat16"),
        },
    )
    extrap_summary_path = ROOT / "outputs" / "stage5" / extrap_run_id / "summary.json"
    extrap_payload = maybe_read_json(extrap_summary_path)

    probe_run_id = os.environ.get("STAGE5_POST_ANNEAL_PROBE_RUN_ID", f"{run_id}_synthetic_probe")
    run(
        [sys.executable, "colab/run_stage5_synthetic_probe_battery.py"],
        env={
            "STAGE5_PROBE_RUN_ID": probe_run_id,
            "STAGE5_PROBE_CHECKPOINT": source_summary,
            "STAGE5_PROBE_N_SYMBOLS": os.environ.get("STAGE5_POST_ANNEAL_N_SYMBOLS", "16"),
            "STAGE5_PROBE_MAX_DEPTH": os.environ.get("STAGE5_POST_ANNEAL_PROBE_MAX_DEPTH", "6"),
            "STAGE5_PROBE_ROWS_PER_DEPTH": os.environ.get("STAGE5_POST_ANNEAL_ROWS_PER_DEPTH", "64"),
            "STAGE5_PROBE_LOOP_COUNTS": os.environ.get("STAGE5_POST_ANNEAL_PROBE_LOOP_COUNTS", "1,2,3,4,5,6"),
            "STAGE5_PROBE_TARGET_STEPS": os.environ.get("STAGE5_POST_ANNEAL_PROBE_TARGET_STEPS", "0,1,2,3,4,5,6"),
            "STAGE5_PROBE_DTYPE": os.environ.get("STAGE5_POST_ANNEAL_DTYPE", "bfloat16"),
        },
    )
    probe_summary_path = ROOT / "outputs" / "stage5" / probe_run_id / "summary.json"
    probe_payload = maybe_read_json(probe_summary_path)

    payload.update(
        {
            "status": "finished",
            "reader_alignment": reader_alignment,
            "reader_alignment_summary": path_for_cli(reader_alignment_path) if reader_alignment else None,
            "post_anneal_extrapolation": compact_extrapolation(extrap_payload),
            "post_anneal_probe": compact_probe(probe_payload),
            "decision_read": {
                "persistence_positive": True,
                "final_answer_metric_suspended": bool(
                    reader_alignment and reader_alignment.get("final_answer_metric_suspended")
                ),
                "extrapolation_resolved": bool(extrap_payload),
                "probe_resolved": bool(probe_payload),
            },
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 post-anneal readouts {run_id} [skip ci]", update_pointer=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
