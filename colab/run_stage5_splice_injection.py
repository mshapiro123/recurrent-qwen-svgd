"""Run the synthetic-depth splice-injection diagnostic on the support-route checkpoint."""

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

from colab.stage5_chain_consolidation_utils import (
    ROOT,
    path_for_cli,
    publish_run,
    read_json,
    resolve_checkpoint_reference,
    write_json,
)


def run(cmd: list[str | os.PathLike[str]], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout)
    return proc


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    splice = payload.get("splice_summary", {})
    lines = [
        f"# Synthetic Depth Splice Injection - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload.get('source_summary')}`",
        f"- Checkpoint: `{payload.get('checkpoint')}`",
        f"- Data: `{payload.get('data_jsonl')}`",
        f"- Target depth: `{payload.get('target_depth')}`",
        f"- Splice points: `{payload.get('splice_points')}`",
        f"- Verdict: `{splice.get('verdict')}`",
        f"- Source-orbit fraction j<=3: `{splice.get('source_orbit_fraction_j1_to_j3')}`",
        f"- Lawful fraction j<=3: `{splice.get('lawful_fraction_j1_to_j3')}`",
        f"- Shortcut fraction j<=3: `{splice.get('shortcut_fraction_j1_to_j3')}`",
        f"- Records: `{splice.get('records')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_SPLICE_RUN_ID") or time.strftime("stage5_splice_injection_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_summary = os.environ.get(
        "STAGE5_SPLICE_SOURCE_SUMMARY",
        "outputs/stage5/stage5_depth_support_route_20260705_124320/summary.json",
    )
    source_payload = read_json(source_summary)
    checkpoint, checkpoint_meta = resolve_checkpoint_reference(
        source_summary,
        run_dir / "restored" / "support_route_final.pt",
        label="support_route_final",
    )
    frozen = source_payload.get("frozen_eval_set") or {}
    data_jsonl = os.environ.get("STAGE5_SPLICE_DATA_JSONL") or frozen.get("test_chain_mcq")
    if not data_jsonl:
        raise RuntimeError("Could not resolve frozen test_chain_mcq from source summary")

    target_depth = int(os.environ.get("STAGE5_SPLICE_TARGET_DEPTH", "8"))
    splice_points = os.environ.get("STAGE5_SPLICE_POINTS", "2,4")
    max_loops = int(os.environ.get("STAGE5_SPLICE_MAX_LOOPS", "8"))
    n_pairs = int(os.environ.get("STAGE5_SPLICE_N_PAIRS", "128"))
    dtype = os.environ.get("STAGE5_SPLICE_DTYPE", "bfloat16")
    value_prefix = os.environ.get("STAGE5_SPLICE_VALUE_PREFIX", "letter:")
    n_symbols = int(os.environ.get("STAGE5_SPLICE_N_SYMBOLS", str(frozen.get("n_symbols") or 16)))

    payload: dict[str, Any] = {
        "kind": "stage5_synthetic_depth_splice_injection",
        "run_id": run_id,
        "status": "started",
        "source_summary": path_for_cli(source_summary),
        "source_run_id": source_payload.get("run_id"),
        "checkpoint": path_for_cli(checkpoint),
        "checkpoint_metadata": checkpoint_meta,
        "data_jsonl": path_for_cli(data_jsonl),
        "target_depth": target_depth,
        "splice_points": splice_points,
        "max_loops": max_loops,
        "n_pairs": n_pairs,
        "n_symbols": n_symbols,
        "value_prefix": value_prefix,
        "interpretation_rule": {
            "source_state_continuation": "source_orbit_fraction_j1_to_j3 >= 0.75",
            "a_table_state_driven": "lawful_fraction_j1_to_j3 >= 0.75",
            "prompt_position_shortcut": "shortcut_fraction_j1_to_j3 >= 0.50",
            "mixed": "otherwise",
        },
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)

    records_path = run_dir / "splice_records.jsonl"
    splice_summary_path = run_dir / "splice_summary.json"
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_splice.py",
            "--data_jsonl",
            path_for_cli(data_jsonl),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(records_path),
            "--output_summary",
            path_for_cli(splice_summary_path),
            "--target_depth",
            str(target_depth),
            "--splice_points",
            splice_points,
            "--max_loops",
            str(max_loops),
            "--n_pairs",
            str(n_pairs),
            "--n_symbols",
            str(n_symbols),
            "--value_prefix",
            value_prefix,
            "--bridge_projection_mode",
            "split",
            "--dtype",
            dtype,
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    splice_summary = read_json(splice_summary_path)
    payload.update(
        {
            "status": "finished",
            "splice_records": path_for_cli(records_path),
            "splice_summary_path": path_for_cli(splice_summary_path),
            "splice_summary": splice_summary,
            "decision_read": {
                "question": "Does the support-6 synthetic model continue from spliced hidden state or ignore it?",
                "verdict": splice_summary.get("verdict"),
                "source_orbit_fraction_j1_to_j3": splice_summary.get("source_orbit_fraction_j1_to_j3"),
                "lawful_fraction_j1_to_j3": splice_summary.get("lawful_fraction_j1_to_j3"),
                "shortcut_fraction_j1_to_j3": splice_summary.get("shortcut_fraction_j1_to_j3"),
                "ladder_interpretation_gate": "Run support-8 ladder only after reading this verdict.",
            },
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 splice injection {run_id} [skip ci]", update_pointer=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
