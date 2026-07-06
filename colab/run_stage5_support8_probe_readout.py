"""Run envelope/clock/probe diagnostics on the support-8 ladder checkpoint."""

from __future__ import annotations

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
from colab.stage5_support8_followup import (
    DEFAULT_SUPPORT8_SOURCE_SUMMARY,
    active_diagonal,
    decay_alignment,
    validate_support8_source_summary,
)


DEFAULT_LOOP_COUNTS = ",".join(str(idx) for idx in range(1, 15))
DEFAULT_TARGET_STEPS = ",".join(str(idx) for idx in range(0, 15))


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


def compact_probe(probe: dict[str, Any]) -> dict[str, Any]:
    raw_feature = (probe.get("feature_transform_probes") or {}).get("raw") or {}
    unit_feature = (probe.get("feature_transform_probes") or {}).get("unit_norm") or {}
    rms_feature = (probe.get("feature_transform_probes") or {}).get("rms_norm") or {}
    return {
        "summary_kind": probe.get("kind"),
        "rows": probe.get("rows"),
        "state_records": probe.get("state_records"),
        "loop_counts": probe.get("loop_counts"),
        "target_steps": probe.get("target_steps"),
        "probe_diagonal": {
            str(loop): float((probe.get("grid") or {}).get(str(loop), {}).get(str(loop), {}).get("accuracy", 0.0))
            for loop in probe.get("loop_counts", [])
            if str(loop) in (probe.get("grid") or {})
        },
        "loop_index_probe": probe.get("loop_index_probe"),
        "loop_index_deflation_curve": probe.get("loop_index_deflation_curve"),
        "state_envelope": probe.get("state_envelope"),
        "feature_transform_loop_index": {
            "raw": raw_feature.get("loop_index_probe"),
            "unit_norm": unit_feature.get("loop_index_probe"),
            "rms_norm": rms_feature.get("loop_index_probe"),
        },
        "feature_transform_state_envelope": {
            "raw": raw_feature.get("state_envelope"),
            "unit_norm": unit_feature.get("state_envelope"),
            "rms_norm": rms_feature.get("state_envelope"),
        },
        "router_leak_exclusion": probe.get("router_leak_exclusion"),
    }


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    probe = payload.get("probe_compact") or {}
    decay = payload.get("decay_alignment") or {}
    lines = [
        f"# Support-8 Probe Readout - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Frozen rows: `{payload.get('frozen_data_jsonl')}`",
        f"- Registered prediction: `{decay.get('registered_prediction')}`",
        f"- First depth below 0.90: `{decay.get('first_depth_below_0_90')}`",
        f"- First depth below strong-scaling bar: `{decay.get('first_depth_below_strong_scaling_bar')}`",
        f"- Loop-index probe: `{probe.get('loop_index_probe')}`",
        f"- Probe diagonal: `{probe.get('probe_diagonal')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_SUPPORT8_PROBE_RUN_ID") or time.strftime(
        "stage5_support8_probe_readout_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_summary = os.environ.get("STAGE5_SUPPORT8_SOURCE_SUMMARY", DEFAULT_SUPPORT8_SOURCE_SUMMARY)
    source_payload = read_json(source_summary)
    source_check = validate_support8_source_summary(source_summary, source_payload)
    checkpoint, checkpoint_meta = resolve_checkpoint_reference(
        source_summary,
        run_dir / "restored" / "support8_final.pt",
        label="support8_final",
    )
    frozen_data = source_check["frozen_test_chain_mcq"]
    if not (ROOT / frozen_data).exists():
        raise FileNotFoundError(f"Missing frozen support-8 probe rows: {frozen_data}")

    payload: dict[str, Any] = {
        "kind": "stage5_support8_probe_readout",
        "run_id": run_id,
        "status": "started",
        "source_summary": source_summary,
        "source_check": source_check,
        "checkpoint": path_for_cli(checkpoint),
        "checkpoint_metadata": checkpoint_meta,
        "frozen_data_jsonl": frozen_data,
        "decay_alignment": decay_alignment(active_diagonal(source_payload)),
    }
    write_json(run_dir / "summary.json", payload)

    probe_summary = run_dir / "probe" / "support8_frozen_depth14_state_probe.json"
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_probe.py",
            "--data_jsonl",
            frozen_data,
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_summary",
            path_for_cli(probe_summary),
            "--loop_counts",
            os.environ.get("STAGE5_SUPPORT8_PROBE_LOOP_COUNTS", DEFAULT_LOOP_COUNTS),
            "--target_steps",
            os.environ.get("STAGE5_SUPPORT8_PROBE_TARGET_STEPS", DEFAULT_TARGET_STEPS),
            "--feature_transforms",
            os.environ.get("STAGE5_SUPPORT8_PROBE_FEATURE_TRANSFORMS", "raw,unit_norm,rms_norm"),
            "--n_symbols",
            os.environ.get("STAGE5_SUPPORT8_PROBE_N_SYMBOLS", "16"),
            "--value_prefix",
            os.environ.get("STAGE5_SUPPORT8_PROBE_VALUE_PREFIX", "letter:"),
            "--permutations",
            os.environ.get("STAGE5_SUPPORT8_PROBE_PERMUTATIONS", "20"),
            "--ridge_l2",
            os.environ.get("STAGE5_SUPPORT8_PROBE_RIDGE_L2", "0.01"),
            "--envelope_rank",
            os.environ.get("STAGE5_SUPPORT8_PROBE_ENVELOPE_RANK", "32"),
            "--envelope_fit_loop_max",
            os.environ.get("STAGE5_SUPPORT8_PROBE_ENVELOPE_FIT_LOOP_MAX", "8"),
            "--envelope_fit_depth_max",
            os.environ.get("STAGE5_SUPPORT8_PROBE_ENVELOPE_FIT_DEPTH_MAX", "8"),
            "--bridge_projection_mode",
            "split",
            "--dtype",
            os.environ.get("STAGE5_SUPPORT8_PROBE_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    probe = read_json(probe_summary)
    payload.update(
        {
            "status": "finished",
            "probe_summary": path_for_cli(probe_summary),
            "probe_compact": compact_probe(probe),
            "decision_read": {
                "question": "Does state-envelope/clock drift align with support-8 decay onset near depths 9/10?",
                "registered_prediction": payload["decay_alignment"]["registered_prediction"],
                "first_depth_below_0_90": payload["decay_alignment"]["first_depth_below_0_90"],
                "first_depth_below_strong_scaling_bar": payload["decay_alignment"][
                    "first_depth_below_strong_scaling_bar"
                ],
                "readout_type": "eval_only_envelope_clock_probe",
            },
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 support-8 probe readout {run_id} [skip ci]", update_pointer=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
