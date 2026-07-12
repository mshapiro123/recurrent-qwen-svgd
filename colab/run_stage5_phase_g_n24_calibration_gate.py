"""Gate the deterministic keeper on the arbitrary N=24 calibration split."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_natural_surface_transfer import restore_checkpoint, run
from colab.stage5_publish_utils import publishable_artifact_paths


SOURCE_SUMMARY = (
    ROOT
    / "outputs/stage5/stage5_phase_g_experiment1_fixed_boundary_20260712/summary.json"
)
CALIBRATION_DATA = ROOT / "data/phase_g_alpha/calibration_n24.jsonl"
POOLED_FLOOR = 0.75
DEPTH_FLOOR = 0.60
STRATUM_FLOORS = {"unique": 0.80, "small": 0.60, "large": 0.60}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assess_calibration(summary: dict[str, Any]) -> dict[str, Any]:
    pooled = float(summary["overall"]["greedy_valid_rate"])
    depths = {
        str(key): float(value["greedy_valid_rate"])
        for key, value in summary["by_depth"].items()
    }
    strata = {
        str(key): float(value["greedy_valid_rate"])
        for key, value in summary["by_preimage_stratum"].items()
    }
    missing = sorted(set(STRATUM_FLOORS).difference(strata))
    if missing:
        raise RuntimeError(f"Calibration summary is missing preimage strata: {missing}")
    checks = {
        "pooled": pooled >= POOLED_FLOOR,
        "depth_min": min(depths.values()) >= DEPTH_FLOOR,
        **{
            f"stratum_{name}": strata[name] >= floor
            for name, floor in STRATUM_FLOORS.items()
        },
    }
    return {
        "pooled_greedy_valid_rate": pooled,
        "pooled_floor": POOLED_FLOOR,
        "by_depth_greedy_valid_rate": depths,
        "min_depth_greedy_valid_rate": min(depths.values()),
        "depth_floor": DEPTH_FLOOR,
        "by_stratum_greedy_valid_rate": strata,
        "stratum_floors": STRATUM_FLOORS,
        "checks": checks,
        "passed": all(checks.values()),
    }


def source_checkpoint(source: dict[str, Any]) -> tuple[str, str]:
    if source.get("status") != "experiment1_passed":
        raise RuntimeError(
            f"Experiment 1 is not complete and green: status={source.get('status')!r}"
        )
    if not bool((source.get("abductive_gate") or {}).get("passed")):
        raise RuntimeError("Experiment 1 abductive gate did not pass")
    if not bool((source.get("synthetic_guardrail") or {}).get("passed")):
        raise RuntimeError("Experiment 1 synthetic guardrail did not pass")
    stage = source["abductive_train"]
    reference = stage.get("checkpoint_drive_backup") or stage.get("checkpoint")
    expected_sha = str(stage["checkpoint_sha256"])
    return str(reference), expected_sha


def publish(run_dir: Path) -> None:
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in publishable_artifact_paths(run_dir):
        if path.name.endswith("rows.jsonl"):
            continue
        subprocess.run(["git", "add", "-f", path.relative_to(ROOT).as_posix()], cwd=ROOT, check=False)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(
        ["git", "commit", "-m", f"Record Phase G N24 calibration gate {run_dir.name} [skip ci]"],
        cwd=ROOT,
        check=True,
    )
    push = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
    if push.returncode:
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def main() -> int:
    run_id = os.environ.get("STAGE5_PHASE_G_N24_GATE_RUN_ID") or time.strftime(
        "stage5_phase_g_n24_calibration_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs/stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(os.environ.get("STAGE5_PHASE_G_N24_GATE_SOURCE", str(SOURCE_SUMMARY)))
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source = read_json(source_path)
    checkpoint_reference, expected_sha = source_checkpoint(source)
    checkpoint, restore = restore_checkpoint(
        [checkpoint_reference],
        run_dir / "restored" / "deterministic_abductive_keeper.pt",
        label="phase_g_n24_deterministic_keeper",
    )
    if restore["selected_checkpoint_sha256"] != expected_sha:
        raise RuntimeError(
            f"N24 calibration checkpoint SHA mismatch: "
            f"{restore['selected_checkpoint_sha256']} != {expected_sha}"
        )
    eval_dir = run_dir / "calibration"
    run(
        [
            sys.executable,
            "eval/eval_abductive_coverage.py",
            "--data_jsonl",
            CALIBRATION_DATA.relative_to(ROOT).as_posix(),
            "--checkpoint",
            checkpoint.relative_to(ROOT).as_posix(),
            "--output_jsonl",
            (eval_dir / "rows.jsonl").relative_to(ROOT).as_posix(),
            "--output_summary",
            (eval_dir / "summary.json").relative_to(ROOT).as_posix(),
            "--sample_counts",
            "1",
            "--temperature",
            "0.7",
            "--bridge_projection_mode",
            "split",
            "--dtype",
            os.environ.get("STAGE5_PHASE_G_N24_GATE_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    calibration = assess_calibration(read_json(eval_dir / "summary.json"))
    payload = {
        "kind": "stage5_phase_g_n24_calibration_gate",
        "run_id": run_id,
        "status": "passed" if calibration["passed"] else "needs_deterministic_arbitrary_continuation",
        "source_summary": source_path.relative_to(ROOT).as_posix(),
        "source_checkpoint_sha256": expected_sha,
        "source_synthetic_guardrail": source["synthetic_guardrail"],
        "calibration_data": CALIBRATION_DATA.relative_to(ROOT).as_posix(),
        "test_split_opened": False,
        "calibration": calibration,
        "phase_g_alpha_substrate_ready": bool(calibration["passed"]),
        "next_action": (
            "lock_checkpoint_and_power_primary_gate"
            if calibration["passed"]
            else "run_one_bounded_deterministic_arbitrary_table_continuation"
        ),
        "do_not_claim": (
            "This is a deterministic substrate calibration gate, not evidence that latent width helps."
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Phase G N24 Calibration Gate - {run_id}",
                "",
                f"- Status: `{payload['status']}`",
                f"- Pooled validity: `{calibration['pooled_greedy_valid_rate']:.6f}`",
                f"- Minimum depth validity: `{calibration['min_depth_greedy_valid_rate']:.6f}`",
                f"- Strata: `{calibration['by_stratum_greedy_valid_rate']}`",
                f"- Test split opened: `{payload['test_split_opened']}`",
                f"- Next action: `{payload['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    publish(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
