"""Run the no-training D1 plus K+ depth-discrimination probe."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from colab.run_stage5_paper2_phase3_p34_a2 import (
    DRIVE_STAGE5, MIGRATED_SHA, P33_SHA, rsync, sha256_file, write_json,
)
from colab.run_stage5_paper2_phase3_p35 import I1_SHA, stage_chain
from colab.run_stage5_paper2_phase3_p35_amplitude_t1_preflight import (
    P34_SHA, P35_SHA,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_p35_depth_discrimination_20260816"
P35_ID = "stage5_paper2_phase3_p35_20260815"
LOCK_PATH = ROOT / "training/paper2_phase3_p35_depth_discrimination_lock.json"
PRIMARY_K = (1, 2, 3, 4)
EXPLORATORY_K = (5, 6)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 70 * 1024**3:
            return root / "recurrent-qwen-svgd-stage" / RUN_ID
    raise RuntimeError("K+ probe requires at least 70 GiB local scratch")


def stage_k4(*, seed: int, rows: Path, summary: Path) -> None:
    source = DRIVE_STAGE5 / P35_ID / f"private/score_bundle/arm_s_seed_{seed}"
    label = "step_4400_ema_ceiling_0.02"
    source_rows = source / f"{label}.jsonl"
    source_summary = source / f"{label}.json"
    rsync(source_rows, rows)
    rsync(source_summary, summary)
    records = [json.loads(line) for line in rows.read_text(encoding="utf-8").splitlines() if line]
    for record in records:
        record["condition"] = f"seed_{seed}_k_4"
        record["look"] = 0
        record["flow_loops"] = 4
        record["clamped_extension"] = False
    rows.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    p35 = [item for item in payload.get("checkpoint_receipts", []) if item.get("label") == "p35"]
    if (
        payload.get("status") != "complete_dev_only"
        or float(payload.get("evaluation_gate_ceiling", -1.0)) != 0.02
        or len(p35) != 1
        or p35[0].get("sha256") != P35_SHA[seed]
        or bool(payload.get("confirm_scored"))
        or bool(payload.get("eval_e_scored"))
        or bool(payload.get("optimizer_constructed"))
        or int(payload.get("optimizer_steps", -1)) != 0
    ):
        raise RuntimeError(f"canonical P3.5 K=4 receipt identity changed seed={seed}")
    payload.update({
        "condition": f"seed_{seed}_k_4",
        "look": 0,
        "flow_loops": 4,
        "clamped_extension": False,
        "depth_scope": "registered_trained_support",
        "depth_parameter_indices": {
            "flow_step_embedding": [0, 1, 2, 3],
            "bridge_gate_and_rho": 3,
        },
        "transported_from_canonical_p35_score_bundle": True,
        "source_rows_sha256": sha256_file(source_rows),
        "source_summary_sha256": sha256_file(source_summary),
    })
    write_json(summary, payload)


def main() -> int:
    drive_run = DRIVE_STAGE5 / RUN_ID
    receipts = drive_run / "receipts"
    private = drive_run / "private/depth_discrimination"
    status_path = receipts / "status.json"
    local = ROOT / "outputs/stage5" / RUN_ID
    panel = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
    base_scores = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"

    def status(value: str, **details: object) -> None:
        write_json(status_path, {
            "kind": "paper2_phase3_p35_depth_discrimination_status_v1",
            "status": value,
            "updated_at_unix": time.time(),
            **details,
        })
        print(f"p35_depth_status={value} details={details}", flush=True)

    try:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if lock["status"] != "locked_before_scoring":
            raise RuntimeError("depth-discrimination lock is not active")
        status("staging_inputs")
        scratch = scratch_root()
        chains = {seed: stage_chain(scratch, seed=seed, expected_p34=P34_SHA[seed]) for seed in (0, 1)}
        endpoints = {}
        for seed in (0, 1):
            endpoint = scratch / f"seed_{seed}_p35_ema_step_4400.pt"
            rsync(DRIVE_STAGE5 / P35_ID / f"private/arm_s_seed_{seed}/ema_step_4400.pt", endpoint)
            if sha256_file(endpoint) != P35_SHA[seed]:
                raise RuntimeError(f"P3.5 EMA endpoint SHA mismatch seed={seed}")
            endpoints[seed] = endpoint

        for seed in (0, 1):
            chain = chains[seed]
            for k in (*PRIMARY_K, *EXPLORATORY_K):
                condition = f"seed_{seed}_k_{k}"
                rows = private / f"{condition}.jsonl"
                summary = private / f"{condition}.json"
                if summary.is_file():
                    status("resuming_completed_cell", condition=condition)
                    continue
                if k == 4:
                    status("transporting_canonical_k4", condition=condition)
                    stage_k4(seed=seed, rows=rows, summary=summary)
                    continue
                status("scoring_k", condition=condition, scope=("registered" if k <= 4 else "exploratory_clamped"))
                command = [
                    sys.executable, "-u", "-m", "eval.eval_paper2_phase3_p34_task_trajectory",
                    "--panel", str(panel), "--base_scores", str(base_scores),
                    "--output_jsonl", str(rows), "--summary", str(summary),
                    "--condition", condition, "--look", "0", "--seed", str(seed),
                    "--migrated", str(chain["migrated"]), "--migrated_sha256", MIGRATED_SHA[seed],
                    "--p33", str(chain["p33"]), "--p33_sha256", P33_SHA[seed],
                    "--i1", str(chain["i1"]), "--i1_sha256", I1_SHA[seed],
                    "--p34", str(chain["p34"]), "--p34_sha256", P34_SHA[seed],
                    "--p35", str(endpoints[seed]), "--p35_sha256", P35_SHA[seed],
                    "--gate_ceiling_override", "0.02", "--flow_loops", str(k),
                    "--mcq_batch_size", "32", "--generation_batch_size", "8",
                ]
                if k in EXPLORATORY_K:
                    command.append("--allow_clamped_extension")
                run(command)

        status("analyzing")
        run([
            sys.executable, "-u", "-m", "analysis.build_paper2_phase3_p35_depth_discrimination",
            "--input_dir", str(private), "--lock", str(LOCK_PATH),
            "--output", str(local / "summary.json"),
        ])
        shutil.copy2(local / "summary.json", receipts / "summary.json")
        status(
            "complete",
            summary_sha256=sha256_file(receipts / "summary.json"),
            optimizer_steps=0,
            confirm_scored=False,
            eval_e_scored=False,
        )
        print("D1 plus K+ depth-discrimination receipt landed; release this GPU session.", flush=True)
        return 0
    except Exception as error:
        status("failed", exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
