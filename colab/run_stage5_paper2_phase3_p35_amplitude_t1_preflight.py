"""Run the no-training P3.5 amplitude surface and T1 hash-only preflight."""

from __future__ import annotations

import hashlib
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
from colab.run_stage5_paper2_phase3_p35 import (
    I1_SHA, NEW_ID, OLD_ID, PREREQUISITE_ID, stage_chain, stage_common,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_p35_amplitude_t1_20260816"
P34_ID = "stage5_paper2_phase3_p34_a2_20260814"
P35_ID = "stage5_paper2_phase3_p35_20260815"
LOCK_PATH = ROOT / "training/paper2_phase3_p35_amplitude_t1_lock.json"
CEILINGS = (0.02, 0.05, 0.08, 0.11)
P34_SHA = {
    0: "381955ec5b78d0a00883c29e9f940feac8cfc8665f7a3a4446c79734532f4ed7",
    1: "97ad532a5bffd72b2563799047b517e531e00115793bf4808f060148dfffc1ec",
}
P35_SHA = {
    0: "a047e2e7b35320376a736492c79d913b8690937da785efa2af002c8f54d26ca6",
    1: "e36cddb76407c8f853ccb43824c77cf01d15f144780726dd9aec23215467fccb",
}
DIRECTION_SHA = "294358a7dacc746b733e9f08296494c6f461443a92c093f8019a1dda56422294"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 80 * 1024**3:
            return root / "recurrent-qwen-svgd-stage" / RUN_ID
    raise RuntimeError("amplitude surface requires at least 80 GiB local scratch")


def name(seed: int, ceiling: float) -> str:
    return f"seed_{seed}_ceiling_{str(ceiling).replace('.', 'p')}"


def stage_previously_seen_task_cell(
    *, seed: int, ceiling: float, rows: Path, summary: Path
) -> None:
    source = DRIVE_STAGE5 / P35_ID / f"private/score_bundle/arm_s_seed_{seed}"
    label = f"step_4400_ema_ceiling_{ceiling:.2f}"
    rsync(source / f"{label}.jsonl", rows)
    rsync(source / f"{label}.json", summary)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    expected = P35_SHA[seed]
    p35_receipts = [
        item for item in payload.get("checkpoint_receipts", []) if item.get("label") == "p35"
    ]
    if (
        payload.get("status") != "complete_dev_only"
        or float(payload.get("evaluation_gate_ceiling", -1)) != ceiling
        or len(p35_receipts) != 1
        or p35_receipts[0].get("sha256") != expected
        or bool(payload.get("confirm_scored"))
        or bool(payload.get("eval_e_scored"))
        or bool(payload.get("optimizer_constructed"))
        or int(payload.get("optimizer_steps", -1)) != 0
    ):
        raise RuntimeError(f"previously seen task-cell receipt identity changed seed={seed} ceiling={ceiling}")


def t1_manifest(receipts: Path, panel: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in panel.read_text(encoding="utf-8").splitlines() if line]
    row_ids = [str(row["item_id"]) for row in rows]
    row_sha = hashlib.sha256(("\n".join(row_ids) + "\n").encode()).hexdigest()
    checkpoints = []
    for seed in (0, 1):
        for step in (3400, 3600, 3800, 4000):
            path = DRIVE_STAGE5 / P34_ID / f"private/main_seed_{seed}/checkpoint_step_{step}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoints.append({
                "seed": seed,
                "step": step,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    payload = {
        "kind": "paper2_sidecar_v2_t1_extraction_manifest_preflight_v1",
        "status": "complete_hashes_ready_for_lock",
        "mode": "score_blind_no_model_no_optimizer",
        "rows": len(rows),
        "row_ids_sha256": row_sha,
        "panel_file_sha256": sha256_file(panel),
        "checkpoints": checkpoints,
        "cell_schema": {
            "prelude_slots": 8,
            "recurrent_slots_per_loop": 8,
            "loops": 4,
            "layer_taps": [6, 12, 18, 24],
            "cells_per_row": 44,
            "cell_dim": 128,
        },
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    write_json(receipts / "t1_extraction_manifest_preflight.json", payload)
    return payload


def main() -> int:
    drive_run = DRIVE_STAGE5 / RUN_ID
    receipts = drive_run / "receipts"
    private = drive_run / "private/amplitude_surface"
    status_path = receipts / "status.json"
    local = ROOT / "outputs/stage5" / RUN_ID
    panel = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
    base_scores = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"

    def status(value: str, **details: object) -> None:
        write_json(status_path, {
            "kind": "paper2_phase3_p35_amplitude_t1_status_v1",
            "status": value,
            "updated_at_unix": time.time(),
            **details,
        })
        print(f"p35_amplitude_status={value} details={details}", flush=True)

    try:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        status("t1_hash_preflight")
        manifest = t1_manifest(receipts, panel)
        status("staging_amplitude_inputs", t1_manifest_sha256=sha256_file(receipts / "t1_extraction_manifest_preflight.json"))
        scratch = scratch_root()
        common = stage_common(scratch)
        chains = {seed: stage_chain(scratch, seed=seed, expected_p34=P34_SHA[seed]) for seed in (0, 1)}
        endpoints = {}
        for seed in (0, 1):
            endpoint = scratch / f"seed_{seed}_p35_ema_step_4400.pt"
            rsync(DRIVE_STAGE5 / P35_ID / f"private/arm_s_seed_{seed}/ema_step_4400.pt", endpoint)
            if sha256_file(endpoint) != P35_SHA[seed]:
                raise RuntimeError(f"P3.5 EMA endpoint SHA mismatch seed={seed}")
            endpoints[seed] = endpoint

        preflight = common["preflight"] / "private/p33_prep"
        for seed in (0, 1):
            chain = chains[seed]
            for ceiling in CEILINGS:
                condition = name(seed, ceiling)
                task_summary = private / f"{condition}_summary.json"
                task_rows = private / f"{condition}.jsonl"
                audit_summary = private / f"{condition}_audit.json"
                if not task_summary.is_file():
                    if ceiling in (0.02, 0.08):
                        status("transporting_previously_seen_task_cell", condition=condition)
                        stage_previously_seen_task_cell(
                            seed=seed, ceiling=ceiling, rows=task_rows, summary=task_summary
                        )
                    else:
                        status("scoring_task", condition=condition)
                        command = [
                            sys.executable, "-u", "-m", "eval.eval_paper2_phase3_p34_task_trajectory",
                            "--panel", str(panel), "--base_scores", str(base_scores),
                            "--output_jsonl", str(task_rows), "--summary", str(task_summary),
                            "--condition", condition, "--look", "0", "--seed", str(seed),
                            "--migrated", str(chain["migrated"]), "--migrated_sha256", MIGRATED_SHA[seed],
                            "--p33", str(chain["p33"]), "--p33_sha256", P33_SHA[seed],
                            "--i1", str(chain["i1"]), "--i1_sha256", I1_SHA[seed],
                            "--p34", str(chain["p34"]), "--p34_sha256", P34_SHA[seed],
                            "--p35", str(endpoints[seed]), "--p35_sha256", P35_SHA[seed],
                            "--gate_ceiling_override", str(ceiling),
                        ]
                        for authorized in CEILINGS:
                            command.extend(["--authorized_gate_ceiling_override", str(authorized)])
                        run(command)
                if not audit_summary.is_file():
                    status("scoring_causal_audit", condition=condition)
                    run([
                        sys.executable, "-u", "-m", "eval.eval_paper2_phase3_p35_amplitude_audit",
                        "--seed", str(seed), "--ceiling", str(ceiling),
                        "--old_summary", str(ROOT / "outputs/stage5" / OLD_ID / "summary.json"),
                        "--old_private", str(common["old"]),
                        "--new_summary", str(DRIVE_STAGE5 / NEW_ID / "receipts/full_cache_summary.json"),
                        "--new_private", str(common["new"]),
                        "--positive_audit", str(preflight / "p33_audit_slice.jsonl"),
                        "--negative_audit", str(preflight / "p33_negative_audit_slice.jsonl"),
                        "--retention_panel", str(preflight / "p33_retention_panel.jsonl"),
                        "--direction_cache", str(common["direction_cache"]),
                        "--direction_cache_sha256", DIRECTION_SHA,
                        "--migrated", str(chain["migrated"]), "--migrated_sha256", MIGRATED_SHA[seed],
                        "--p33", str(chain["p33"]), "--p33_sha256", P33_SHA[seed],
                        "--i1", str(chain["i1"]), "--i1_sha256", I1_SHA[seed],
                        "--p34", str(chain["p34"]), "--p34_sha256", P34_SHA[seed],
                        "--p35", str(endpoints[seed]), "--p35_sha256", P35_SHA[seed],
                        "--output", str(audit_summary),
                    ])
        status("analyzing")
        run([
            sys.executable, "-u", "-m", "analysis.build_paper2_phase3_p35_amplitude_surface",
            "--input_dir", str(private), "--lock", str(LOCK_PATH),
            "--output", str(local / "summary.json"),
        ])
        shutil.copy2(local / "summary.json", receipts / "summary.json")
        status(
            "complete",
            summary_sha256=sha256_file(receipts / "summary.json"),
            t1_manifest_sha256=sha256_file(receipts / "t1_extraction_manifest_preflight.json"),
            optimizer_steps=0,
            confirm_scored=False,
            eval_e_scored=False,
        )
        print("Amplitude surface and T1 hash preflight landed; release this GPU session.", flush=True)
        return 0
    except Exception as error:
        status("failed", exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
