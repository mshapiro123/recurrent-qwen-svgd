"""Stage and execute the signed Stage 2B-S preludes without training."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from colab.run_stage5_paper2_phase3_p34_a2 import DRIVE_STAGE5, MIGRATED_SHA, P33_SHA, rsync
from colab.run_stage5_paper2_phase3_p35 import I1_SHA, stage_chain
from colab.run_stage5_paper2_phase3_p35_amplitude_t1_preflight import P34_SHA, P35_ID, P35_SHA
from training.paper2_stage2bs_preludes import load_lock, sha256_file


ROOT = Path(__file__).resolve().parents[1]
MODE = os.environ.get("STAGE2BS_PRELUDE_MODE", "preflight").strip().lower()
RUN_ID = "stage5_paper2_stage2bs_preludes_20260821"
SOURCE_RUN_ID = "stage5_paper2_stage2b_depth_20260819"
AUTOPSY_RUN_ID = "stage5_paper2_stage2b_autopsy_20260820"
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
DRIVE_SOURCE = DRIVE_STAGE5 / SOURCE_RUN_ID
DRIVE_AUTOPSY = DRIVE_STAGE5 / AUTOPSY_RUN_ID
LOCK = ROOT / "training/paper2_stage2bs_preludes_lock.json"
PANEL = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
STOP_SHA = {
    0: "50cbf437adda668812dbe53a015792d3dc8ebc02cb785fba594c512b64bf2f58",
    1: "830bbfa11dca4d3b9ed56db96a7c40c887f56fb4a5227555edc1bd447b6662bc",
}
CORRECTION_SHA = {
    0: "edcb287f1b90ea812655746a855abf477d7e240501e078b1b70fe17cdd7564ed",
    1: "8adf371acbe0f7237c7ac26796def257bfc2109173f9036564c15332df480824",
}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def scratch_root() -> Path:
    for candidate in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if candidate.exists() and shutil.disk_usage(candidate).free >= 80 * 1024**3:
            target = candidate / "recurrent-qwen-svgd-stage" / RUN_ID
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise RuntimeError("Stage 2B-S preludes require at least 80 GiB local scratch")


def session_fingerprint() -> dict[str, str]:
    gpu_uuid = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()
    return {"hostname": platform.node(), "gpu_uuid": gpu_uuid}


def _model_args(scratch: Path, seed: int) -> tuple[list[str], dict[str, Any]]:
    chain = stage_chain(scratch / f"chain_seed_{seed}", seed=seed, expected_p34=P34_SHA[seed])
    p35 = scratch / f"seed_{seed}_p35_ema_step_4400.pt"
    rsync(DRIVE_STAGE5 / P35_ID / f"private/arm_s_seed_{seed}/ema_step_4400.pt", p35)
    if sha256_file(p35) != P35_SHA[seed]:
        raise RuntimeError("Stage 2B-S P3.5 endpoint changed")
    args = [
        "--migrated", str(chain["migrated"]), "--migrated_sha256", MIGRATED_SHA[seed],
        "--p33", str(chain["p33"]), "--p33_sha256", P33_SHA[seed],
        "--i1", str(chain["i1"]), "--i1_sha256", I1_SHA[seed],
        "--p34", str(chain["p34"]), "--p34_sha256", P34_SHA[seed],
        "--p35", str(p35), "--p35_sha256", P35_SHA[seed],
        "--model_cache", str(scratch / "hf_student_cache"),
    ]
    return args, chain


def preflight(scratch: Path) -> dict[str, Any]:
    lock = load_lock(LOCK)
    summaries = []
    for seed in (0, 1):
        output = DRIVE_RUN / f"receipts/preflight/seed_{seed}"
        private = scratch / f"preflight_seed_{seed}"
        reference = scratch / f"reference_k_sweep_seed_{seed}"
        reference.mkdir(parents=True, exist_ok=True)
        for loop in range(1, 5):
            source = DRIVE_AUTOPSY / f"private/seed_{seed}/k_sweep__initialization__k{loop}.jsonl"
            destination = reference / source.name
            rsync(source, destination)
        model_args, _chain = _model_args(scratch, seed)
        command = [
            sys.executable, "-u", "-m", "eval.eval_paper2_stage2bs_preludes",
            "--phase", "preflight", "--seed", str(seed), "--lock", str(LOCK),
            "--dev1_panel", str(PANEL), "--reference_k_sweep_dir", str(reference),
            "--output_dir", str(output), "--private_dir", str(private),
            *model_args,
        ]
        run(command)
        receipt = output / "preflight.json"
        if json.loads(receipt.read_text(encoding="utf-8")).get("status") != "PASS":
            raise RuntimeError(f"Stage 2B-S preflight did not pass for seed {seed}")
        durable_private = DRIVE_RUN / f"private/preflight/seed_{seed}"
        durable_private.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "--archive", f"{private}/", f"{durable_private}/"], check=True)
        summaries.append({"seed": seed, "path": str(receipt), "sha256": sha256_file(receipt)})
    result = {
        "kind": "paper2_stage2bs_preflight_wave_v1",
        "status": "PASS_RELAY_REQUIRED",
        "lock_sha256": sha256_file(LOCK),
        "session_fingerprint": session_fingerprint(),
        "seed_receipts": summaries,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(DRIVE_RUN / "receipts/preflight_wave.json", result)
    return result


def execute(scratch: Path) -> dict[str, Any]:
    load_lock(LOCK)
    wave = json.loads((DRIVE_RUN / "receipts/preflight_wave.json").read_text(encoding="utf-8"))
    if wave.get("status") != "PASS_RELAY_REQUIRED" or wave.get("lock_sha256") != sha256_file(LOCK):
        raise RuntimeError("Stage 2B-S probes require the matching relayed preflight wave")
    if wave.get("session_fingerprint") != session_fingerprint():
        raise RuntimeError(
            "Stage 2B-S session changed after preflight; rerun the preflight in this session"
        )
    probe_receipts = []
    init_states = []
    stop_checkpoints = []
    correction_artifacts = []
    for seed in (0, 1):
        model_args, _chain = _model_args(scratch, seed)
        preflight_receipt = DRIVE_RUN / f"receipts/preflight/seed_{seed}/preflight.json"
        preflight_private = DRIVE_RUN / f"private/preflight/seed_{seed}"
        output = DRIVE_RUN / f"receipts/prelude1/seed_{seed}"
        private = DRIVE_RUN / f"private/prelude1/seed_{seed}"
        run([
            sys.executable, "-u", "-m", "eval.eval_paper2_stage2bs_preludes",
            "--phase", "probes", "--seed", str(seed), "--lock", str(LOCK),
            "--dev1_panel", str(PANEL), "--preflight_receipt", str(preflight_receipt),
            "--preflight_private_dir", str(preflight_private),
            "--output_dir", str(output), "--private_dir", str(private),
            *model_args,
        ])
        receipt = output / "prelude1.json"
        probe_receipts.append({"seed": seed, "path": str(receipt), "sha256": sha256_file(receipt)})
        init_states.append(preflight_private / "initialization_state.pt")
        stop = scratch / f"seed_{seed}_ema_step_01000.pt"
        rsync(DRIVE_SOURCE / f"private/seed_{seed}/ema_step_01000.pt", stop)
        if sha256_file(stop) != STOP_SHA[seed]:
            raise RuntimeError("Stage 2B-S stop checkpoint changed")
        stop_checkpoints.append(stop)
        correction = scratch / f"seed_{seed}_correction_field_initialization.pt"
        rsync(DRIVE_AUTOPSY / f"private/seed_{seed}/correction_field__initialization.pt", correction)
        if sha256_file(correction) != CORRECTION_SHA[seed]:
            raise RuntimeError("Stage 2B-S Arm-6 correction artifact changed")
        correction_artifacts.append(correction)
    desk_output = DRIVE_RUN / "receipts/prelude2"
    desk_private = DRIVE_RUN / "private/prelude2"
    run([
        sys.executable, "-u", "-m", "eval.eval_paper2_stage2bs_preludes",
        "--phase", "desk", "--lock", str(LOCK),
        "--output_dir", str(desk_output), "--private_dir", str(desk_private),
        "--initialization_states", *map(str, init_states),
        "--stop_checkpoints", *map(str, stop_checkpoints),
        "--correction_artifacts", *map(str, correction_artifacts),
    ])
    desk_receipt = desk_output / "prelude2.json"
    seed_verdicts = [
        json.loads(Path(row["path"]).read_text(encoding="utf-8"))["seed_verdict"]
        for row in probe_receipts
    ]
    from training.paper2_stage2bs_preludes import prelude1_decision

    result = {
        "kind": "paper2_stage2bs_preludes_wave_v1",
        "status": "complete_score_only_and_cpu_audit",
        "prelude1_seed_receipts": probe_receipts,
        "prelude1_decision": prelude1_decision(seed_verdicts),
        "prelude2": {"path": str(desk_receipt), "sha256": sha256_file(desk_receipt)},
        "retention_verification": [
            *[
                {
                    "role": f"preflight_seed_{row['seed']}",
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "exists": Path(row["path"]).is_file(),
                }
                for row in wave["seed_receipts"]
            ],
            *[
                {
                    "role": f"prelude1_seed_{row['seed']}",
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "exists": Path(row["path"]).is_file(),
                }
                for row in probe_receipts
            ],
            {
                "role": "prelude2_table",
                "path": str(desk_receipt),
                "sha256": sha256_file(desk_receipt),
                "exists": desk_receipt.is_file(),
            },
        ],
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    if not all(row["exists"] for row in result["retention_verification"]):
        raise RuntimeError("Stage 2B-S retention verification failed at wave close")
    atomic_json(DRIVE_RUN / "receipts/summary.json", result)
    return result


def main() -> int:
    if MODE not in {"preflight", "run"}:
        raise RuntimeError("STAGE2BS_PRELUDE_MODE must be preflight or run")
    scratch = scratch_root()
    status = DRIVE_RUN / "receipts/status.json"
    try:
        result = preflight(scratch) if MODE == "preflight" else execute(scratch)
        atomic_json(status, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        failure = {
            "kind": "paper2_stage2bs_preludes_status_v1",
            "status": "failed",
            "mode": MODE,
            "updated_at_unix": time.time(),
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
        }
        atomic_json(status, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
