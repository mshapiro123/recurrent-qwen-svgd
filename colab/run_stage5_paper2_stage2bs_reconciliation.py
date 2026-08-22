"""Stage and run the authorized Stage 2B-S serving-graph reconciliation."""

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
from eval.eval_paper2_stage2bs_reconciliation import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_stage2bs_reconciliation_20260822"
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
LOCK = ROOT / "training/paper2_stage2bs_reconciliation_lock.json"
PANEL = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"


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
        if candidate.exists() and shutil.disk_usage(candidate).free >= 40 * 1024**3:
            target = candidate / "recurrent-qwen-svgd-stage" / RUN_ID
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise RuntimeError("Stage 2B-S reconciliation requires at least 40 GiB local scratch")


def main() -> int:
    scratch = scratch_root()
    result_run = Path(os.environ.get("STAGE5_RECONCILIATION_RESULT_ROOT", str(DRIVE_RUN)))
    status_path = result_run / "receipts/status.json"
    try:
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        if not lock.get("locked_before_trace"):
            raise RuntimeError("Stage 2B-S reconciliation identity contract is not locked")
        seed_receipts = []
        for seed in (0, 1):
            chain = stage_chain(scratch / f"chain_seed_{seed}", seed=seed, expected_p34=P34_SHA[seed])
            p35 = scratch / f"seed_{seed}_p35_ema_step_4400.pt"
            rsync(DRIVE_STAGE5 / P35_ID / f"private/arm_s_seed_{seed}/ema_step_4400.pt", p35)
            if sha256_file(p35) != P35_SHA[seed]:
                raise RuntimeError("Stage 2B-S reconciliation P3.5 endpoint changed")
            output = result_run / f"receipts/seed_{seed}"
            private = result_run / f"private/seed_{seed}"
            run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "eval.eval_paper2_stage2bs_reconciliation",
                    "--seed",
                    str(seed),
                    "--lock",
                    str(LOCK),
                    "--dev1_panel",
                    str(PANEL),
                    "--migrated",
                    str(chain["migrated"]),
                    "--migrated_sha256",
                    MIGRATED_SHA[seed],
                    "--p33",
                    str(chain["p33"]),
                    "--p33_sha256",
                    P33_SHA[seed],
                    "--i1",
                    str(chain["i1"]),
                    "--i1_sha256",
                    I1_SHA[seed],
                    "--p34",
                    str(chain["p34"]),
                    "--p34_sha256",
                    P34_SHA[seed],
                    "--p35",
                    str(p35),
                    "--p35_sha256",
                    P35_SHA[seed],
                    "--model_cache",
                    str(scratch / "hf_student_cache"),
                    "--output_dir",
                    str(output),
                    "--private_dir",
                    str(private),
                ]
            )
            receipt = output / f"seed_{seed}_summary.json"
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            if payload.get("status") != "complete_score_only":
                raise RuntimeError(f"Stage 2B-S reconciliation seed {seed} did not complete")
            seed_receipts.append(
                {
                    "seed": seed,
                    "path": str(receipt),
                    "sha256": sha256_file(receipt),
                    "first_divergence": payload["first_divergence"],
                    "decision_mapping": payload["decision_mapping"],
                }
            )
        decisions = {row["decision_mapping"] for row in seed_receipts}
        divergences = {row["first_divergence"]["classification"] for row in seed_receipts}
        result = {
            "kind": "paper2_stage2bs_reconciliation_wave_v1",
            "status": "complete_score_only",
            "lock_sha256": sha256_file(LOCK),
            "runtime": {
                "hostname": platform.node(),
                "gpu_uuid": subprocess.run(
                    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                ).stdout.strip(),
            },
            "seed_receipts": seed_receipts,
            "seed_agreement": len(decisions) == 1 and len(divergences) == 1,
            "decision_mapping": next(iter(decisions)) if len(decisions) == 1 else "ESCALATE",
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "training_performed": False,
            "confirm_scored": False,
            "eval_e_scored": False,
        }
        atomic_json(result_run / "receipts/summary.json", result)
        atomic_json(status_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        failure = {
            "kind": "paper2_stage2bs_reconciliation_status_v1",
            "status": "failed",
            "updated_at_unix": time.time(),
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "training_performed": False,
            "confirm_scored": False,
            "eval_e_scored": False,
        }
        atomic_json(status_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
