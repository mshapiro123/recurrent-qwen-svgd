"""Run Stage 2B-D M0 stability and R-1 fixed-prompt receipts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from colab.run_stage5_paper2_phase3_p34_a2 import DRIVE_STAGE5, MIGRATED_SHA, P33_SHA
from colab.run_stage5_paper2_phase3_p35 import I1_SHA
from colab.run_stage5_paper2_phase3_p35_amplitude_t1_preflight import P34_SHA, P35_SHA
from colab.run_stage5_paper2_stage2a_cv1 import PANEL_PATH, stage_inputs
from training.paper2_phase3_p31_completion import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_stage2b_preflight_20260818"
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
RUNTIME_LABEL = os.environ.get("STAGE2B_RUNTIME_LABEL", "unknown").strip().lower()
RUN_M0 = os.environ.get("STAGE2B_RUN_M0", "0").strip() == "1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 30 * 1024**3:
            target = root / "recurrent-qwen-svgd-stage" / RUN_ID
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise RuntimeError("Stage 2B preflight requires 30 GiB local scratch")


def main() -> int:
    if RUNTIME_LABEL not in {"a100_40gb", "a100_80gb", "l4"}:
        raise RuntimeError("STAGE2B_RUNTIME_LABEL must identify A100 40GB, A100 80GB, or L4")
    receipts = DRIVE_RUN / "receipts" / RUNTIME_LABEL
    status_path = receipts / "status.json"

    def status(value: str, **details: Any) -> None:
        payload = {
            "kind": "paper2_stage2b_preflight_status_v1",
            "status": value,
            "runtime_label": RUNTIME_LABEL,
            "run_m0": RUN_M0,
            "updated_at_unix": time.time(),
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
            **details,
        }
        write_json(status_path, payload)
        print(f"stage2b_preflight_status={value} details={details}", flush=True)

    try:
        status("staging_score_only_inputs")
        inputs = stage_inputs(scratch_root())
        if RUN_M0:
            m0 = receipts / "m0_stability_summary.json"
            reusable_m0 = False
            if m0.is_file():
                try:
                    reusable_m0 = bool(json.loads(m0.read_text(encoding="utf-8"))["passed"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    reusable_m0 = False
            if not reusable_m0:
                status("running_m0_stability")
                run([
                    sys.executable, "-u", "-m", "eval.eval_paper2_stage2b_m0_stability",
                    "--panel", str(PANEL_PATH),
                    "--migrated", str(inputs["migrated"]),
                    "--migrated_sha256", MIGRATED_SHA[0],
                    "--p33", str(inputs["p33"]),
                    "--p33_sha256", P33_SHA[0],
                    "--i1", str(inputs["i1"]),
                    "--i1_sha256", I1_SHA[0],
                    "--p34", str(inputs["p34"]),
                    "--p34_sha256", P34_SHA[0],
                    "--p35", str(inputs["p35"]),
                    "--p35_sha256", P35_SHA[0],
                    "--model_cache", str(inputs["model_cache"]),
                    "--output", str(m0),
                ])
            if not bool(json.loads(m0.read_text(encoding="utf-8")).get("passed")):
                raise RuntimeError("M0 receipt is not passing after score-only evaluation")
        fixed = receipts / "r1_fixed_prompt"
        if not (fixed / "summary.json").is_file():
            status("running_r1_fixed_prompt")
            lock = json.loads((ROOT / "training/paper2_stage2a_preregistration.json").read_text(encoding="utf-8"))
            run([
                sys.executable, "-u", "-m", "eval.eval_paper2_stage2b_r1_fixed_prompt",
                "--panel", str(PANEL_PATH),
                "--geometry", str(inputs["geometry"]),
                "--t3a_checkpoint", str(inputs["t3a"]),
                "--memory_slots", str(lock["data_separation"]["memory_slots"]),
                "--migrated", str(inputs["migrated"]),
                "--migrated_sha256", MIGRATED_SHA[0],
                "--p33", str(inputs["p33"]),
                "--p33_sha256", P33_SHA[0],
                "--i1", str(inputs["i1"]),
                "--i1_sha256", I1_SHA[0],
                "--p34", str(inputs["p34"]),
                "--p34_sha256", P34_SHA[0],
                "--p35", str(inputs["p35"]),
                "--p35_sha256", P35_SHA[0],
                "--model_cache", str(inputs["model_cache"]),
                "--runtime_label", RUNTIME_LABEL,
                "--output_dir", str(fixed),
            ])
        status(
            "complete_score_only",
            m0_sha256=(sha256_file(receipts / "m0_stability_summary.json") if RUN_M0 else None),
            fixed_prompt_summary_sha256=sha256_file(fixed / "summary.json"),
            fixed_prompt_logits_sha256=sha256_file(fixed / "first_token_logits.pt"),
        )
        return 0
    except Exception as error:
        status(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
