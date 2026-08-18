"""Run the authorized score-only Stage 2A CV-1 and D5 diagnostics."""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_paper2_phase3_kp1_t1 import (
    P35_ID,
    P35_SHA,
    stage_chain_with_verified_p34,
)
from colab.run_stage5_paper2_phase3_p34_a2 import DRIVE_STAGE5, MIGRATED_SHA, P33_SHA, rsync
from colab.run_stage5_paper2_phase3_p35 import I1_SHA
from colab.run_stage5_paper2_phase3_p35_amplitude_t1_preflight import P34_SHA
from training.paper2_phase3_p31_completion import sha256_file


RUN_ID = "stage5_paper2_stage2a_cv1_d5_20260818"
SOURCE_RUN_ID = "stage5_paper2_stage2a_t3_screen_20260817"
CONTENT_ID = "stage5_paper2_stage2a_content_geometry_20260817"
LOCK_PATH = ROOT / "training/paper2_stage2a_preregistration.json"
SPEC_PATH = ROOT / "eval/paper2_stage2a_cv1_spec.json"
AUTHORITY_PATH = ROOT / "docs/STRATEGY_T3_SCREEN_RESPONSE_20260818.md"
PANEL_PATH = ROOT / "training/paper2_stage2a_p34_task_panel.jsonl"
BASE_SCORES_PATH = (
    ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"
)
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
PRIVATE_DIR = DRIVE_RUN / "private"
RECEIPT_DIR = DRIVE_RUN / "receipts"
LOCAL_DIR = ROOT / "outputs/stage5" / RUN_ID


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    tail: deque[str] = deque(maxlen=500)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    returncode = process.wait()
    if returncode:
        print("stage2a_cv1_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("stage2a_cv1_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 25 * 1024**3:
            target = root / "recurrent-qwen-svgd-stage" / RUN_ID
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise RuntimeError("Stage 2A CV-1 requires 25 GiB local scratch")


def stage_inputs(scratch: Path) -> dict[str, Path]:
    chain = stage_chain_with_verified_p34(
        scratch / "chain_seed_0", seed=0, expected_p34=P34_SHA[0]
    )
    p35 = scratch / "seed_0_p35_ema_step_4400.pt"
    rsync(DRIVE_STAGE5 / P35_ID / "private/arm_s_seed_0/ema_step_4400.pt", p35)
    if sha256_file(p35) != P35_SHA[0]:
        raise RuntimeError("CV-1 P3.5 seed-0 endpoint SHA mismatch")

    source_private = DRIVE_STAGE5 / SOURCE_RUN_ID / "private"
    geometry = scratch / "stage2a_memory_geometry.pt"
    t3a = scratch / "t3a_seed_0_checkpoint_step_1200.pt"
    t3b = scratch / "t3b_seed_0_checkpoint_step_1200.pt"
    rsync(DRIVE_STAGE5 / CONTENT_ID / "private/stage2a_memory_geometry.pt", geometry)
    rsync(source_private / "t3a_seed_0/checkpoint_step_1200.pt", t3a)
    rsync(source_private / "t3b_seed_0/checkpoint_step_1200.pt", t3b)
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    for host, path in (("t3a", t3a), ("t3b", t3b)):
        if sha256_file(path) != spec["checkpoint_sha256"][host]:
            raise RuntimeError(f"CV-1 staged {host} checkpoint SHA mismatch")
    model_cache = scratch / "hf_model_cache" / "student_0p5b"
    model_cache.mkdir(parents=True, exist_ok=True)
    return chain | {
        "p35": p35,
        "geometry": geometry,
        "t3a": t3a,
        "t3b": t3b,
        "model_cache": model_cache,
    }


def main() -> int:
    for path in (PRIVATE_DIR, RECEIPT_DIR, LOCAL_DIR):
        path.mkdir(parents=True, exist_ok=True)
    status_path = RECEIPT_DIR / "status.json"

    def status(value: str, **details: Any) -> None:
        payload = {
            "kind": "paper2_stage2a_cv1_colab_status_v1",
            "status": value,
            "updated_at_unix": time.time(),
            "optimizer_constructed": False,
            "confirm_scored": False,
            "eval_e_scored": False,
            **details,
        }
        write_json(status_path, payload)
        print(f"stage2a_cv1_status={value} details={details}", flush=True)

    try:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        if sha256_file(AUTHORITY_PATH) != spec["authority"]["sha256"]:
            raise RuntimeError("CV-1 strategy authority bytes changed")
        status("staging_score_only_inputs")
        inputs = stage_inputs(scratch_root())
        cv1_dir = PRIVATE_DIR / "cv1"
        analysis_dir = PRIVATE_DIR / "analysis"
        status("running_cv1_crossed_values")
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_stage2a_cv1",
                "--spec", str(SPEC_PATH),
                "--lock", str(LOCK_PATH),
                "--panel", str(PANEL_PATH),
                "--base_scores", str(BASE_SCORES_PATH),
                "--geometry", str(inputs["geometry"]),
                "--t3a_checkpoint", str(inputs["t3a"]),
                "--t3b_checkpoint", str(inputs["t3b"]),
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
                "--output_dir", str(cv1_dir),
            ]
        )
        status("running_d5_analysis")
        figure_png = analysis_dir / "paper2_stage2a_cv1_d5_20260818.png"
        figure_svg = analysis_dir / "paper2_stage2a_cv1_d5_20260818.svg"
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "analysis.analyze_paper2_stage2a_cv1_d5",
                "--input_dir", str(cv1_dir),
                "--output_dir", str(analysis_dir),
                "--figure_png", str(figure_png),
                "--figure_svg", str(figure_svg),
            ]
        )
        for source in (
            cv1_dir / "summary.json",
            analysis_dir / "analysis_summary.json",
            figure_png,
            figure_svg,
        ):
            shutil.copy2(source, RECEIPT_DIR / source.name)
            shutil.copy2(source, LOCAL_DIR / source.name)
        archive = RECEIPT_DIR / "stage2a_cv1_d5_receipts.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(cv1_dir, arcname="cv1")
            handle.add(analysis_dir, arcname="analysis")
        status(
            "complete_dev_score_only",
            summary_sha256=sha256_file(cv1_dir / "summary.json"),
            analysis_sha256=sha256_file(analysis_dir / "analysis_summary.json"),
            archive_sha256=sha256_file(archive),
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
