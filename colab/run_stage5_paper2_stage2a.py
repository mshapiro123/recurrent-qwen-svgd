"""Run the locked Stage 2A T3a/T3b screen and registered controls."""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
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
from colab.run_stage5_paper2_phase3_p34_a2 import (
    DRIVE_STAGE5,
    MIGRATED_SHA,
    P33_SHA,
    rsync,
)
from colab.run_stage5_paper2_phase3_p35 import I1_SHA
from colab.run_stage5_paper2_phase3_p35_amplitude_t1_preflight import P34_SHA
from training.paper2_phase3_p31_completion import sha256_file
from training.paper2_stage2a_lock import assert_stage2a_training_authorized


RUN_ID = "stage5_paper2_stage2a_t3_screen_20260817"
CONTENT_ID = "stage5_paper2_stage2a_content_geometry_20260817"
CACHE_ID = "stage5_paper2_stage2a_training_cache_20260817"
LOCK_PATH = ROOT / "training/paper2_stage2a_preregistration.json"
PANEL_PATH = ROOT / "training/paper2_stage2a_p34_task_panel.jsonl"
BASE_SCORES_PATH = (
    ROOT
    / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"
)
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
PRIVATE_DIR = DRIVE_RUN / "private"
RECEIPT_DIR = DRIVE_RUN / "receipts"
LOCAL_DIR = ROOT / "outputs/stage5" / RUN_ID


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        print("stage2a_arm_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("stage2a_arm_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 40 * 1024**3:
            target = root / "recurrent-qwen-svgd-stage" / RUN_ID
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise RuntimeError("Stage 2A screen requires 40 GiB local scratch")


def latest_checkpoint(output: Path) -> Path | None:
    candidates = []
    for path in output.glob("checkpoint_step_*.pt"):
        try:
            step = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        candidates.append((step, path))
    return max(candidates, default=(0, None))[1]


def stage_inputs(scratch: Path) -> dict[str, Any]:
    content_private = DRIVE_STAGE5 / CONTENT_ID / "private"
    cache_private = DRIVE_STAGE5 / CACHE_ID / "private"
    paths = {
        "geometry": content_private / "stage2a_memory_geometry.pt",
        "teacher_lattice": cache_private / "stage2a_teacher_lattice.pt",
        "population_manifest": cache_private / "stage2a_training_population.jsonl",
        "owner_manifest": cache_private / "stage2a_memory_owner_manifest.jsonl",
        "student_features": cache_private / "stage2a_student_prefix_features.pt",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing Stage 2A input {label}: {path}")
    chains = {}
    for seed in (0, 1):
        chain = stage_chain_with_verified_p34(
            scratch / f"chain_seed_{seed}", seed=seed, expected_p34=P34_SHA[seed]
        )
        p35 = scratch / f"seed_{seed}_p35_ema_step_4400.pt"
        rsync(DRIVE_STAGE5 / P35_ID / f"private/arm_s_seed_{seed}/ema_step_4400.pt", p35)
        if sha256_file(p35) != P35_SHA[seed]:
            raise RuntimeError(f"Stage 2A P3.5 endpoint SHA mismatch seed={seed}")
        chains[seed] = chain | {"p35": p35}
    model_cache_candidates = (
        Path("/mnt/local-scratch/recurrent-qwen-svgd-stage")
        / CONTENT_ID
        / "hf_model_cache"
        / "student_0p5b",
        Path("/mnt/local-scratch/recurrent-qwen-svgd-stage")
        / CACHE_ID
        / "hf_model_cache"
        / "student_0p5b",
        Path("/content/recurrent-qwen-svgd-stage")
        / CACHE_ID
        / "hf_model_cache"
        / "student_0p5b",
    )
    model_cache = next(
        (path for path in model_cache_candidates if path.is_dir()), None
    )
    if model_cache is None:
        raise FileNotFoundError(
            "Stage 2A verified 0.5B model cache is absent; checked "
            + ", ".join(str(path) for path in model_cache_candidates)
        )
    return {"paths": paths, "chains": chains, "model_cache": model_cache}


def command_common(inputs: dict[str, Any], seed: int) -> list[str]:
    paths = inputs["paths"]
    chain = inputs["chains"][seed]
    return [
        "--lock", str(LOCK_PATH),
        "--geometry", str(paths["geometry"]),
        "--migrated", str(chain["migrated"]),
        "--migrated_sha256", MIGRATED_SHA[seed],
        "--p33", str(chain["p33"]),
        "--p33_sha256", P33_SHA[seed],
        "--i1", str(chain["i1"]),
        "--i1_sha256", I1_SHA[seed],
        "--p34", str(chain["p34"]),
        "--p34_sha256", P34_SHA[seed],
        "--p35", str(chain["p35"]),
        "--p35_sha256", P35_SHA[seed],
        "--model_cache", str(inputs["model_cache"]),
    ]


def train_and_score(
    *,
    arm: str,
    seed: int,
    inputs: dict[str, Any],
    control_escalation: bool = False,
) -> dict[str, Any]:
    output = PRIVATE_DIR / f"{arm}_seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if not summary_path.is_file() or read_json(summary_path).get("status") != "complete_dev_evaluation_pending":
        command = [
            sys.executable,
            "-u",
            "-m",
            "training.run_paper2_stage2a",
            "--arm", arm,
            "--seed", str(seed),
            *command_common(inputs, seed),
            "--teacher_lattice", str(inputs["paths"]["teacher_lattice"]),
            "--population_manifest", str(inputs["paths"]["population_manifest"]),
            "--owner_manifest", str(inputs["paths"]["owner_manifest"]),
            "--student_features", str(inputs["paths"]["student_features"]),
            "--output_dir", str(output),
        ]
        resume = latest_checkpoint(output)
        if resume is not None:
            command.extend(("--resume", str(resume)))
        if control_escalation:
            command.append("--control_escalation_authorized")
        run(command)

    evaluation = output / "evaluation"
    evaluation_summary = evaluation / "summary.json"
    if not evaluation_summary.is_file():
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_stage2a",
                "--arm", arm,
                "--seed", str(seed),
                *command_common(inputs, seed),
                "--panel", str(PANEL_PATH),
                "--base_scores", str(BASE_SCORES_PATH),
                "--checkpoint", str(output / "checkpoint_step_1200.pt"),
                "--output_dir", str(evaluation),
            ]
        )
    return read_json(evaluation_summary)


def main() -> int:
    for path in (PRIVATE_DIR, RECEIPT_DIR, LOCAL_DIR):
        path.mkdir(parents=True, exist_ok=True)
    status_path = RECEIPT_DIR / "status.json"

    def status(value: str, **details: Any) -> None:
        write_json(
            status_path,
            {
                "kind": "paper2_stage2a_colab_status_v1",
                "status": value,
                "updated_at_unix": time.time(),
                "confirm_scored": False,
                "eval_e_scored": False,
                **details,
            },
        )
        print(f"stage2a_status={value} details={details}", flush=True)

    try:
        lock = read_json(LOCK_PATH)
        assert_stage2a_training_authorized(lock)
        if sha256_file(PANEL_PATH) != lock["data_separation"]["panel_manifest_sha256"]:
            raise RuntimeError("Stage 2A frozen DEV panel SHA changed")
        status("staging_registered_inputs")
        inputs = stage_inputs(scratch_root())
        summaries: dict[str, Any] = {}
        schedule = [
            ("t3a", 0),
            ("t3a", 1),
            ("t3b", 0),
            ("t3b", 1),
            ("shuffled", 0),
            ("random", 0),
        ]
        for arm, seed in schedule:
            status("running_arm", arm=arm, seed=seed)
            summaries[f"{arm}_seed_{seed}"] = train_and_score(
                arm=arm, seed=seed, inputs=inputs
            )
        for arm in ("shuffled", "random"):
            delta = int(summaries[f"{arm}_seed_0"]["pooled"]["delta_rows"])
            if delta > 3:
                status("running_registered_control_escalation", arm=arm, seed=1, trigger_delta=delta)
                summaries[f"{arm}_seed_1"] = train_and_score(
                    arm=arm, seed=1, inputs=inputs, control_escalation=True
                )

        t3a = [int(summaries[f"t3a_seed_{seed}"]["pooled"]["delta_rows"]) for seed in (0, 1)]
        t3b = [int(summaries[f"t3b_seed_{seed}"]["pooled"]["delta_rows"]) for seed in (0, 1)]
        controls = {
            key: int(value["pooled"]["delta_rows"])
            for key, value in summaries.items()
            if key.startswith(("shuffled", "random"))
        }
        t3a_gate = sum(t3a) / 2 >= 8 and all(value > 0 for value in t3a)
        controls_flat = all(-3 <= value <= 3 for value in controls.values())
        verdict = (
            "T3_FULL_PROCEEDS"
            if t3a_gate and controls_flat
            else "SCREEN_POSITIVE_CONTROL_AMBIGUOUS"
            if t3a_gate
            else "SCREEN_BELOW_PROCEED_THRESHOLD"
        )
        aggregate = {
            "kind": "paper2_stage2a_t3_screen_summary_v1",
            "status": "complete_dev_only",
            "verdict": verdict,
            "t3a_delta_rows_by_seed": t3a,
            "t3a_mean_delta_rows": sum(t3a) / 2,
            "t3a_both_seeds_positive": all(value > 0 for value in t3a),
            "t3a_minimum_net_row_gain_pass": t3a_gate,
            "t3b_delta_rows_by_seed": t3b,
            "t3b_mean_delta_rows": sum(t3b) / 2,
            "control_delta_rows": controls,
            "controls_inside_equivalence_band": controls_flat,
            "arm_summaries": summaries,
            "confirm_scored": False,
            "eval_e_scored": False,
        }
        write_json(LOCAL_DIR / "summary.json", aggregate)
        shutil.copy2(LOCAL_DIR / "summary.json", RECEIPT_DIR / "summary.json")
        status("complete_dev_only", verdict=verdict, summary_sha256=sha256_file(LOCAL_DIR / "summary.json"))
        print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
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
