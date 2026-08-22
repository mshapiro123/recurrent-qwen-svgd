"""Stage and run the locked Stage 2B-S depth-capability study without training."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from colab.run_stage5_paper2_phase3_p34_a2 import DRIVE_STAGE5, MIGRATED_SHA, P33_SHA, rsync
from colab.run_stage5_paper2_phase3_p35 import I1_SHA, stage_chain
from colab.run_stage5_paper2_phase3_p35_amplitude_t1_preflight import P34_SHA, P35_ID, P35_SHA
from training.paper2_stage2bs_depth_study import load_lock, resolve_keys, sha256_file


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_stage2bs_depth_study_20260822"
SOURCE_RUN_ID = "stage5_paper2_stage2b_depth_20260819"
LOCK = ROOT / "training/paper2_stage2bs_depth_study_lock.json"
PANEL = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
BASE_SCORES = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"
REFERENCE_ROWS = DRIVE_STAGE5 / "stage5_paper2_phase3_p31_completion_20260810/private/p31_partitioned_rows.jsonl"
DRIVE_SOURCE = DRIVE_STAGE5 / SOURCE_RUN_ID
DEV2_SHA = "6b9ebf40128ed21b0351710e9f828bcacb096512704f02f34274a3b8adcc0adb"
STOP_SHA = {
    0: "50cbf437adda668812dbe53a015792d3dc8ebc02cb785fba594c512b64bf2f58",
    1: "830bbfa11dca4d3b9ed56db96a7c40c887f56fb4a5227555edc1bd447b6662bc",
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
    raise RuntimeError("Stage 2B-S depth study requires at least 80 GiB local scratch")


def result_root(scratch: Path) -> Path:
    configured = os.environ.get("STAGE2BS_DEPTH_RESULT_ROOT", "").strip()
    root = Path(configured) if configured else scratch / "result"
    root.mkdir(parents=True, exist_ok=True)
    return root


class DurableMirror:
    """Mirror atomic local receipts to Drive without putting hot I/O on DriveFS."""

    def __init__(self, source: Path, destination: Path, *, interval_seconds: int = 300) -> None:
        self.source = source
        self.destination = destination
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def sync(self) -> None:
        if not self.destination.parent.is_dir() or not os.access(self.destination.parent, os.W_OK):
            raise RuntimeError("Stage 2B-S durable Drive destination is not writable")
        self.destination.mkdir(parents=True, exist_ok=True)
        with self.lock:
            subprocess.run(
                [
                    "rsync",
                    "--archive",
                    "--partial",
                    str(self.source) + os.sep,
                    str(self.destination) + os.sep,
                ],
                cwd=ROOT,
                check=True,
            )
        print(f"stage2bs_depth_durable_sync destination={self.destination}", flush=True)

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self.sync()

    def start(self) -> None:
        self.sync()
        self.thread = threading.Thread(target=self._loop, name="stage2bs-drive-mirror", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=30)
        self.sync()


def session_id() -> str:
    gpu_uuid = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()
    return f"{platform.node()}__{gpu_uuid}".replace("/", "_")


def model_args(scratch: Path, seed: int) -> list[str]:
    chain = stage_chain(scratch / f"chain_seed_{seed}", seed=seed, expected_p34=P34_SHA[seed])
    p35 = scratch / f"seed_{seed}_p35_ema_step_4400.pt"
    rsync(DRIVE_STAGE5 / P35_ID / f"private/arm_s_seed_{seed}/ema_step_4400.pt", p35)
    if sha256_file(p35) != P35_SHA[seed]:
        raise RuntimeError("Stage 2B-S P3.5 endpoint changed")
    stop = scratch / f"seed_{seed}_ema_step_01000.pt"
    rsync(DRIVE_SOURCE / f"private/seed_{seed}/ema_step_01000.pt", stop)
    if sha256_file(stop) != STOP_SHA[seed]:
        raise RuntimeError("Stage 2B-S step-1000 endpoint changed")
    return [
        "--migrated", str(chain["migrated"]), "--migrated_sha256", MIGRATED_SHA[seed],
        "--p33", str(chain["p33"]), "--p33_sha256", P33_SHA[seed],
        "--i1", str(chain["i1"]), "--i1_sha256", I1_SHA[seed],
        "--p34", str(chain["p34"]), "--p34_sha256", P34_SHA[seed],
        "--p35", str(p35), "--p35_sha256", P35_SHA[seed],
        "--stop_checkpoint", str(stop), "--stop_sha256", STOP_SHA[seed],
        "--model_cache", str(scratch / "hf_student_cache"),
    ]


def main() -> int:
    lock = load_lock(LOCK)
    mode = os.environ.get("STAGE2BS_DEPTH_MODE", "preflight").strip().lower()
    if mode not in {"preflight", "run"}:
        raise RuntimeError(f"Unknown Stage 2B-S depth-study mode: {mode}")
    scratch = scratch_root()
    result = result_root(scratch)
    durable = DRIVE_STAGE5 / RUN_ID
    if durable.is_dir():
        rsync(durable, result)
    mirror = DurableMirror(result, durable)
    mirror.start()
    receipts = result / "receipts"
    status_path = receipts / "status.json"
    session = session_id()

    def status(value: str, **details: Any) -> None:
        atomic_json(
            status_path,
            {
                "kind": "paper2_stage2bs_depth_study_status_v1",
                "status": value,
                "session_id": session,
                "updated_at_unix": time.time(),
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "confirm_scored": False,
                "eval_e_scored": False,
                **details,
            },
        )
        print(f"stage2bs_depth_status={value} details={details}", flush=True)

    try:
        status("staging_registered_inputs")
        reference = scratch / "p31_partitioned_rows.jsonl"
        rsync(REFERENCE_ROWS, reference)
        dev2 = scratch / "dev2_manifest.jsonl"
        rsync(DRIVE_SOURCE / "private/dev2/dev2_manifest.jsonl", dev2)
        if sha256_file(dev2) != DEV2_SHA:
            raise RuntimeError("Stage 2B-S DEV-2 manifest changed")
        summaries = []
        for seed in (0, 1):
            status("running_seed", seed=seed)
            output = receipts / f"seed_{seed}"
            private = result / f"private/seed_{seed}"
            command = [
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_stage2bs_depth_study",
                "--seed",
                str(seed),
                "--lock",
                str(LOCK),
                "--dev1_panel",
                str(PANEL),
                "--dev2_manifest",
                str(dev2),
                "--reference_rows",
                str(reference),
                "--base_scores",
                str(BASE_SCORES),
                "--output_dir",
                str(output),
                "--private_dir",
                str(private),
                "--session_id",
                session,
                *model_args(scratch, seed),
            ]
            if mode == "preflight":
                command.append("--preflight_only")
            run(command)
            summary_path = output / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            expected_status = (
                "preflight_pass_score_only" if mode == "preflight" else "complete_score_only"
            )
            if payload.get("status") != expected_status:
                raise RuntimeError(f"Stage 2B-S seed {seed} did not complete")
            summaries.append(
                {
                    "seed": seed,
                    "path": str(summary_path),
                    "sha256": sha256_file(summary_path),
                    "payload": payload,
                }
            )
            mirror.sync()
        if mode == "preflight":
            wave = {
                "kind": "paper2_stage2bs_depth_preflight_wave_v1",
                "status": "PASS_AWAITING_REQUIRED_RELAY",
                "session_id": session,
                "lock_sha256": sha256_file(LOCK),
                "observed_native_counts": {
                    str(row["seed"]): row["payload"]["preflight"]["observed_correct_by_k"]
                    for row in summaries
                },
                "expected_native_counts": lock["expected_native_counts"],
                "variant_cells_scored": 0,
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "confirm_scored": False,
                "eval_e_scored": False,
            }
            atomic_json(receipts / "preflight_wave.json", wave)
            status("preflight_pass_awaiting_required_relay")
            mirror.close()
            print(json.dumps(wave, indent=2, sort_keys=True))
            return 0
        primary = [
            cell
            for summary in summaries
            for cell in summary["payload"]["cells"]
            if cell["endpoint"] == "initialization"
        ]
        keys = resolve_keys(primary, native_k1_by_seed={0: 162, 1: 162})
        retention = []
        for summary in summaries:
            seed = summary["seed"]
            retention.extend(
                [
                    {
                        "role": f"seed_{seed}_summary",
                        "path": summary["path"],
                        "sha256": summary["sha256"],
                        "exists": Path(summary["path"]).is_file(),
                    },
                    {
                        "role": f"seed_{seed}_preflight",
                        "path": str(receipts / f"seed_{seed}/preflight.json"),
                        "sha256": sha256_file(receipts / f"seed_{seed}/preflight.json"),
                        "exists": (receipts / f"seed_{seed}/preflight.json").is_file(),
                    },
                ]
            )
        wave = {
            "kind": "paper2_stage2bs_depth_study_wave_v1",
            "status": "complete_score_only",
            "session_id": session,
            "lock_sha256": sha256_file(LOCK),
            "seed_summaries": [
                {key: row[key] for key in ("seed", "path", "sha256")} for row in summaries
            ],
            "registered_keys": keys,
            "retention_verification": retention,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
        }
        if not all(row["exists"] for row in retention):
            raise RuntimeError("Stage 2B-S retention verification failed")
        atomic_json(receipts / "summary.json", wave)
        mirror.sync()
        durable_summary = durable / "receipts/summary.json"
        if not durable_summary.is_file() or sha256_file(durable_summary) != sha256_file(
            receipts / "summary.json"
        ):
            raise RuntimeError("Stage 2B-S durable summary verification failed")
        status("complete_score_only_release_gpu", summary_sha256=sha256_file(receipts / "summary.json"))
        mirror.close()
        print(json.dumps(wave, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        status(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        mirror.close()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
