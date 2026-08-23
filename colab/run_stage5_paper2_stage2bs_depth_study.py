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

from colab.run_stage5_paper2_phase3_p34_a2 import (
    DRIVE_STAGE5,
    I1_ID,
    MIGRATED_SHA,
    MIGRATION_ID,
    P33_ID,
    P33_SHA,
    rsync,
)
from colab.run_stage5_paper2_phase3_p35 import I1_SHA, P34_ID, stage_chain
from colab.run_stage5_paper2_phase3_p35_amplitude_t1_preflight import P34_SHA, P35_ID, P35_SHA
from training.paper2_stage2bs_depth_study import (
    load_lock,
    resolve_direct_branch,
    resolve_final_cell,
    resolve_keys,
    sha256_file,
)


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


def stage_registered_chain(scratch: Path, seed: int) -> tuple[dict[str, Path], dict[str, Any]]:
    """Stage the locked checkpoint chain, allowing one SHA-exact retained mirror."""

    chain_root = scratch / f"chain_seed_{seed}"
    drive_p34 = (
        DRIVE_STAGE5 / P34_ID / f"private/main_seed_{seed}/checkpoint_step_4000.pt"
    )
    if drive_p34.is_file():
        chain = stage_chain(chain_root, seed=seed, expected_p34=P34_SHA[seed])
        return chain, {
            "p34_source": "drive_canonical",
            "source_path": str(drive_p34),
            "expected_sha256": P34_SHA[seed],
            "observed_sha256": sha256_file(chain["p34"]),
        }

    if seed != 1:
        raise FileNotFoundError(drive_p34)
    fallback_value = os.environ.get("STAGE2BS_SEED1_P34_FALLBACK", "").strip()
    if not fallback_value:
        raise RuntimeError(
            "Seed-1 P3.4 Drive artifact is absent and STAGE2BS_SEED1_P34_FALLBACK is unset"
        )
    fallback = Path(fallback_value)
    if not fallback.is_file():
        raise FileNotFoundError(fallback)
    fallback_sha = sha256_file(fallback)
    if fallback_sha != P34_SHA[seed]:
        raise RuntimeError(
            "Seed-1 P3.4 retained-mirror SHA mismatch: "
            f"expected={P34_SHA[seed]} observed={fallback_sha}"
        )

    chain = {
        "migrated": chain_root / f"seed_{seed}_migrated.pt",
        "p33": chain_root / f"seed_{seed}_p33_step_1000.pt",
        "i1": chain_root / f"seed_{seed}_i1.pt",
        "p34": chain_root / f"seed_{seed}_p34_step_4000.pt",
    }
    rsync(
        DRIVE_STAGE5
        / MIGRATION_ID
        / f"private/migrated_checkpoints/seed_{seed}_full_a2_phase3_migrated.pt",
        chain["migrated"],
    )
    rsync(DRIVE_STAGE5 / P33_ID / f"private/seed_{seed}/checkpoint_step_1000.pt", chain["p33"])
    rsync(DRIVE_STAGE5 / I1_ID / f"private/seed_{seed}/resume.pt", chain["i1"])
    chain["p34"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fallback, chain["p34"])
    expected = {
        "migrated": MIGRATED_SHA[seed],
        "p33": P33_SHA[seed],
        "i1": I1_SHA[seed],
        "p34": P34_SHA[seed],
    }
    for name, path in chain.items():
        observed = sha256_file(path)
        if observed != expected[name]:
            raise RuntimeError(
                f"Stage 2B-S staged {name} SHA mismatch: "
                f"expected={expected[name]} observed={observed}"
            )
    return chain, {
        "p34_source": "local_durable_sha_exact_fallback",
        "source_path": str(fallback),
        "missing_drive_path": str(drive_p34),
        "expected_sha256": P34_SHA[seed],
        "observed_sha256": fallback_sha,
    }


def model_args(scratch: Path, seed: int, receipts: Path) -> list[str]:
    chain, provenance = stage_registered_chain(scratch, seed)
    atomic_json(receipts / f"seed_{seed}/checkpoint_provenance.json", provenance)
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


def banked_preflight_inputs(
    *, result: Path, receipts: Path, seed: int
) -> tuple[Path, Path]:
    source = receipts / f"seed_{seed}/preflight.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing banked Stage 2B-S preflight receipt: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    retained = receipts / f"banked_preflight/seed_{seed}/preflight.json"
    if not str(payload.get("session_id", "")).strip() and retained.is_file():
        source = retained
        payload = json.loads(source.read_text(encoding="utf-8"))
    expected = [162, 10, 2, 2] if seed == 0 else [162, 9, 5, 1]
    if payload.get("observed_correct_by_k") != expected:
        raise RuntimeError(f"Banked Stage 2B-S seed-{seed} preflight changed")
    session = str(payload.get("session_id", "")).strip()
    if not session:
        raise RuntimeError(f"Banked Stage 2B-S seed-{seed} session is absent")
    private = result / f"private/seed_{seed}/preflight/{session}"
    if not private.is_dir():
        raise FileNotFoundError(private)
    retained.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != retained.resolve():
        shutil.copy2(source, retained)
    atomic_json(
        receipts / f"banked_preflight/seed_{seed}/provenance.json",
        {
            "kind": "paper2_stage2bs_banked_preflight_provenance_v1",
            "seed": seed,
            "source_receipt": str(source),
            "source_receipt_sha256": sha256_file(source),
            "retained_receipt": str(retained),
            "retained_receipt_sha256": sha256_file(retained),
            "private_source": str(private),
            "observed_correct_by_k": expected,
        },
    )
    return retained, private


def main() -> int:
    lock = load_lock(LOCK)
    mode = os.environ.get("STAGE2BS_DEPTH_MODE", "preflight").strip().lower()
    if mode not in {"preflight", "cascade_direct", "cascade_final", "run"}:
        raise RuntimeError(f"Unknown Stage 2B-S depth-study mode: {mode}")
    generation_batch_size = int(
        os.environ.get(
            "STAGE2BS_DEPTH_GENERATION_BATCH_SIZE",
            str(lock["runtime"]["generation_batch_size"]),
        )
    )
    margin_batch_size = int(
        os.environ.get(
            "STAGE2BS_DEPTH_MARGIN_BATCH_SIZE",
            str(lock["runtime"]["margin_batch_size"]),
        )
    )
    if generation_batch_size < 1 or margin_batch_size < 1:
        raise RuntimeError("Stage 2B-S batch sizes must be positive")
    scratch = scratch_root()
    result = result_root(scratch)
    configured_durable = os.environ.get("STAGE2BS_DEPTH_DURABLE_ROOT", "").strip()
    durable = Path(configured_durable) if configured_durable else DRIVE_STAGE5 / RUN_ID
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
                "durable_root": str(durable),
                "generation_batch_size": generation_batch_size,
                "margin_batch_size": margin_batch_size,
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
                "--generation_batch_size",
                str(generation_batch_size),
                "--margin_batch_size",
                str(margin_batch_size),
                *model_args(scratch, seed, receipts),
            ]
            if mode == "preflight":
                command.append("--preflight_only")
            elif mode in {"cascade_direct", "cascade_final"}:
                banked_receipt, banked_private = banked_preflight_inputs(
                    result=result, receipts=receipts, seed=seed
                )
                command.extend(
                    [
                        "--cascade_stage",
                        "direct" if mode == "cascade_direct" else "final",
                        "--banked_preflight_receipt",
                        str(banked_receipt),
                        "--banked_preflight_private",
                        str(banked_private),
                    ]
                )
            run(command)
            summary_path = output / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            expected_status = {
                "preflight": "preflight_pass_score_only",
                "cascade_direct": "cascade_direct_complete_score_only",
                "cascade_final": "cascade_final_complete_score_only",
                "run": "complete_score_only",
            }[mode]
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
        if mode == "cascade_direct":
            primary = [
                cell for summary in summaries for cell in summary["payload"]["cells"]
            ]
            decision = resolve_direct_branch(
                primary, native_k1_by_seed={0: 162, 1: 162}
            )
            retention = []
            for summary in summaries:
                seed = summary["seed"]
                retention.append(
                    {
                        "role": f"seed_{seed}_direct_summary",
                        "path": summary["path"],
                        "sha256": summary["sha256"],
                        "exists": Path(summary["path"]).is_file(),
                    }
                )
                for k in range(1, 5):
                    path = (
                        result
                        / f"private/seed_{seed}/cascade_direct/initialization/generation"
                        / f"deferred_terminal_write_no_reentry__k{k}__gamma_0p05.jsonl"
                    )
                    retention.append(
                        {
                            "role": f"seed_{seed}_direct_k{k}",
                            "path": str(path),
                            "sha256": sha256_file(path),
                            "exists": path.is_file(),
                        }
                    )
            if not all(row["exists"] for row in retention):
                raise RuntimeError("Stage 2B-S direct-discriminator retention failed")
            wave = {
                "kind": "paper2_stage2bs_cascade_direct_wave_v1",
                "status": decision["branch"],
                "session_id": session,
                "lock_sha256": sha256_file(LOCK),
                "registered_decision": decision,
                "seed_summaries": [
                    {key: row[key] for key in ("seed", "path", "sha256")}
                    for row in summaries
                ],
                "retention_verification": retention,
                "dev2_margin_rows_scored": 0,
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "confirm_scored": False,
                "eval_e_scored": False,
                "branch_execution_started": False,
            }
            atomic_json(receipts / "cascade_direct_wave.json", wave)
            mirror.sync()
            durable_wave = durable / "receipts/cascade_direct_wave.json"
            if not durable_wave.is_file() or sha256_file(durable_wave) != sha256_file(
                receipts / "cascade_direct_wave.json"
            ):
                raise RuntimeError("Stage 2B-S direct wave durability check failed")
            status(
                decision["branch"].lower(),
                cascade_direct_sha256=sha256_file(receipts / "cascade_direct_wave.json"),
            )
            mirror.close()
            print(json.dumps(wave, indent=2, sort_keys=True))
            return 0
        if mode == "cascade_final":
            primary = [
                cell for summary in summaries for cell in summary["payload"]["cells"]
            ]
            decision = resolve_final_cell(primary)
            margin_summaries = []
            if decision["score_registered_deferred_margins"]:
                for seed in (0, 1):
                    status("running_registered_final_margins", seed=seed)
                    output = receipts / f"seed_{seed}_final_margins"
                    private = result / f"private/seed_{seed}"
                    banked_receipt, banked_private = banked_preflight_inputs(
                        result=result, receipts=receipts, seed=seed
                    )
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
                        "--generation_batch_size",
                        str(generation_batch_size),
                        "--margin_batch_size",
                        str(margin_batch_size),
                        *model_args(scratch, seed, receipts),
                        "--cascade_stage",
                        "final_margins",
                        "--banked_preflight_receipt",
                        str(banked_receipt),
                        "--banked_preflight_private",
                        str(banked_private),
                    ]
                    run(command)
                    summary_path = output / "summary.json"
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    if payload.get("status") != "cascade_final_margins_complete_score_only":
                        raise RuntimeError(
                            f"Stage 2B-S seed {seed} final margins did not complete"
                        )
                    margin_summaries.append(
                        {
                            "seed": seed,
                            "path": str(summary_path),
                            "sha256": sha256_file(summary_path),
                            "payload": payload,
                        }
                    )
                    mirror.sync()
            retention = []
            for summary in summaries:
                seed = summary["seed"]
                retention.append(
                    {
                        "role": f"seed_{seed}_final_summary",
                        "path": summary["path"],
                        "sha256": summary["sha256"],
                        "exists": Path(summary["path"]).is_file(),
                    }
                )
                for k in range(1, 5):
                    path = (
                        result
                        / f"private/seed_{seed}/cascade_final/initialization/generation"
                        / f"per_loop_write_no_reentry__k{k}__gamma_0p05.jsonl"
                    )
                    retention.append(
                        {
                            "role": f"seed_{seed}_final_k{k}",
                            "path": str(path),
                            "sha256": sha256_file(path),
                            "exists": path.is_file(),
                        }
                    )
            for summary in margin_summaries:
                seed = summary["seed"]
                retention.append(
                    {
                        "role": f"seed_{seed}_margin_summary",
                        "path": summary["path"],
                        "sha256": summary["sha256"],
                        "exists": Path(summary["path"]).is_file(),
                    }
                )
                for k in (1, 4):
                    path = (
                        result
                        / f"private/seed_{seed}/cascade_final_margins/initialization/margins"
                        / f"deferred_terminal_write_no_reentry__k{k}__gamma_0p05.jsonl"
                    )
                    retention.append(
                        {
                            "role": f"seed_{seed}_deferred_margin_k{k}",
                            "path": str(path),
                            "sha256": sha256_file(path),
                            "exists": path.is_file(),
                        }
                    )
            if not all(row["exists"] for row in retention):
                raise RuntimeError("Stage 2B-S final-cell retention failed")
            final_status = (
                "SCHEDULE_NEUTRALIZED_MARGIN_BANKED"
                if decision["score_registered_deferred_margins"]
                else decision["verdict"]
            )
            wave = {
                "kind": "paper2_stage2bs_cascade_final_wave_v1",
                "status": final_status,
                "session_id": session,
                "lock_sha256": sha256_file(LOCK),
                "registered_decision": decision,
                "seed_summaries": [
                    {key: row[key] for key in ("seed", "path", "sha256")}
                    for row in summaries
                ],
                "margin_summaries": [
                    {key: row[key] for key in ("seed", "path", "sha256")}
                    for row in margin_summaries
                ],
                "margin_cells": [
                    cell
                    for summary in margin_summaries
                    for cell in summary["payload"]["margin_cells"]
                ],
                "retention_verification": retention,
                "partial_interleave_executed": False,
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "confirm_scored": False,
                "eval_e_scored": False,
            }
            final_path = receipts / "cascade_final_wave.json"
            atomic_json(final_path, wave)
            mirror.sync()
            durable_wave = durable / "receipts/cascade_final_wave.json"
            if not durable_wave.is_file() or sha256_file(durable_wave) != sha256_file(
                final_path
            ):
                raise RuntimeError("Stage 2B-S final wave durability check failed")
            status(
                final_status.lower(),
                cascade_final_sha256=sha256_file(final_path),
            )
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
