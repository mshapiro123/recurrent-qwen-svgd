"""Run the terminal additive-versus-FiLM oracle re-entry interface probe."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab import run_stage5_phase_g_alpha as alpha  # noqa: E402
from colab.run_stage5_phase_g_multitarget_control import publish_receipts  # noqa: E402
from training.oracle_interface_probe_spec import (  # noqa: E402
    LOCKED_CONTROL_GROUPS,
    LOCKED_CONTROL_ROWS,
    LOCKED_CONTROL_TRANSITIONS,
    LOCKED_ROUTES,
    preregistration_payload,
)


SOURCE_RUN_ID = "stage5_phase_g_multitarget_control_20260718"
SOURCE_DIR = ROOT / "outputs" / "stage5" / SOURCE_RUN_ID
SOURCE_SUMMARY = SOURCE_DIR / "summary.json"
TRAIN_JSONL = SOURCE_DIR / "data" / "train.jsonl"
CONTROL_JSONL = SOURCE_DIR / "data" / "posterior_control.jsonl"


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    printable = "$ " + " ".join(map(str, command))
    print(printable, flush=True)
    alpha.append_runtime_transcript(printable + "\n")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        alpha.append_runtime_transcript(line)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def validate_source() -> None:
    if not SOURCE_SUMMARY.exists() or not TRAIN_JSONL.exists() or not CONTROL_JSONL.exists():
        raise FileNotFoundError("Missing committed Phase G A0 multi-target source receipts")
    source = read_json(SOURCE_SUMMARY)
    if source.get("status") != "blocked_posterior_control_after_confirmation":
        raise AssertionError("Oracle probe requires the ratified blocked A0 source")
    if source.get("keeper_sha256") != alpha.KEEPER_SHA256:
        raise AssertionError("Oracle probe source keeper differs from the locked keeper")
    train_rows = read_jsonl(TRAIN_JSONL)
    control_rows = read_jsonl(CONTROL_JSONL)
    if len(train_rows) != 1899:
        raise AssertionError("Oracle probe requires the committed 1,899-row training set")
    if len(control_rows) != LOCKED_CONTROL_ROWS:
        raise AssertionError("Oracle probe requires the committed 106-row control set")
    if len({str(row["base_problem_id"]) for row in control_rows}) != LOCKED_CONTROL_GROUPS:
        raise AssertionError("Oracle probe requires the committed 32 control groups")
    if sum(int(row["depth"]) for row in control_rows) != LOCKED_CONTROL_TRANSITIONS:
        raise AssertionError("Oracle probe requires the committed 305 transitions")


def restore_if_present(source: Path, destination: Path) -> None:
    if destination.exists() or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def train_arm(
    *,
    route: str,
    keeper: Path,
    run_dir: Path,
    drive_checkpoint_dir: Path,
    drive_artifacts: Path,
    steps: int,
    seed: int,
    bottleneck_dim: int,
    dtype: str,
) -> dict[str, Any]:
    train_dir = run_dir / "train" / route
    raw_path = train_dir / f"oracle_{route}_raw_step_{steps}.pt"
    ema_path = train_dir / f"oracle_{route}_ema_step_{steps}.pt"
    summary_path = train_dir / "summary.json"
    drive_raw = drive_checkpoint_dir / f"{route}_raw_step_{steps}.pt"
    drive_ema = drive_checkpoint_dir / f"{route}_ema_step_{steps}.pt"
    restore_if_present(drive_raw, raw_path)
    restore_if_present(drive_ema, ema_path)
    if not (summary_path.exists() and raw_path.exists() and ema_path.exists()):
        progress_path = train_dir / "training_progress.pt"
        progress_backup = drive_checkpoint_dir / f"{route}_training_progress.pt"
        run(
            [
                sys.executable,
                "training/train_oracle_interface_probe.py",
                "--train_jsonl",
                str(TRAIN_JSONL.relative_to(ROOT)),
                "--keeper",
                str(keeper),
                "--expected_keeper_sha256",
                alpha.KEEPER_SHA256,
                "--output_dir",
                str(train_dir.relative_to(ROOT)),
                "--route",
                route,
                "--steps",
                str(steps),
                "--seed",
                str(seed),
                "--bottleneck_dim",
                str(bottleneck_dim),
                "--checkpoint_every",
                "100",
                "--progress_checkpoint",
                str(progress_path),
                "--progress_backup_path",
                str(progress_backup),
                "--progress_backup_dir",
                str(drive_artifacts / "train" / route),
                "--dtype",
                dtype,
                "--device",
                "cuda",
            ]
        )
    if not (summary_path.exists() and raw_path.exists() and ema_path.exists()):
        raise AssertionError(f"Oracle {route} training did not produce complete checkpoints")
    drive_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, drive_raw)
    shutil.copy2(ema_path, drive_ema)
    summary = read_json(summary_path)
    if summary.get("status") != "finished" or summary.get("route") != route:
        raise AssertionError(f"Oracle {route} training summary is invalid")
    config = dict(summary.get("config") or {})
    if not config.get("step_zero_identity_exact"):
        raise AssertionError(f"Oracle {route} failed its step-zero identity gate")
    if int(config.get("frozen_gradient_assertions", -1)) != steps:
        raise AssertionError(f"Oracle {route} did not assert frozen gradients every step")
    if int(config.get("gradient_liveness_assertions", -1)) != steps:
        raise AssertionError(f"Oracle {route} did not assert conditioner liveness every step")
    if config.get("active_lineage_sha256_start") != config.get(
        "active_lineage_sha256_end"
    ):
        raise AssertionError(f"Oracle {route} changed the frozen keeper lineage")
    summary["ema_checkpoint_sha256"] = sha256_file(ema_path)
    alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
    publish_receipts(
        run_dir,
        f"Record Phase G oracle interface training {route} {run_dir.name} [skip ci]",
    )
    return summary


def assert_matched_training(additive: dict[str, Any], film: dict[str, Any]) -> None:
    additive_config = dict(additive["config"])
    film_config = dict(film["config"])
    matched_fields = (
        "keeper_sha256",
        "steps",
        "learning_rate",
        "weight_decay",
        "ema_decay",
        "bottleneck_dim",
        "seed",
        "max_length",
        "sampling_policy",
        "trainable_parameter_count",
    )
    mismatches = {
        field: (additive_config.get(field), film_config.get(field))
        for field in matched_fields
        if additive_config.get(field) != film_config.get(field)
    }
    if mismatches:
        raise AssertionError(f"Oracle arm matching failed: {mismatches}")
    additive_rows = [
        json.loads(line)["row_id"]
        for line in Path(additive["training_trace"]).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    film_rows = [
        json.loads(line)["row_id"]
        for line in Path(film["training_trace"]).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if additive_rows != film_rows:
        raise AssertionError("Oracle arms did not receive the same sampled-row sequence")


def main() -> int:
    run_id = os.environ.get(
        "STAGE5_ORACLE_INTERFACE_RUN_ID",
        "stage5_phase_g_oracle_interface_probe_20260718",
    )
    steps = int(os.environ.get("STAGE5_ORACLE_INTERFACE_STEPS", "1500"))
    seed = int(os.environ.get("STAGE5_ORACLE_INTERFACE_SEED", "20260718"))
    bottleneck_dim = int(
        os.environ.get("STAGE5_ORACLE_INTERFACE_BOTTLENECK_DIM", "256")
    )
    dtype = os.environ.get("STAGE5_ORACLE_INTERFACE_DTYPE", "bfloat16")
    if steps != 1500 or seed != 20260718 or bottleneck_dim != 256:
        raise AssertionError(
            "Oracle interface steps, seed, and bottleneck are preregistered"
        )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    drive_artifacts = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5") / run_id
    )
    drive_checkpoint_dir = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints") / run_id
    )
    if drive_artifacts.exists():
        shutil.copytree(drive_artifacts, run_dir, dirs_exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    alpha.configure_runtime_transcript(run_dir / "runtime.log")
    validate_source()
    keeper = alpha.restore_keeper(run_dir)
    preregistration = preregistration_payload()
    alpha.write_json(run_dir / "preregistration.json", preregistration)
    summary: dict[str, Any] = {
        "kind": "stage5_phase_g_oracle_interface_probe",
        "status": "started",
        "run_id": run_id,
        "source_summary": str(SOURCE_SUMMARY.relative_to(ROOT)),
        "source_verdict": "NO-CHANNEL ratified",
        "keeper_sha256": alpha.KEEPER_SHA256,
        "routes": list(LOCKED_ROUTES),
        "steps": steps,
        "seed": seed,
        "bottleneck_dim": bottleneck_dim,
        "training_rows": 1899,
        "heldout_rows": LOCKED_CONTROL_ROWS,
        "heldout_groups": LOCKED_CONTROL_GROUPS,
        "heldout_transitions": LOCKED_CONTROL_TRANSITIONS,
        "variational_training_performed": False,
        "coverage_performed": False,
        "automatic_successor_authorized": False,
        "arms": [],
    }
    alpha.write_json(run_dir / "summary.json", summary)
    alpha.write_runtime_status(
        run_dir,
        drive_artifacts,
        stage="startup",
        status="started",
        run_id=run_id,
    )
    publish_receipts(
        run_dir,
        f"Preregister Phase G oracle interface probe {run_id} [skip ci]",
    )

    training: dict[str, dict[str, Any]] = {}
    for route in LOCKED_ROUTES:
        training[route] = train_arm(
            route=route,
            keeper=keeper,
            run_dir=run_dir,
            drive_checkpoint_dir=drive_checkpoint_dir,
            drive_artifacts=drive_artifacts,
            steps=steps,
            seed=seed,
            bottleneck_dim=bottleneck_dim,
            dtype=dtype,
        )
    assert_matched_training(training["additive"], training["film"])

    arm_summaries: list[Path] = []
    for route in LOCKED_ROUTES:
        train_summary = training[route]
        checkpoint = Path(train_summary["ema_checkpoint"])
        checkpoint_sha = sha256_file(checkpoint)
        eval_dir = run_dir / "eval" / route
        cache_path = drive_artifacts / "eval_cache" / route / "rows.jsonl"
        run(
            [
                sys.executable,
                "eval/eval_oracle_interface_probe.py",
                "--data_jsonl",
                str(CONTROL_JSONL.relative_to(ROOT)),
                "--keeper",
                str(keeper),
                "--expected_keeper_sha256",
                alpha.KEEPER_SHA256,
                "--conditioner_checkpoint",
                str(checkpoint),
                "--expected_conditioner_sha256",
                checkpoint_sha,
                "--route",
                route,
                "--output_dir",
                str(eval_dir.relative_to(ROOT)),
                "--resume_cache_path",
                str(cache_path),
                "--bottleneck_dim",
                str(bottleneck_dim),
                "--dtype",
                dtype,
                "--device",
                "cuda",
            ]
        )
        arm_summary = eval_dir / "summary.json"
        arm_summaries.append(arm_summary)
        summary["arms"].append(
            {
                "route": route,
                "training_summary": str(
                    (run_dir / "train" / route / "summary.json").relative_to(ROOT)
                ),
                "checkpoint_sha256": checkpoint_sha,
                "eval_summary": str(arm_summary.relative_to(ROOT)),
            }
        )
        alpha.write_json(run_dir / "summary.json", summary)
        alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
        publish_receipts(
            run_dir,
            f"Record Phase G oracle interface eval {route} {run_id} [skip ci]",
        )

    gate_json = run_dir / "gate.json"
    gate_md = run_dir / "gate.md"
    run(
        [
            sys.executable,
            "eval/score_oracle_interface_probe.py",
            "--arm_summaries",
            *[str(path.relative_to(ROOT)) for path in arm_summaries],
            "--output_json",
            str(gate_json.relative_to(ROOT)),
            "--output_md",
            str(gate_md.relative_to(ROOT)),
        ]
    )
    gate = read_json(gate_json)
    summary["gate"] = gate
    summary["status"] = "finished_terminal_probe"
    summary["automatic_successor_authorized"] = False
    alpha.write_json(run_dir / "summary.json", summary)
    alpha.write_runtime_status(
        run_dir,
        drive_artifacts,
        stage="terminal_gate",
        status="finished_terminal_probe",
        measured_reading=gate["measured_reading"],
        automatic_successor_authorized=False,
    )
    alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
    publish_receipts(
        run_dir,
        f"Record Phase G oracle interface terminal probe {run_id} [skip ci]",
    )
    return 0


def guarded_main() -> int:
    run_id = os.environ.get(
        "STAGE5_ORACLE_INTERFACE_RUN_ID",
        "stage5_phase_g_oracle_interface_probe_20260718",
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    drive_artifacts = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5") / run_id
    )
    try:
        return main()
    except BaseException as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        alpha.record_runtime_failure(run_dir, drive_artifacts, exc)
        raise
    finally:
        alpha.configure_runtime_transcript(None)


if __name__ == "__main__":
    raise SystemExit(guarded_main())
