"""Prepare, train, evaluate, and publish the locked Paper Two T1-lite run."""

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

from training.internal_think_token_t1 import augment_control_row, build_pilot_mixture_rows
from training.internal_think_token_t1 import phase_t1_seed_1_trigger
from training.internal_think_token_t1_spec import phase_t1_locked, validate_locked_phase_t1
from training.synthetic_depth_task import SyntheticDepthConfig, write_synthetic_depth_dataset
from colab.run_stage5_depth_support_ladder import manifest_for_rows


RUN_ID = os.environ.get("STAGE5_PAPER2_T1_LITE_RUN_ID", "stage5_paper2_t1_lite_20260724")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
DRIVE_ROOT = Path(
    os.environ.get(
        "STAGE5_PAPER2_T1_LITE_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/{RUN_ID}",
    )
)
TRAIN_SOURCE = ROOT / "outputs/stage5/stage5_depth_support_ladder8_20260705_204923/data/train_chain_symbol_sft.jsonl"
FROZEN_EVAL = ROOT / "outputs/stage5/stage5_synthetic_depth_frozen_eval_v2_depth14/data/test_chain_mcq.jsonl"
CANARY = ROOT / "outputs/stage5/stage5_adapter_budget_arm_e_20260718/data/base_capability_canary_64.jsonl"
T0_RECEIPT = ROOT / "outputs/stage5/stage5_paper2_internal_token_t0_preflight_20260722/summary.json"
REFERENCE_RECEIPT = ROOT / "outputs/stage5/stage5_phase_a_surpass_receipt_20260714/summary.json"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_for_cli(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(value)


def run(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    result = subprocess.CompletedProcess(command, process.wait(), "".join(lines), None)
    if result.returncode not in allowed:
        print("T1_LITE_FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(result.stdout.splitlines()[-300:]), flush=True)
        print("T1_LITE_FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result


def publish(message: str) -> None:
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in sorted(RUN_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md", ".log"}:
            continue
        if path.name.endswith("control.jsonl") or path.name == "causal_override_progress.jsonl":
            continue
        subprocess.run(["git", "add", "-f", path.relative_to(ROOT).as_posix()], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    if subprocess.run(["git", "push", "origin", "main"], cwd=ROOT).returncode:
        run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
        run(["git", "push", "origin", "main"])


def prepare_registered_data(run_dir: Path, *, seed: int) -> dict[str, Any]:
    required = [TRAIN_SOURCE, FROZEN_EVAL, CANARY, T0_RECEIPT, REFERENCE_RECEIPT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"T1-lite immutable source artifacts missing: {missing}")
    data_dir = run_dir / "data"
    source_rows = read_jsonl(TRAIN_SOURCE)
    mixture, mixture_manifest = build_pilot_mixture_rows(source_rows, seed=int(seed))
    train_path = data_dir / "t1_lite_train_70_30.jsonl"
    write_jsonl(train_path, mixture)
    if mixture_manifest["control_rows"] != 1400 or mixture_manifest["rehearsal_rows"] != 600:
        raise AssertionError(f"T1-lite 70/30 mixture drifted: {mixture_manifest}")

    pilot_root = data_dir / "pilot_source"
    write_synthetic_depth_dataset(
        output_dir=pilot_root,
        config=SyntheticDepthConfig(
            n_symbols=16, max_depth=8, rows_per_depth=32, seed=2026072399,
            num_choices=4, max_target_loops=8, value_prefix="letter:",
        ),
    )
    pilot_path = data_dir / "liveness_pilot_256.jsonl"
    write_jsonl(pilot_path, [augment_control_row(row) for row in read_jsonl(pilot_root / "test_chain_symbol_sft.jsonl")])

    calibration_root = data_dir / "calibration_source"
    write_synthetic_depth_dataset(
        output_dir=calibration_root,
        config=SyntheticDepthConfig(
            n_symbols=16, max_depth=8, rows_per_depth=64, seed=2026072401,
            num_choices=4, max_target_loops=8, value_prefix="letter:",
        ),
    )
    calibration_rows = read_jsonl(calibration_root / "test_chain_mcq.jsonl")
    for row in calibration_rows:
        row["id"] = "t1_calibration_" + str(row["id"])
    calibration_path = data_dir / "calibration_512.jsonl"
    write_jsonl(calibration_path, calibration_rows)
    calibration_manifest = manifest_for_rows(calibration_rows, max_depth=8)
    registered_calibration = phase_t1_locked()["evaluation"]["calibration"]
    for key in ("row_id_sha256", "row_sha256"):
        if calibration_manifest[key] != registered_calibration[key]:
            raise AssertionError(
                f"T1-lite calibration manifest mismatch for {key}: "
                f"{calibration_manifest[key]} != {registered_calibration[key]}"
            )
    return {
        "train": path_for_cli(train_path),
        "train_sha256": sha256_file(train_path),
        "train_manifest": mixture_manifest,
        "pilot": path_for_cli(pilot_path),
        "pilot_sha256": sha256_file(pilot_path),
        "calibration": path_for_cli(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_manifest": calibration_manifest,
        "frozen_eval": path_for_cli(FROZEN_EVAL),
        "canary": path_for_cli(CANARY),
    }


def prepare_data() -> dict[str, Any]:
    return prepare_registered_data(RUN_DIR, seed=0)


def copy_eval_from_drive(label: str) -> Path:
    source = DRIVE_ROOT / "eval" / label
    destination = RUN_DIR / "eval" / label
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.jsonl"))
    return destination


def main() -> int:
    prereg = phase_t1_locked()
    validate_locked_phase_t1(prereg)
    for required_commit in ("44459f30", "8ea5ce64"):
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", required_commit, "HEAD"],
            cwd=ROOT,
        ).returncode:
            raise RuntimeError(f"T1-lite checkout does not contain required pre-training commit {required_commit}")
    t0 = read_json(T0_RECEIPT)
    if t0.get("status") != "passed_all_five_contracts":
        raise RuntimeError(f"T1-lite Phase T0 receipt is not green: {t0.get('status')}")
    reference = read_json(REFERENCE_RECEIPT)
    arm_a = reference.get("checkpoint_receipts", {}).get("A", {})
    expected_reference_sha = prereg["fresh_base_lineages"]["full_block"]["nonhalting_reference"]["checkpoint_sha256"]
    if arm_a.get("status") != "verified" or arm_a.get("sha256") != expected_reference_sha:
        raise RuntimeError(f"T1-lite full-block reference receipt mismatch: {arm_a}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(RUN_DIR / "preregistration.json", prereg)
    data = prepare_data()
    write_json(RUN_DIR / "data_manifest.json", data)
    training_dir = RUN_DIR / "train"
    result = run(
        [
            sys.executable, "training/run_internal_think_token_t1_lite.py",
            "--train_jsonl", data["train"],
            "--pilot_jsonl", data["pilot"],
            "--canary_jsonl", data["canary"],
            "--output_dir", path_for_cli(training_dir),
            "--backup_dir", str(DRIVE_ROOT / "checkpoints"),
            "--device", os.environ.get("DEVICE", "cuda"),
            "--dtype", os.environ.get("STAGE5_PAPER2_T1_LITE_DTYPE", "bfloat16"),
            "--attn_implementation", os.environ.get("ATTN_IMPLEMENTATION", "default"),
            "--seed", "0",
        ],
        allowed=(0, 2),
    )
    publish(f"Record T1-lite training and stage receipts {RUN_ID} [skip ci]")
    if result.returncode == 2:
        return 2
    training = read_json(training_dir / "training_summary.json")
    eval_specs = (
        ("ema_primary", training["ema_checkpoint"], True),
        ("raw_secondary", training["raw_checkpoint"], False),
    )
    evaluations: dict[str, Any] = {}
    for label, checkpoint, causal in eval_specs:
        drive_eval = DRIVE_ROOT / "eval" / label
        command = [
            sys.executable, "eval/eval_internal_think_token_t1_lite.py",
            "--checkpoint", checkpoint,
            "--gated_jsonl", data["frozen_eval"],
            "--extrapolation_jsonl", data["frozen_eval"],
            "--calibration_jsonl", data["calibration"],
            "--output_dir", str(drive_eval),
            "--device", os.environ.get("DEVICE", "cuda"),
            "--dtype", os.environ.get("STAGE5_PAPER2_T1_LITE_DTYPE", "bfloat16"),
            "--attn_implementation", os.environ.get("ATTN_IMPLEMENTATION", "default"),
            "--batch_size", os.environ.get("STAGE5_PAPER2_T1_LITE_EVAL_BATCH_SIZE", "8"),
        ]
        if causal:
            command.extend(
                [
                    "--run_causal_sweep",
                    "--causal_progress_path", str(DRIVE_ROOT / "causal_override_progress.jsonl"),
                ]
            )
        run(command)
        local_eval = copy_eval_from_drive(label)
        evaluations[label] = read_json(local_eval / "summary.json")
        publish(f"Record T1-lite {label} evaluation {RUN_ID} [skip ci]")
    verdict = evaluations["ema_primary"]["registered_gates"]
    seed_1 = phase_t1_seed_1_trigger(verdict)
    summary = {
        "kind": "stage5_paper2_t1_lite",
        "run_id": RUN_ID,
        "status": "finished",
        "registered_attempt": 1,
        "primary_weights": "final_step_ema",
        "verdict": verdict["verdict"],
        "all_four_passed": verdict["all_four_passed"],
        "preregistration": path_for_cli(RUN_DIR / "preregistration.json"),
        "pretraining_manifest_amendment": "docs/PHASE_T1_LITE_PRETRAINING_MANIFEST_AMENDMENT_20260724.md",
        "data": data,
        "training_summary": path_for_cli(training_dir / "training_summary.json"),
        "evaluations": {
            label: path_for_cli(RUN_DIR / "eval" / label / "summary.json")
            for label in evaluations
        },
        "seed_1_trigger": seed_1,
        "seed_1_triggered": bool(seed_1["triggered"]),
        "seed_1_not_launched_by_this_target": True,
        "d0_authorized": False,
        "c_track_authorized": False,
    }
    write_json(RUN_DIR / "summary.json", summary)
    write_json(DRIVE_ROOT / "summary.json", summary)
    publish(f"Finish registered Paper Two T1-lite {RUN_ID} [skip ci]")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
