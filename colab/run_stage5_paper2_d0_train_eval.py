"""Run, resume, evaluate, and publish the registered Paper Two D0 pilot."""

from __future__ import annotations

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

from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_postlock import D0_LOCK_COMMIT, D0_RUN_ID, validate_cache_summary
from training.speculative_depth_d0_spec import DRAFTER_CHECKPOINT_SHA256, validate_locked_d0
from eval.eval_speculative_depth_router_feasibility import summarize_teacher_demand


LOCK_RUN = ROOT / "outputs/stage5/stage5_paper2_d0_preregistration_20260726"
RUN_DIR = ROOT / "outputs" / "stage5" / D0_RUN_ID
DRIVE_ROOT = Path(
    os.environ.get(
        "STAGE5_PAPER2_D0_RUN_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{D0_RUN_ID}",
    )
)
CHECKPOINT_SOURCE = Path(
    os.environ.get(
        "STAGE5_PAPER2_D0_DRAFTER_CHECKPOINT",
        "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/"
        "stage5_paper2_t1_lite_r_20260725/checkpoints/t1_lite_r_raw_step_10500.pt",
    )
)
T1_RUN = ROOT / "outputs/stage5/stage5_paper2_t1_lite_r_20260725"
PRELAUNCH_DIR = RUN_DIR / "prelaunch"
PRELAUNCH_SUMMARY = PRELAUNCH_DIR / "summary.json"
TARGET_POLICY_RECEIPT = PRELAUNCH_DIR / "target_policy_receipt.json"


def run(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> int:
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
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code not in allowed:
        raise subprocess.CalledProcessError(code, command)
    return code


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def restore_inputs() -> dict[str, Path]:
    manifest = read_json(LOCK_RUN / "data_manifest.json")
    restored: dict[str, Path] = {}
    for name in ("label_train", "calibration", "evaluation"):
        receipt = manifest["artifacts"][name]
        source = Path(receipt["drive_path"])
        destination = RUN_DIR / "private_inputs" / f"{name}.jsonl"
        if not source.exists() or sha256_file(source) != receipt["sha256"]:
            raise RuntimeError(f"D0 locked partition is missing or corrupt: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha256_file(destination) != receipt["sha256"]:
            shutil.copy2(source, destination)
        restored[name] = destination
    if not CHECKPOINT_SOURCE.exists() or sha256_file(CHECKPOINT_SOURCE) != DRAFTER_CHECKPOINT_SHA256:
        raise RuntimeError("D0 locked drafter checkpoint is missing or corrupt")
    checkpoint = RUN_DIR / "restored" / CHECKPOINT_SOURCE.name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists() or sha256_file(checkpoint) != DRAFTER_CHECKPOINT_SHA256:
        shutil.copy2(CHECKPOINT_SOURCE, checkpoint)
    restored["checkpoint"] = checkpoint
    return restored


def teacher_shift(private_rows_path: Path) -> dict[str, Any]:
    payload = read_json(private_rows_path)
    rows = payload.get("all_position_rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(
            "D0 teacher-shift evaluation requires all calibration positions; "
            "the legacy dual-teacher union is not an admissible substitute"
        )
    result = summarize_teacher_demand(rows)
    seven = result["teacher_7b_own_rejections"]
    fourteen = result["teacher_14b_own_rejections"]
    seven_median = seven["median_first_correct_depth_recoverable"]
    fourteen_median = fourteen["median_first_correct_depth_recoverable"]
    if seven_median is None or fourteen_median is None:
        reading = "inconclusive_one_or_both_teacher_populations_not_recoverable_by_depth_6"
    elif fourteen_median > seven_median:
        reading = "median_first_correct_depth_shifts_upward_with_teacher_depth"
    elif fourteen_median == seven_median:
        reading = "same_median_first_correct_depth_against_both_teachers"
    else:
        reading = "deeper_teacher_has_earlier_median_first_correct_depth_unexpected"
    result.update(
        {
            "kind": "paper2_d0_teacher_shift_signature",
            "status": "complete",
            "registered_reading": reading,
            "primary_statistic": "median first-correct depth among recoverable positions",
            "population_rule": (
                "7B demand uses 7B loop-1 rejections; 14B demand uses 14B loop-1 "
                "rejections; 14B on 7B rejections is descriptive overlap only"
            ),
            "amendment_timing": (
                "specified after the pretraining floor landed and before D0 training; "
                "the trained-model result remained untouched"
            ),
            "binary_target_caveat": (
                "depth-2 rejected targets may compress trained demand toward depth 2 "
                "for both teachers"
            ),
        }
    )
    return result


def publish(paths: list[Path]) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    relative = [path.relative_to(ROOT).as_posix() for path in paths]
    run(["git", "add", "-f", "--", *relative])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        raise RuntimeError("D0 train/eval produced no aggregate receipt changes")
    run(["git", "commit", "-m", "Record Paper Two D0 pilot [skip ci]"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run(["git", "push", "origin", "main"])
    return commit


def main() -> int:
    if subprocess.run(["git", "merge-base", "--is-ancestor", D0_LOCK_COMMIT, "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("D0 lock commit is not an ancestor of the training checkout")
    prereg = read_json(LOCK_RUN / "preregistration.json")
    validate_locked_d0(prereg)
    if prereg.get("training_authorized") is not True:
        raise RuntimeError("D0 training was not authorized by the landed lock")
    cache_summary_path = RUN_DIR / "labeling" / "summary.json"
    floor_summary_path = RUN_DIR / "floor" / "summary.json"
    cache_summary = read_json(cache_summary_path)
    validate_cache_summary(cache_summary)
    floor = read_json(floor_summary_path)
    if floor.get("status") != "complete":
        raise RuntimeError("D0 training requires the completed floor receipt")
    prelaunch = read_json(PRELAUNCH_SUMMARY)
    target_policy = read_json(TARGET_POLICY_RECEIPT)
    if prelaunch.get("status") != "complete" or prelaunch.get("evaluation_partition_touched") is not False:
        raise RuntimeError("D0 training requires the completed read-only prelaunch receipt")
    if target_policy.get("status") != "verified_before_training":
        raise RuntimeError("D0 training requires the verified binned target-policy receipt")
    if target_policy.get("descriptive_mapping_used_for_targets") is not False:
        raise RuntimeError("D0 target-policy receipt illegally uses the descriptive mapping")
    if prelaunch.get("target_policy_receipt_sha256") != sha256_file(TARGET_POLICY_RECEIPT):
        raise RuntimeError("D0 target-policy receipt hash disagrees with the prelaunch receipt")
    restored = restore_inputs()
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    training_dir = RUN_DIR / "train"
    training_summary = training_dir / "summary.json"
    training_code = run(
        [
            sys.executable,
            "training/run_speculative_depth_d0.py",
            "--preregistration",
            str(LOCK_RUN / "preregistration.json"),
            "--label_train_jsonl",
            str(restored["label_train"]),
            "--calibration_jsonl",
            str(restored["calibration"]),
            "--teacher_cache_summary",
            str(cache_summary_path),
            "--floor_summary",
            str(floor_summary_path),
            "--floor_private_rows",
            str(DRIVE_ROOT / "private/floor/floor_rows.json"),
            "--target_policy_receipt",
            str(TARGET_POLICY_RECEIPT),
            "--prelaunch_summary",
            str(PRELAUNCH_SUMMARY),
            "--checkpoint",
            str(restored["checkpoint"]),
            "--rehearsal_jsonl",
            str(T1_RUN / "data/t1_lite_train_70_30.jsonl"),
            "--rehearsal_pilot_jsonl",
            str(T1_RUN / "data/liveness_pilot_256.jsonl"),
            "--output_dir",
            str(training_dir),
            "--backup_dir",
            str(DRIVE_ROOT / "private/training"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_PAPER2_D0_DTYPE", "bfloat16"),
            "--attn_implementation",
            os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        ],
        allowed=(0, 2),
    )
    if training_code == 2:
        summary_md = RUN_DIR / "summary.md"
        summary_md.write_text("# Paper Two D0\n\nTraining stopped at the registered liveness guardrail.\n", encoding="utf-8")
        publish([training_summary, summary_md, *sorted((training_dir / "guardrails").glob("*.json"))])
        return 2

    training = read_json(training_summary)
    ema = Path(training["ema_checkpoint"])
    ema_sha = str(training["ema_checkpoint_sha256"])
    eval_dir = RUN_DIR / "eval"
    final_summary = eval_dir / "natural_summary.json"
    natural_code = run(
        [
            sys.executable,
            "eval/eval_speculative_depth_d0.py",
            "--preregistration",
            str(LOCK_RUN / "preregistration.json"),
            "--evaluation_jsonl",
            str(restored["evaluation"]),
            "--teacher_cache_summary",
            str(cache_summary_path),
            "--floor_summary",
            str(floor_summary_path),
            "--initial_checkpoint",
            str(restored["checkpoint"]),
            "--trained_checkpoint",
            str(ema),
            "--trained_checkpoint_sha256",
            ema_sha,
            "--output_summary",
            str(final_summary),
            "--private_rows_output",
            str(DRIVE_ROOT / "private/eval/evaluation_rows.json"),
            "--resume_dir",
            str(DRIVE_ROOT / "private/eval/row_cache"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ],
        allowed=(0, 2),
    )
    trained_floor = eval_dir / "trained_teacher_shift_summary.json"
    trained_floor_private = DRIVE_ROOT / "private/eval/trained_teacher_shift_rows.json"
    run(
        [
            sys.executable,
            "eval/eval_speculative_depth_d0_floor.py",
            "--preregistration",
            str(LOCK_RUN / "preregistration.json"),
            "--data_jsonl",
            str(restored["calibration"]),
            "--teacher_cache_summary",
            str(cache_summary_path),
            "--checkpoint",
            str(ema),
            "--expected_checkpoint_sha256",
            ema_sha,
            "--measurement_label",
            "trained_teacher_shift",
            "--output_summary",
            str(trained_floor),
            "--private_rows_output",
            str(trained_floor_private),
            "--resume_dir",
            str(DRIVE_ROOT / "private/eval/trained_teacher_shift_row_cache"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--batch_size",
            "1",
        ]
    )
    shift_summary = eval_dir / "teacher_shift_signature.json"
    write_json(shift_summary, teacher_shift(trained_floor_private))

    t1_private_dir = DRIVE_ROOT / "private/eval/t1_retention"
    t1_code = run(
        [
            sys.executable,
            "eval/eval_speculative_depth_d0_t1_retention.py",
            "--checkpoint",
            str(ema),
            "--checkpoint_sha256",
            ema_sha,
            "--frozen_eval_jsonl",
            str(ROOT / "outputs/stage5/stage5_synthetic_depth_frozen_eval_v2_depth14/data/test_chain_mcq.jsonl"),
            "--output_dir",
            str(t1_private_dir),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--batch_size",
            os.environ.get("STAGE5_PAPER2_D0_T1_BATCH_SIZE", "4"),
        ],
        allowed=(0, 2),
    )
    t1_summary = eval_dir / "t1_retention_summary.json"
    shutil.copy2(t1_private_dir / "summary.json", t1_summary)

    arc_summary = eval_dir / "arc_allocation_summary.json"
    run(
        [
            sys.executable,
            "eval/eval_speculative_depth_d0_arc_allocation.py",
            "--checkpoint",
            str(ema),
            "--checkpoint_sha256",
            ema_sha,
            "--output_summary",
            str(arc_summary),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--limit_per_benchmark",
            os.environ.get("STAGE5_PAPER2_D0_ARC_LIMIT", "128"),
        ]
    )

    blocked = natural_code == 2 or t1_code == 2
    overall = {
        "kind": "paper2_speculative_depth_d0",
        "status": "blocked_guardrail" if blocked else "complete",
        "lock_commit": D0_LOCK_COMMIT,
        "prelaunch_summary": PRELAUNCH_SUMMARY.relative_to(ROOT).as_posix(),
        "target_policy_receipt": TARGET_POLICY_RECEIPT.relative_to(ROOT).as_posix(),
        "training_summary": training_summary.relative_to(ROOT).as_posix(),
        "natural_summary": final_summary.relative_to(ROOT).as_posix(),
        "teacher_shift_summary": shift_summary.relative_to(ROOT).as_posix(),
        "t1_retention_summary": t1_summary.relative_to(ROOT).as_posix(),
        "arc_allocation_summary": arc_summary.relative_to(ROOT).as_posix(),
        "ema_checkpoint_sha256": ema_sha,
        "hard_guardrails": {
            "natural_surface": read_json(final_summary)["natural_surface_guardrail"],
            "t1_mechanism": read_json(t1_summary),
        },
        "interpretation_band": read_json(final_summary)["interpretation_band"],
        "training_authorized": True,
        "single_seed": True,
        "tie_policy": "fp32 logits; exact ties choose the lowest token id and are counted",
        "teacher_shift_population_rule": (
            "each teacher is measured on its own loop-1 rejection population"
        ),
    }
    overall_path = RUN_DIR / "summary.json"
    write_json(overall_path, overall)
    summary_md = RUN_DIR / "summary.md"
    summary_md.write_text(
        "# Paper Two D0 Pilot\n\n"
        f"- Status: `{overall['status']}`\n"
        f"- Interpretation band: `{overall['interpretation_band']}`\n"
        f"- EMA checkpoint SHA-256: `{ema_sha}`\n"
        "- Scope: teacher-forced next-token agreement, single seed\n",
        encoding="utf-8",
    )
    paths = [
        overall_path,
        summary_md,
        training_summary,
        final_summary,
        trained_floor,
        shift_summary,
        t1_summary,
        arc_summary,
        *sorted((training_dir / "guardrails").glob("*.json")),
    ]
    publish_commit = publish(paths)
    receipt = DRIVE_ROOT / "receipts" / "final_summary.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(overall_path, receipt)
    print(json.dumps({**overall, "publish_commit": publish_commit}, indent=2, sort_keys=True), flush=True)
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
