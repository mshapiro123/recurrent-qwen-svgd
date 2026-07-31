"""Run locked DC1 Stage A training followed by the sole EVAL-C pass."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.speculative_depth_d0_corpus import sha256_file  # noqa: E402


RUN_ID = "stage5_paper2_dc1_stage_a_20260730"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_RUN = Path(
    f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}"
)
D0_DRIVE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_d0_20260726"
)
DC1_DRIVE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_dc1_preflight_20260729"
)
PREREG = ROOT / "docs/stage_a_prereg.json"
LOCK_COMMIT = "d25b3d0e7811c0b12a525636d5f26f69dc05c3bc"


def run(command: list[str]) -> None:
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
    if code:
        raise subprocess.CalledProcessError(code, command)


def restore_checkpoint(expected_sha: str) -> Path:
    candidates = [
        D0_DRIVE / "private/training/d0_ema_step_4000.pt",
        D0_DRIVE / "private/train/d0_ema_step_4000.pt",
        D0_DRIVE / "checkpoints/d0_ema_step_4000.pt",
    ]
    destination = RUN_DIR / "runtime/d0_ema_step_4000.pt"
    for source in candidates:
        if source.exists() and sha256_file(source) == expected_sha:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or sha256_file(destination) != expected_sha:
                shutil.copy2(source, destination)
            return destination
    raise FileNotFoundError(f"Stage A checkpoint not found with locked SHA: {candidates}")


def publish(paths: list[Path], message: str) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", *[path.relative_to(ROOT).as_posix() for path in paths]])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", f"{message} [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def restore_public_receipt(drive_name: str, destination: Path) -> bool:
    source = DRIVE_RUN / "receipts" / drive_name
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def main() -> int:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg.get("locked_before_training") is not True:
        raise RuntimeError("Stage A preregistration is not locked")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", LOCK_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    init_checkpoint = restore_checkpoint(prereg["init_checkpoint_sha256"])
    dev_c = DC1_DRIVE / "private/dev_c/dev_c.jsonl"
    dev_cache = DC1_DRIVE / "private/dev_c/teacher_cache_summary.json"
    dev_manifest = DC1_DRIVE / "private/dev_c/document_manifest.json"
    eval_private = DRIVE_RUN / "private/eval_c"
    eval_c = eval_private / "eval_c.jsonl"
    eval_manifest = eval_private / "document_manifest.json"
    eval_cache = eval_private / "teacher_cache_summary.json"
    eval_freeze = RUN_DIR / "eval_c/summary.json"
    for path in (
        dev_c,
        dev_cache,
        dev_manifest,
        eval_c,
        eval_manifest,
        eval_cache,
        eval_freeze,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Stage A prerequisite is missing: {path}")
    if sha256_file(dev_manifest) != prereg["train_partition"]["manifest_sha256"]:
        raise RuntimeError("Stage A DEV-C manifest SHA-256 mismatch")

    receipts = DRIVE_RUN / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    training_summary = RUN_DIR / "training/summary.json"
    if not training_summary.exists():
        restore_public_receipt("training_summary.json", training_summary)
    private_train = DRIVE_RUN / "private/training"
    primary_checkpoint = private_train / "stage_a_step_2000.pt"
    if not training_summary.exists() or not primary_checkpoint.exists():
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "training.train_paper2_dc1_stage_a",
                "--prereg",
                str(PREREG),
                "--data_jsonl",
                str(dev_c),
                "--teacher_cache_summary",
                str(dev_cache),
                "--checkpoint",
                str(init_checkpoint),
                "--private_train_dir",
                str(private_train),
                "--output_summary",
                str(training_summary),
                "--device",
                os.environ.get("DEVICE", "cuda"),
                "--attn_implementation",
                "sdpa",
            ]
        )
        shutil.copy2(training_summary, receipts / "training_summary.json")
    training = json.loads(training_summary.read_text(encoding="utf-8"))
    if training["status"] != "complete_ready_for_single_eval_c_pass":
        raise RuntimeError("Stage A training did not reach its registered endpoint")
    if sha256_file(primary_checkpoint) != training["primary_checkpoint"]["sha256"]:
        raise RuntimeError("Stage A primary bridge checkpoint SHA-256 mismatch")
    training_commit = publish(
        [training_summary], f"Record DC1 Stage A training {RUN_ID}"
    )
    print(f"stage_a_training_publish_commit={training_commit}", flush=True)

    eval_summary = RUN_DIR / "eval_c_result/summary.json"
    eval_verdict = RUN_DIR / "eval_c_result/verdict.json"
    if not eval_summary.exists():
        restored = restore_public_receipt("eval_c_result_summary.json", eval_summary)
        if restored:
            restore_public_receipt("eval_c_result_verdict.json", eval_verdict)
    if not eval_summary.exists():
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_dc1_stage_a",
                "--prereg",
                str(PREREG),
                "--eval_freeze_summary",
                str(eval_freeze),
                "--data_jsonl",
                str(eval_c),
                "--document_manifest",
                str(eval_manifest),
                "--teacher_cache_summary",
                str(eval_cache),
                "--init_checkpoint",
                str(init_checkpoint),
                "--trained_checkpoint",
                str(primary_checkpoint),
                "--output_dir",
                str(eval_summary.parent),
                "--private_dir",
                str(DRIVE_RUN / "private/eval_c_result"),
                "--device",
                os.environ.get("DEVICE", "cuda"),
                "--append_batch_size",
                os.environ.get("STAGE5_DC1_STAGE_A_EVAL_BATCH_SIZE", "24"),
            ]
        )
        shutil.copy2(eval_summary, receipts / "eval_c_result_summary.json")
        shutil.copy2(eval_verdict, receipts / "eval_c_result_verdict.json")

    evaluation = json.loads(eval_summary.read_text(encoding="utf-8"))
    combined = {
        "kind": "paper2_dc1_stage_a",
        "status": "complete",
        "training": training,
        "evaluation": evaluation,
        "registered_verdict": evaluation["verdict"]["verdict"],
        "registered_consequence": evaluation["verdict"]["consequence"],
        "read_once_scoring_spent": True,
    }
    combined_path = RUN_DIR / "summary.json"
    combined_path.write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(combined_path, receipts / "combined_summary.json")
    final_commit = publish(
        [eval_summary, eval_verdict, combined_path],
        f"Record DC1 Stage A verdict {RUN_ID}",
    )
    print(
        json.dumps(
            {
                "status": combined["status"],
                "registered_verdict": combined["registered_verdict"],
                "registered_consequence": combined["registered_consequence"],
                "publish_commit": final_commit,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
