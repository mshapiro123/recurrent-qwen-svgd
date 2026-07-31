"""Freeze EVAL-C and publish only its pre-lock hash receipt."""

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
DC0_DRIVE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_dc0_20260728"
)
DC1_DRIVE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_dc1_preflight_20260729"
)
D0_LOCK = ROOT / "outputs/stage5/stage5_paper2_d0_preregistration_20260726"
CHECKPOINT_SHA = "8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf"
DEV_C_SHA = "05bca2ee3ba71421296b2e31a0439746eb9c1b0e15e2cea4471be202ab6ac29d"
DEV_C_MANIFEST_SHA = "1816d9e953280cfb335c23de80292b64e36270599c3b4d273474b25f2e476caf"


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


def restore_checkpoint() -> Path:
    candidates = [
        D0_DRIVE / "private/training/d0_ema_step_4000.pt",
        D0_DRIVE / "private/train/d0_ema_step_4000.pt",
        D0_DRIVE / "checkpoints/d0_ema_step_4000.pt",
    ]
    destination = RUN_DIR / "runtime/d0_ema_step_4000.pt"
    for source in candidates:
        if source.exists() and sha256_file(source) == CHECKPOINT_SHA:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or sha256_file(destination) != CHECKPOINT_SHA:
                shutil.copy2(source, destination)
            return destination
    raise FileNotFoundError(f"EVAL-C checkpoint not found with locked SHA: {candidates}")


def publish(path: Path) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", path.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", f"Freeze DC1 EVAL-C {RUN_ID} [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    checkpoint = restore_checkpoint()
    prior_eval_b = DC0_DRIVE / "private/eval_b/eval_b.jsonl"
    prior_dev_c = DC1_DRIVE / "private/dev_c/dev_c.jsonl"
    prior_dev_manifest = DC1_DRIVE / "private/dev_c/document_manifest.json"
    for path in (prior_eval_b, prior_dev_c, prior_dev_manifest):
        if not path.exists():
            raise FileNotFoundError(f"EVAL-C disjointness prerequisite missing: {path}")
    if sha256_file(prior_dev_c) != DEV_C_SHA:
        raise RuntimeError("DEV-C JSONL differs from the locked Stage A training receipt")
    if sha256_file(prior_dev_manifest) != DEV_C_MANIFEST_SHA:
        raise RuntimeError("DEV-C manifest differs from the locked Stage A training receipt")

    private = DRIVE_RUN / "private/eval_c"
    data = private / "eval_c.jsonl"
    cache_root = private / "teacher_cache"
    cache_summary = private / "teacher_cache_summary.json"
    public_summary = RUN_DIR / "eval_c/summary.json"
    private.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.prepare_paper2_dc1_eval_c",
            "--data_manifest",
            str(D0_LOCK / "data_manifest.json"),
            "--prior_partition_jsonl",
            str(prior_eval_b),
            "--prior_partition_jsonl",
            str(prior_dev_c),
            "--checkpoint",
            str(checkpoint),
            "--expected_checkpoint_sha256",
            CHECKPOINT_SHA,
            "--output_data",
            str(data),
            "--private_cache_root",
            str(cache_root),
            "--private_cache_summary",
            str(cache_summary),
            "--output_summary",
            str(public_summary),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_DC1_EVAL_C_DTYPE", "bfloat16"),
            "--attn_implementation",
            "sdpa",
        ]
    )

    receipt = json.loads(public_summary.read_text(encoding="utf-8"))
    if receipt["manifest_sha256"] != sha256_file(private / "document_manifest.json"):
        raise RuntimeError("EVAL-C manifest hash is inconsistent with public receipt")
    if receipt["teacher_cache_sha256"] != sha256_file(cache_summary):
        raise RuntimeError("EVAL-C teacher-cache hash is inconsistent with public receipt")
    if receipt["read_once_scoring_spent"] or receipt["scores_exposed"]:
        raise RuntimeError("EVAL-C freeze receipt incorrectly marks the scoring pass spent")

    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(public_summary, receipt_dir / "eval_c_freeze_summary.json")
    commit = publish(public_summary)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "manifest_sha256": receipt["manifest_sha256"],
                "teacher_cache_sha256": receipt["teacher_cache_sha256"],
                "read_once_scoring_spent": receipt["read_once_scoring_spent"],
                "publish_commit": commit,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
