"""Freeze EVAL-D/E and publish hash-only feature/cache receipts."""

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

from training.paper2_dc1_followups import POST_D0_CHECKPOINT_SHA256  # noqa: E402
from training.speculative_depth_d0_corpus import sha256_file  # noqa: E402


RUN_ID = "stage5_paper2_phase2_prewindow_20260731"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_RUN = Path(f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}")
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
STAGE_A_DRIVE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_dc1_stage_a_20260730"
)
D0_LOCK = ROOT / "outputs/stage5/stage5_paper2_d0_preregistration_20260726"


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
        if source.is_file() and sha256_file(source) == POST_D0_CHECKPOINT_SHA256:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or sha256_file(destination) != POST_D0_CHECKPOINT_SHA256:
                shutil.copy2(source, destination)
            return destination
    raise FileNotFoundError(f"EVAL-D/E checkpoint missing: {candidates}")


def publish(path: Path) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", path.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Freeze Phase-2 EVAL-D EVAL-E [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    checkpoint = restore_checkpoint()
    prior_paths = [
        DC0_DRIVE / "private/eval_b/eval_b.jsonl",
        DC1_DRIVE / "private/dev_c/dev_c.jsonl",
        STAGE_A_DRIVE / "private/eval_c/eval_c.jsonl",
    ]
    for path in prior_paths:
        if not path.exists():
            raise FileNotFoundError(f"EVAL-D/E disjointness prerequisite missing: {path}")
    private_root = DRIVE_RUN / "private/eval_de"
    output = RUN_DIR / "eval_de/summary.json"
    command = [
        sys.executable,
        "-u",
        "-m",
        "eval.prepare_paper2_phase2_eval_de",
        "--data_manifest",
        str(D0_LOCK / "data_manifest.json"),
    ]
    for path in prior_paths:
        command.extend(["--prior_partition_jsonl", str(path)])
    command.extend(
        [
            "--checkpoint",
            str(checkpoint),
            "--expected_checkpoint_sha256",
            POST_D0_CHECKPOINT_SHA256,
            "--private_root",
            str(private_root),
            "--output_summary",
            str(output),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_PHASE2_EVAL_DE_DTYPE", "bfloat16"),
            "--attn_implementation",
            "sdpa",
        ]
    )
    run(command)
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload["cross_partition_document_overlap"]:
        raise RuntimeError("EVAL-D/E public receipt reports document overlap")
    for partition in ("eval_d", "eval_e"):
        receipt = payload["partitions"][partition]
        if receipt["scores_exposed"] or receipt["read_once_scoring_spent"]:
            raise RuntimeError(f"{partition} was not frozen score-blind")
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, receipt_dir / "eval_de_freeze_summary.json")
    commit = publish(output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "summary_sha256": sha256_file(output),
                "publish_commit": commit,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
