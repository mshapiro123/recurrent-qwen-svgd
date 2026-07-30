"""Run and publish the read-only DC1 residual-stream scale probe."""

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

from training.paper2_dc1_followups import POST_D0_CHECKPOINT_SHA256
from training.speculative_depth_d0_corpus import sha256_file


RUN_ID = "stage5_paper2_dc1_scale_response_20260730"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_RUN = Path(f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}")
D0_DRIVE = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_d0_20260726")
DC1_DRIVE = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_dc1_preflight_20260729")
DC1_PUBLIC = ROOT / "outputs/stage5/stage5_paper2_dc1_preflight_20260729/dc1_p/summary.json"


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


def resolve_file(label: str, candidates: list[Path], expected_sha256: str | None = None) -> Path:
    diagnostics = []
    for source in candidates:
        observed = sha256_file(source) if source.is_file() else None
        diagnostics.append({"path": str(source), "exists": source.is_file(), "sha256": observed})
        if source.is_file() and (expected_sha256 is None or observed == expected_sha256):
            print(f"resolved_{label}=" + json.dumps(diagnostics), flush=True)
            return source
    raise FileNotFoundError(f"missing {label}: {json.dumps(diagnostics)}")


def publish(paths: list[Path]) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", *[path.relative_to(ROOT).as_posix() for path in paths]])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record DC1 scale-response probe [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    data = resolve_file("dev_c_data", [DC1_DRIVE / "private/dev_c/dev_c.jsonl"])
    teacher = resolve_file(
        "dev_c_teacher_cache_summary",
        [DC1_DRIVE / "private/dev_c/teacher_cache_summary.json"],
    )
    checkpoint = resolve_file(
        "post_d0_checkpoint",
        [
            D0_DRIVE / "private/training/d0_ema_step_4000.pt",
            D0_DRIVE / "private/train/d0_ema_step_4000.pt",
            D0_DRIVE / "checkpoints/d0_ema_step_4000.pt",
        ],
        expected_sha256=POST_D0_CHECKPOINT_SHA256,
    )
    DRIVE_RUN.mkdir(parents=True, exist_ok=True)
    output = RUN_DIR / "summary.json"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_dc1_scale_response",
            "--data_jsonl",
            str(data),
            "--teacher_cache_summary",
            str(teacher),
            "--checkpoint",
            str(checkpoint),
            "--expected_checkpoint_sha256",
            POST_D0_CHECKPOINT_SHA256,
            "--preflight_summary",
            str(DC1_PUBLIC),
            "--output_summary",
            str(output),
            "--private_dir",
            str(DRIVE_RUN / "private"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_DC1_SCALE_DTYPE", "bfloat16"),
            "--batch_size",
            os.environ.get("STAGE5_DC1_SCALE_BATCH_SIZE", "8"),
        ]
    )
    receipts = DRIVE_RUN / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, receipts / "summary.json")
    commit = publish([output])
    print(json.dumps({"status": "complete", "publish_commit": commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
