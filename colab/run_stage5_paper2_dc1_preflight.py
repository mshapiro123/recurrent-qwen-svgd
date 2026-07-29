"""Prepare DEV-C, run DC1-P, and rerun RG-4/RG-11 without training."""

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


RUN_ID = os.environ.get("STAGE5_PAPER2_DC1_PREFLIGHT_RUN_ID", "stage5_paper2_dc1_preflight_20260729")
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_RUN = Path(f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}")
D0_DRIVE = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_d0_20260726")
DC0_DRIVE = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_dc0_20260728")
D0_LOCK = ROOT / "outputs/stage5/stage5_paper2_d0_preregistration_20260726"
DC0_PUBLIC = ROOT / "outputs/stage5/stage5_paper2_dc0_20260728/dc0/summary.json"
CHECKPOINT_SHA = "8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf"


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
    return int(code)


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
    raise FileNotFoundError(f"DC1-P checkpoint not found with locked SHA: {candidates}")


def publish(paths: list[Path]) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", *[path.relative_to(ROOT).as_posix() for path in paths]])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", f"Record DC1 preflight {RUN_ID} [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    checkpoint = restore_checkpoint()
    prior_eval_b = DC0_DRIVE / "private/eval_b/eval_b.jsonl"
    if not prior_eval_b.exists():
        raise FileNotFoundError(
            "DEV-C disjointness requires the private EVAL-B document ids; "
            f"missing {prior_eval_b}"
        )
    private = DRIVE_RUN / "private"
    data = private / "dev_c/dev_c.jsonl"
    cache_root = private / "dev_c/teacher_cache"
    cache_summary = private / "dev_c/teacher_cache_summary.json"
    dev_public = RUN_DIR / "dev_c/summary.json"
    DRIVE_RUN.mkdir(parents=True, exist_ok=True)
    if not dev_public.exists() or not cache_summary.exists():
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.prepare_paper2_dc1_dev_c",
                "--data_manifest",
                str(D0_LOCK / "data_manifest.json"),
                "--prior_partition_jsonl",
                str(prior_eval_b),
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
                str(dev_public),
                "--device",
                os.environ.get("DEVICE", "cuda"),
                "--dtype",
                os.environ.get("STAGE5_DC1_DTYPE", "bfloat16"),
                "--attn_implementation",
                "sdpa",
            ]
        )
    if sha256_file(data) != json.loads(dev_public.read_text(encoding="utf-8"))["data"]["sha256"]:
        raise RuntimeError("resumed DEV-C data hash differs from its public receipt")

    dc1_output = RUN_DIR / "dc1_p"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_dc1_preflight",
            "--data_jsonl",
            str(data),
            "--teacher_cache_summary",
            str(cache_summary),
            "--checkpoint",
            str(checkpoint),
            "--expected_checkpoint_sha256",
            CHECKPOINT_SHA,
            "--dc0_summary",
            str(DC0_PUBLIC),
            "--output_dir",
            str(dc1_output),
            "--private_dir",
            str(private / "dc1_p"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_DC1_DTYPE", "bfloat16"),
            "--attn_implementation",
            "sdpa",
            "--append_batch_size",
            os.environ.get("STAGE5_DC1_APPEND_BATCH_SIZE", "8"),
        ]
    )
    numerics_output = RUN_DIR / "rg4_rg11"
    numerics_code = run(
        [
            sys.executable,
            "-u",
            "eval/eval_coconut_composite_numerics.py",
            "--output_dir",
            str(numerics_output),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ],
        allowed=(0, 2),
    )
    dc1 = json.loads((dc1_output / "summary.json").read_text(encoding="utf-8"))
    numerics = json.loads((numerics_output / "summary.json").read_text(encoding="utf-8"))
    combined = {
        "kind": "paper2_dc1_preflight_packet",
        "status": (
            "complete_ready_for_preregistration_draft"
            if numerics_code == 0
            else "complete_numerics_need_review_before_preregistration"
        ),
        "dc1_p": dc1,
        "rg4_rg11": numerics,
        "training_started": False,
        "optimizer_steps": 0,
        "evaluation_c_touched": False,
        "training_authorized_by_this_packet": False,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = RUN_DIR / "summary.json"
    combined_path.write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        dev_public,
        dc1_output / "summary.json",
        numerics_output / "summary.json",
        combined_path,
    ]
    for path in paths:
        shutil.copy2(path, receipt_dir / f"{path.parent.name}_{path.name}")
    commit = publish(paths)
    print(json.dumps({"status": combined["status"], "publish_commit": commit}, indent=2))
    return numerics_code


if __name__ == "__main__":
    raise SystemExit(main())

