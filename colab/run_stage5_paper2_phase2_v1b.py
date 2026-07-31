"""Run and publish the authorized DEV-only Phase-2 V1b receipt."""

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
DC1_DRIVE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_dc1_preflight_20260729"
)


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


def resolve_checkpoint() -> Path:
    candidates = [
        D0_DRIVE / "private/training/d0_ema_step_4000.pt",
        D0_DRIVE / "private/train/d0_ema_step_4000.pt",
        D0_DRIVE / "checkpoints/d0_ema_step_4000.pt",
    ]
    diagnostics = []
    for path in candidates:
        observed = sha256_file(path) if path.is_file() else None
        diagnostics.append(
            {"path": str(path), "exists": path.is_file(), "sha256": observed}
        )
        if observed == POST_D0_CHECKPOINT_SHA256:
            print("phase2_v1b_resolved_post_d0=" + json.dumps(diagnostics), flush=True)
            return path
    raise FileNotFoundError("missing post-D0 checkpoint: " + json.dumps(diagnostics))


def resolve_v1_summary() -> Path:
    candidates = [
        RUN_DIR / "v1_v2/summary.json",
        DRIVE_RUN / "receipts/v1_v2_summary.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        v1_v2 = json.loads(path.read_text(encoding="utf-8"))
        if v1_v2["status"] == "complete_no_training_dev_only":
            return path
    raise FileNotFoundError(
        "V1b requires completed V1/V2 first: "
        + json.dumps([str(path) for path in candidates])
    )


def publish(path: Path) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", path.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 V1b receipt [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    data = DC1_DRIVE / "private/dev_c/dev_c.jsonl"
    teacher = DC1_DRIVE / "private/dev_c/teacher_cache_summary.json"
    v1_private = DRIVE_RUN / "private/v1_v2"
    for path in (data, teacher, v1_private / "config.json"):
        if not path.exists():
            raise FileNotFoundError(f"V1b DEV prerequisite missing: {path}")
    checkpoint = resolve_checkpoint()
    v1_summary = resolve_v1_summary()
    output = RUN_DIR / "v1b/summary.json"
    private = DRIVE_RUN / "private/v1b"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase2_v1b",
            "--data_jsonl",
            str(data),
            "--teacher_cache_summary",
            str(teacher),
            "--checkpoint",
            str(checkpoint),
            "--checkpoint_sha256",
            POST_D0_CHECKPOINT_SHA256,
            "--v1_summary",
            str(v1_summary),
            "--v1_private_dir",
            str(v1_private),
            "--output_summary",
            str(output),
            "--private_dir",
            str(private),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--sample_size",
            os.environ.get("STAGE5_PHASE2_V1B_SAMPLE_SIZE", "2000"),
            "--sample_seed",
            "20260731",
            "--perturbation_batch",
            os.environ.get("STAGE5_PHASE2_V1B_BATCH", "8"),
            "--logit_position_chunk",
            os.environ.get("STAGE5_PHASE2_V1B_LOGIT_CHUNK", "16"),
            "--gamma",
            "0.05",
            "--rho",
            "0.8",
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload["training_started"] or payload["optimizer_steps"]:
        raise RuntimeError("V1b violated the no-update policy")
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, receipt_dir / "v1b_summary.json")
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
