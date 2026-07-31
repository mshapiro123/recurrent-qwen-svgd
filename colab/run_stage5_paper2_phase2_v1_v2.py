"""Run and publish the development-only Phase-2 V1/V2 diagnostics."""

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

from training.paper2_dc1_followups import (  # noqa: E402
    POST_D0_CHECKPOINT_SHA256,
    PRE_D0_CHECKPOINT_SHA256,
)
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
STAGE_A_DRIVE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_dc1_stage_a_20260730"
)
TRAINED_BRIDGE_SHA256 = (
    "5f2f2d89d26642e16c0e4640ea01fa79a408c25bdf71794e6235948ed96ce0cb"
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


def resolve(label: str, candidates: list[Path], expected_sha256: str) -> Path:
    diagnostics = []
    for path in candidates:
        observed = sha256_file(path) if path.is_file() else None
        diagnostics.append({"path": str(path), "exists": path.is_file(), "sha256": observed})
        if observed == expected_sha256:
            print(f"phase2_resolved_{label}=" + json.dumps(diagnostics), flush=True)
            return path
    raise FileNotFoundError(f"missing {label}: {json.dumps(diagnostics)}")


def publish(path: Path) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", path.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 V1 V2 diagnostics [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    data = DC1_DRIVE / "private/dev_c/dev_c.jsonl"
    teacher = DC1_DRIVE / "private/dev_c/teacher_cache_summary.json"
    for path in (data, teacher):
        if not path.exists():
            raise FileNotFoundError(f"V1/V2 DEV-C prerequisite missing: {path}")
    post = resolve(
        "post_d0",
        [
            D0_DRIVE / "private/training/d0_ema_step_4000.pt",
            D0_DRIVE / "private/train/d0_ema_step_4000.pt",
            D0_DRIVE / "checkpoints/d0_ema_step_4000.pt",
        ],
        POST_D0_CHECKPOINT_SHA256,
    )
    pre = resolve(
        "pre_d0",
        [
            Path(
                "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/"
                "stage5_paper2_t1_lite_r_20260725/checkpoints/"
                "t1_lite_r_raw_step_10500.pt"
            ),
            D0_DRIVE / "checkpoints/t1_lite_r_raw_step_10500.pt",
            D0_DRIVE / "private/restored/t1_lite_r_raw_step_10500.pt",
        ],
        PRE_D0_CHECKPOINT_SHA256,
    )
    bridge = resolve(
        "stage_a_bridge",
        [STAGE_A_DRIVE / "private/training/stage_a_step_2000.pt"],
        TRAINED_BRIDGE_SHA256,
    )
    output = RUN_DIR / "v1_v2/summary.json"
    private = DRIVE_RUN / "private/v1_v2"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase2_v1_v2",
            "--data_jsonl",
            str(data),
            "--teacher_cache_summary",
            str(teacher),
            "--post_checkpoint",
            str(post),
            "--post_checkpoint_sha256",
            POST_D0_CHECKPOINT_SHA256,
            "--pre_checkpoint",
            str(pre),
            "--pre_checkpoint_sha256",
            PRE_D0_CHECKPOINT_SHA256,
            "--trained_bridge_checkpoint",
            str(bridge),
            "--trained_bridge_sha256",
            TRAINED_BRIDGE_SHA256,
            "--output_summary",
            str(output),
            "--private_dir",
            str(private),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--max_rows",
            os.environ.get("STAGE5_PHASE2_V1_ROWS", "128"),
            "--max_v1_positions",
            os.environ.get("STAGE5_PHASE2_V1_POSITIONS", "128"),
            "--max_v2_rows",
            os.environ.get("STAGE5_PHASE2_V2_ROWS", "32"),
            "--random_probes",
            os.environ.get("STAGE5_PHASE2_JVP_PROBES", "2"),
            "--append_batch_size",
            os.environ.get("STAGE5_PHASE2_APPEND_BATCH", "8"),
            "--gamma",
            "0.05",
            "--rho",
            "0.8",
        ]
    )
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, receipt_dir / "v1_v2_summary.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload["training_started"] or payload["optimizer_steps"]:
        raise RuntimeError("V1/V2 violated the no-training policy")
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
