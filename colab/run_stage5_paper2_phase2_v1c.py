"""Run and publish the authorized DEV-only Phase-2 V1c radius extension."""

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
            print("phase2_v1c_resolved_post_d0=" + json.dumps(diagnostics), flush=True)
            return path
    raise FileNotFoundError("missing post-D0 checkpoint: " + json.dumps(diagnostics))


def resolve_receipt(relative: str, drive_name: str) -> Path:
    candidates = [RUN_DIR / relative, DRIVE_RUN / "receipts" / drive_name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"V1c prerequisite missing: {json.dumps([str(path) for path in candidates])}"
    )


def publish(path: Path) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", path.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 V1c receipt [skip ci]"])
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
            raise FileNotFoundError(f"V1c DEV prerequisite missing: {path}")
    checkpoint = resolve_checkpoint()
    v1_summary = resolve_receipt("v1_v2/summary.json", "v1_v2_summary.json")
    v1b_summary = resolve_receipt("v1b/summary.json", "v1b_summary.json")
    v1b = json.loads(v1b_summary.read_text(encoding="utf-8"))
    if (
        v1b.get("kind") != "paper2_phase2_v1b_finite_perturbation"
        or v1b.get("status") != "complete_no_training_dev_only"
    ):
        raise RuntimeError("V1c requires the completed canonical V1b receipt")

    output = RUN_DIR / "v1c/summary.json"
    private = DRIVE_RUN / "private/v1c_radius_v1"
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
            "--prior_v1b_summary",
            str(v1b_summary),
            "--output_summary",
            str(output),
            "--private_dir",
            str(private),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--sample_size",
            "2000",
            "--sample_seed",
            "20260731",
            "--perturbation_batch",
            os.environ.get("STAGE5_PHASE2_V1C_BATCH", "8"),
            "--logit_position_chunk",
            os.environ.get("STAGE5_PHASE2_V1C_LOGIT_CHUNK", "16"),
            "--gamma",
            "0.05",
            "--rho",
            "0.8",
            "--c_values",
            "0.075,0.10,0.15",
            "--receipt_kind",
            "paper2_phase2_v1c_radius_extension",
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload["training_started"] or payload["optimizer_steps"]:
        raise RuntimeError("V1c violated the no-update policy")
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, receipt_dir / "v1c_summary.json")
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
