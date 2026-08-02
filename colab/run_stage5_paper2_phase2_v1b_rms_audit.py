"""Run and publish the CPU-only Phase-2 V1b RMS-tail audit."""

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


RUN_ID = "stage5_paper2_phase2_prewindow_20260731"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_RUN = Path(f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}")
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


def resolve_v1b_summary() -> Path:
    candidates = [
        RUN_DIR / "v1b/summary.json",
        DRIVE_RUN / "receipts/v1b_summary.json",
    ]
    for path in candidates:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("kind") == "paper2_phase2_v1b_finite_perturbation"
                and payload.get("status") == "complete_no_training_dev_only"
            ):
                return path
    raise FileNotFoundError(
        "RMS audit requires canonical V1b: "
        + json.dumps([str(path) for path in candidates])
    )


def publish(path: Path) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", path.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 V1b RMS audit [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    private = DRIVE_RUN / "private/v1b_neutral_v2"
    v1_config = DRIVE_RUN / "private/v1_v2/config.json"
    data = DC1_DRIVE / "private/dev_c/dev_c.jsonl"
    for path in (private / "config.json", v1_config, data):
        if not path.exists():
            raise FileNotFoundError(f"RMS audit prerequisite missing: {path}")
    v1b = resolve_v1b_summary()
    output = RUN_DIR / "v1b_rms_audit/summary.json"
    private_detail = DRIVE_RUN / "private/v1b_rms_audit/outliers.json"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "analysis.audit_paper2_phase2_v1b_rms",
            "--private_dir",
            str(private),
            "--v1b_summary",
            str(v1b),
            "--data_jsonl",
            str(data),
            "--v1_config",
            str(v1_config),
            "--output_summary",
            str(output),
            "--private_detail",
            str(private_detail),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload["training_started"] or payload["model_inference_started"]:
        raise RuntimeError("RMS audit violated its read-only contract")
    receipts = DRIVE_RUN / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, receipts / "v1b_rms_audit_summary.json")
    commit = publish(output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cap_recommendation": payload["cap_recommendation"],
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
