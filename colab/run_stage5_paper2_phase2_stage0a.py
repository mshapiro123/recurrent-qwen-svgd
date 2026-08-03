"""Run and publish the resumable development-only Phase-2 Stage 0A cache."""

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

from training.paper2_phase2_stage0a import STAGE0A_CONFIG, sha256_file  # noqa: E402


# Safety marker: DEV-C only sparse lattice and teacher states no optimizer no training
RUN_ID = STAGE0A_CONFIG["run_id"]
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_RUN = Path(
    os.environ.get(
        "STAGE5_PHASE2_STAGE0A_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}",
    )
)
DEV_C = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_dc1_preflight_20260729/private/dev_c/dev_c.jsonl"
)
V1D_RECEIPT = (
    ROOT
    / "outputs/stage5/stage5_paper2_phase2_prewindow_20260731/v1d/summary.json"
)
CONSTANTS = ROOT / "training/paper2_phase2_dc2_constants.json"
EXPECTED_V1D_SHA256 = (
    "39e08a3a0f0a7abb23ae5fd45874a33cd0fba755099b9cc3c70da19ed332c094"
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
    tail: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
        tail = tail[-300:]
    code = process.wait()
    if code:
        print("\nStage 0A child-process tail:\n" + "\n".join(tail), flush=True)
        raise subprocess.CalledProcessError(code, command)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def status_event(status: str, **details: Any) -> None:
    payload = {
        "kind": "paper2_phase2_stage0a_status",
        "status": status,
        "run_id": RUN_ID,
        **details,
    }
    print("stage0a_status:", json.dumps(payload, sort_keys=True), flush=True)
    try:
        write_json(DRIVE_RUN / "receipts/stage0a_status.json", payload)
    except Exception as error:
        print(f"stage0a_status_write_failed={error!r}", flush=True)


def validate_inputs() -> None:
    if not DEV_C.is_file():
        raise FileNotFoundError(f"Stage 0A DEV-C input is missing: {DEV_C}")
    if sha256_file(DEV_C) != STAGE0A_CONFIG["data_sha256"]:
        raise RuntimeError("Stage 0A DEV-C differs from the locked development partition")
    if not V1D_RECEIPT.is_file() or sha256_file(V1D_RECEIPT) != EXPECTED_V1D_SHA256:
        raise RuntimeError("Stage 0A requires the exact landed V1d receipt")
    v1d = json.loads(V1D_RECEIPT.read_text(encoding="utf-8"))
    constants = json.loads(CONSTANTS.read_text(encoding="utf-8"))
    if not v1d["v1d_preservation_reading"]["pass"]:
        raise RuntimeError("Stage 0A cannot proceed after a failed V1d preservation reading")
    if constants.get("status") != "confirmed_by_v1d":
        raise RuntimeError("Stage 0A DC2 constants are not banked as V1d-confirmed")
    if constants.get("source_receipt_sha256") != EXPECTED_V1D_SHA256:
        raise RuntimeError("Stage 0A DC2 constants point at a different V1d receipt")


def publish(path: Path) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", path.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 Stage 0A receipt [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    validate_inputs()
    DRIVE_RUN.mkdir(parents=True, exist_ok=True)
    output = RUN_DIR / "summary.json"
    private = DRIVE_RUN / "private/stage0a"
    status_event(
        "started_or_resumed",
        data_sha256=STAGE0A_CONFIG["data_sha256"],
        private_dir=str(private),
        training_started=False,
        optimizer_steps=0,
    )
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.cache_paper2_phase2_stage0a",
            "--data_jsonl",
            str(DEV_C),
            "--private_dir",
            str(private),
            "--output_summary",
            str(output),
            "--staging_dir",
            os.environ.get("STAGE5_PHASE2_STAGE0A_STAGING", "/content/stage0a_staging"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_PHASE2_STAGE0A_DTYPE", "bfloat16"),
            "--attn_implementation",
            os.environ.get("STAGE5_PHASE2_STAGE0A_ATTN", "sdpa"),
        ]
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    if summary.get("status") != "complete_development_only":
        raise RuntimeError("Stage 0A did not produce its completion receipt")
    if summary.get("training_started") or summary.get("optimizer_steps"):
        raise RuntimeError("Stage 0A violated its no-training contract")
    if summary.get("frozen_evaluation_partitions_touched"):
        raise RuntimeError("Stage 0A touched a frozen evaluation partition")
    if summary["teacher_states"]["samples"] < 200_000:
        raise RuntimeError("Stage 0A collected fewer than 200,000 teacher boundary states")
    expected_audit = round(
        STAGE0A_CONFIG["boundary_sample_count"]
        * STAGE0A_CONFIG["full_logit_audit_fraction"]
    )
    for model_key in STAGE0A_CONFIG["models"]:
        if summary["full_logit_audit"][model_key]["samples"] != expected_audit:
            raise RuntimeError(
                f"Stage 0A {model_key} full-logit audit is incomplete: "
                f"{summary['full_logit_audit'][model_key]['samples']} != {expected_audit}"
            )
        score_summary = summary["union_scores"][model_key]
        if score_summary["topk_equivalence_max_abs_error"] > score_summary[
            "topk_equivalence_tolerance"
        ]:
            raise RuntimeError(f"Stage 0A {model_key} union scorer failed equivalence")
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, receipt_dir / "stage0a_summary.json")
    commit = publish(output)
    status_event(
        "complete",
        summary_sha256=sha256_file(output),
        publish_commit=commit,
        training_started=False,
        optimizer_steps=0,
        frozen_evaluation_partitions_touched=[],
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary_sha256": sha256_file(output),
                "publish_commit": commit,
                "private_drive_root": str(private),
                "teacher_state_samples": summary["teacher_states"]["samples"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
