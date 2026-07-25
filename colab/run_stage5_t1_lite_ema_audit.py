"""Restore, run, and publish the read-only T1-lite EMA audit."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN_ID = "stage5_paper2_t1_lite_20260724"
RUN_ID = os.environ.get(
    "STAGE5_T1_EMA_AUDIT_RUN_ID",
    "stage5_paper2_t1_lite_ema_audit_20260725",
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
SOURCE_DIR = ROOT / "outputs" / "stage5" / SOURCE_RUN_ID
DRIVE_SOURCE = Path(
    os.environ.get(
        "STAGE5_T1_EMA_AUDIT_DRIVE_SOURCE",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/{SOURCE_RUN_ID}/checkpoints",
    )
)
DRIVE_OUTPUT = Path(
    os.environ.get(
        "STAGE5_T1_EMA_AUDIT_DRIVE_OUTPUT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}",
    )
)
MIN_CHECKPOINT_BYTES = 1024


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def checkpoint_file_usable(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_file() and candidate.stat().st_size >= MIN_CHECKPOINT_BYTES


def publish() -> None:
    subprocess.run(
        ["git", "pull", "--rebase", "--autostash", "origin", "main"],
        cwd=ROOT,
        check=False,
    )
    for path in sorted(RUN_DIR.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".md", ".log"}:
            subprocess.run(
                ["git", "add", "-f", path.relative_to(ROOT).as_posix()],
                cwd=ROOT,
                check=True,
            )
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", f"Record T1-lite EMA audit {RUN_ID} [skip ci]"])
        if subprocess.run(["git", "push", "origin", "main"], cwd=ROOT).returncode:
            run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
            run(["git", "push", "origin", "main"])


def restore_inputs() -> tuple[dict[str, Path], list[str]]:
    training = read_json(SOURCE_DIR / "train" / "training_summary.json")
    endpoints = {
        "t1_lite_raw_step_10500.pt": str(training["raw_checkpoint_sha256"]),
        "t1_lite_ema_step_10500.pt": str(training["ema_checkpoint_sha256"]),
    }
    progress = {f"t1_progress_step_{step}.pt": "" for step in (500, 2500, 6500, 8500)}
    restore_dir = RUN_DIR / "restored"
    restore_dir.mkdir(parents=True, exist_ok=True)
    restored: dict[str, Path] = {}
    missing_progress: list[str] = []
    for name, expected_sha in {**endpoints, **progress}.items():
        source = DRIVE_SOURCE / name
        if not checkpoint_file_usable(source):
            if name in endpoints:
                raise FileNotFoundError(
                    f"missing or truncated required Drive endpoint: {source}"
                )
            missing_progress.append(name)
            destination = restore_dir / name
            if destination.exists():
                destination.unlink()
            observed_bytes = source.stat().st_size if source.exists() else None
            print(
                f"unusable_optional_stage_checkpoint={source} bytes={observed_bytes}",
                flush=True,
            )
            continue
        destination = restore_dir / name
        if not checkpoint_file_usable(destination) or destination.stat().st_size != source.stat().st_size:
            shutil.copy2(source, destination)
        restored[name] = destination
        print(
            f"restored_audit_input={destination} bytes={destination.stat().st_size} "
            f"expected_sha256={expected_sha or 'recorded_inside_progress_payload'}",
            flush=True,
        )
    (RUN_DIR / "restore_manifest.json").write_text(
        json.dumps(
            {
                "drive_source": str(DRIVE_SOURCE),
                "restored": {name: str(path) for name, path in restored.items()},
                "missing_stage_checkpoints": missing_progress,
                "endpoint_checkpoints_required": sorted(endpoints),
                "stage_checkpoint_absence_is_partial_evidence": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return restored, missing_progress


def write_receipt(summary: dict[str, Any]) -> None:
    lines = [
        "# T1-lite EMA Audit Receipt",
        "",
        f"- Status: `{summary['status']}`",
        f"- Registered verdict unchanged: `{summary['registered_verdict_immutable']}`",
        f"- Training performed: `{summary['training_performed']}`",
        "",
        "This is a post-hoc localization audit. It does not replace the registered EMA-primary result.",
        "",
        "## Selected confirmations",
        "",
    ]
    lines.append(
        f"- Stage checkpoints available: "
        f"`{summary['stage_checkpoint_coverage']['available']}/"
        f"{summary['stage_checkpoint_coverage']['required']}`."
    )
    for label, value in summary["selected_confirmation_variants"].items():
        metrics = summary["full_pilot_confirmations"][value]
        lines.append(
            f"- {label}: `{value}`; forced `{metrics['forced_correct']}/{metrics['total']}`, "
            f"exact depth `{metrics['exact_selected_depth_correct']}/{metrics['total']}`."
        )
    lines.extend(["", "Interpretation is deferred until the complete receipt is reviewed.", ""])
    (RUN_DIR / "receipt.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    required = [
        SOURCE_DIR / "train" / "training_summary.json",
        SOURCE_DIR / "data" / "liveness_pilot_256.jsonl",
        ROOT / "docs" / "PAPER2_T1_LITE_EMA_AUDIT_SPEC_20260725.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"EMA audit immutable sources missing: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    restored, missing_progress = restore_inputs()
    training = read_json(SOURCE_DIR / "train" / "training_summary.json")
    run(
        [
            sys.executable,
            "eval/eval_t1_lite_ema_audit.py",
            "--raw_checkpoint",
            str(restored["t1_lite_raw_step_10500.pt"]),
            "--ema_checkpoint",
            str(restored["t1_lite_ema_step_10500.pt"]),
            "--progress_dir",
            str(RUN_DIR / "restored"),
            "--archived_stage_receipts_dir",
            str(SOURCE_DIR / "train" / "stage_receipts"),
            "--allow_missing_stage_checkpoints",
            "--pilot_jsonl",
            str(SOURCE_DIR / "data" / "liveness_pilot_256.jsonl"),
            "--output_dir",
            str(RUN_DIR),
            "--expected_raw_sha256",
            str(training["raw_checkpoint_sha256"]),
            "--expected_ema_sha256",
            str(training["ema_checkpoint_sha256"]),
            "--device",
            "cuda",
            "--dtype",
            os.environ.get("STAGE5_T1_EMA_AUDIT_DTYPE", "bfloat16"),
            "--batch_size",
            os.environ.get("STAGE5_T1_EMA_AUDIT_BATCH_SIZE", "8"),
        ]
    )
    summary = read_json(RUN_DIR / "summary.json")
    if summary.get("training_performed") is not False:
        raise AssertionError("EMA audit unexpectedly performed training")
    if summary.get("registered_verdict_immutable") != "registered_negative":
        raise AssertionError("EMA audit changed the registered verdict")
    summary_missing = set(
        summary.get("stage_checkpoint_coverage", {}).get("missing_names", [])
    )
    if not set(missing_progress).issubset(summary_missing):
        raise AssertionError("EMA audit dropped a known-missing stage checkpoint")
    write_receipt(summary)
    if DRIVE_OUTPUT.exists():
        shutil.rmtree(DRIVE_OUTPUT)
    shutil.copytree(RUN_DIR, DRIVE_OUTPUT, ignore=shutil.ignore_patterns("*.pt"))
    publish()
    print(f"T1-lite EMA audit finished: {RUN_DIR.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
