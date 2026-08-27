"""Execute and publish the governed, score-only recirculation Phase-A sweep."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any

from colab.stage5_publish_utils import DEFAULT_PUBLISH_SUFFIXES, publishable_artifact_paths


KIND = "paper2_recirculation_phase_a_runner_v1"
STAGE = "stage5_paper2_recirculation_phase_a_20260827"
PHASE0_STAGE = "stage5_paper2_recirculation_20260827"
PANEL = Path(
    "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
)
PANEL_CANONICAL_LF_SHA256 = (
    "c0e15a890b598544059ac337cc475123f97c05e3c1626febcdee1c6d8fe02615"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_lf_sha256(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in normalized:
        raise RuntimeError("frozen DEV panel contains an unauthorized carriage return")
    return hashlib.sha256(normalized).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], *, cwd: Path) -> None:
    printable = " ".join(command)
    token = os.environ.get("GH_TOKEN", "")
    if token:
        printable = printable.replace(token, "****")
    print("$", printable, flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_tee(command: list[str], *, cwd: Path, log_path: Path) -> None:
    print("$", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if resolved != target and resolved not in target.parents:
                raise RuntimeError(f"archive member escapes destination: {member.name}")
        archive.extractall(destination, filter="data")


def archive_artifacts(artifact_root: Path, export_path: Path) -> dict[str, Any]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = export_path.with_suffix(export_path.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(artifact_root, arcname=STAGE)
    temporary.replace(export_path)
    receipt = {
        "path": str(export_path),
        "bytes": export_path.stat().st_size,
        "sha256": sha256_file(export_path),
    }
    print(
        f"recirculation_phase_a_export path={export_path} bytes={receipt['bytes']} "
        f"sha256={receipt['sha256']}",
        flush=True,
    )
    return receipt


def file_receipts(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def force_add_public_receipts(*, root: Path, public_dir: Path) -> list[Path]:
    suffixes = set(DEFAULT_PUBLISH_SUFFIXES) | {".png", ".svg"}
    paths = publishable_artifact_paths(public_dir, allowed_suffixes=suffixes)
    if not paths:
        raise RuntimeError(f"no publishable Phase-A receipts found under {public_dir}")
    for path in paths:
        run(["git", "add", "-f", str(path.relative_to(root))], cwd=root)
    return paths


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "training" / "paper2_recirculation_phase_a_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    phase0_archive = Path(os.environ["RECIRCULATION_PHASE0_ARCHIVE"])
    if not phase0_archive.is_file():
        raise RuntimeError(f"banked Phase-0 archive is missing: {phase0_archive}")
    expected_archive = lock["phase0"]["archive"]
    observed_archive = {
        "bytes": phase0_archive.stat().st_size,
        "sha256": sha256_file(phase0_archive),
    }
    if observed_archive != expected_archive:
        raise RuntimeError("banked Phase-0 archive identity changed")

    scratch_parent = Path(
        os.environ.get(
            "RECIRCULATION_PHASE_A_SCRATCH_PARENT",
            "/mnt/local-scratch/recurrent-qwen-svgd-stage",
        )
    )
    if not scratch_parent.exists():
        scratch_parent = Path("/content/local-scratch")
    artifact_root = scratch_parent / STAGE
    phase0_extract_parent = scratch_parent / "recirculation_phase0_input"
    phase0_root = phase0_extract_parent / PHASE0_STAGE
    resume_archive = Path(
        os.environ.get(
            "RECIRCULATION_PHASE_A_RESUME_ARCHIVE",
            "/content/recirculation-phase-a-progress.tar.gz",
        )
    )
    if resume_archive.is_file() and not artifact_root.exists():
        safe_extract(resume_archive, scratch_parent)
        print(f"recirculation_phase_a_resume_archive={resume_archive}", flush=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    if not phase0_root.exists():
        safe_extract(phase0_archive, phase0_extract_parent)
    progress_archive = Path(
        os.environ.get(
            "RECIRCULATION_PHASE_A_PROGRESS_ARCHIVE",
            "/content/recirculation-phase-a-progress.tar.gz",
        )
    )
    export_path = Path(
        os.environ.get(
            "RECIRCULATION_PHASE_A_EXPORT_PATH",
            "/content/recirculation-phase-a-artifacts.tar.gz",
        )
    )
    model_cache = scratch_parent / "recirculation_phase_a_hf_cache"
    model_cache.mkdir(parents=True, exist_ok=True)
    child_log = artifact_root / "receipts" / "phase_a.log"
    runner_status_path = artifact_root / "receipts" / "runner_status.json"
    public_dir = root / "outputs" / "stage5" / STAGE / "phase_a"
    started = time.perf_counter()
    status: dict[str, Any] = {
        "kind": KIND,
        "status": "preflight",
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "phase_b_training_authorized": False,
        "confirm_scored": False,
        "eval_e_scored": False,
        "phase0_archive": observed_archive,
    }
    atomic_json(runner_status_path, status)
    try:
        panel = root / PANEL
        if not panel.is_file() or canonical_lf_sha256(panel) != PANEL_CANONICAL_LF_SHA256:
            raise RuntimeError("frozen 1,024-row DEV panel identity changed")
        status.update(
            status="running_phase_a",
            lock_sha256=sha256_file(lock_path),
            artifact_root=str(artifact_root),
            phase0_root=str(phase0_root),
        )
        atomic_json(runner_status_path, status)
        run_tee(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_recirculation_phase_a",
                "--lock",
                str(lock_path),
                "--panel",
                str(panel),
                "--phase0_root",
                str(phase0_root),
                "--repo_root",
                str(root),
                "--artifact_root",
                str(artifact_root),
                "--model_cache",
                str(model_cache),
                "--progress_archive",
                str(progress_archive),
                "--generation_batch_size",
                os.environ.get("RECIRCULATION_GENERATION_BATCH_SIZE", "8"),
                "--nll_batch_size",
                os.environ.get("RECIRCULATION_NLL_BATCH_SIZE", "32"),
            ],
            cwd=root,
            log_path=child_log,
        )
        phase_a_status = json.loads(
            (artifact_root / "receipts" / "status.json").read_text(encoding="utf-8")
        )
        scientific_outcome = phase_a_status.get("status")
        if scientific_outcome not in {
            "phase_a_complete_awaiting_strategy_adjudication",
            "overrun_stop_awaiting_relay",
        }:
            raise RuntimeError(f"Phase A ended without a registered outcome: {scientific_outcome}")

        if public_dir.exists():
            shutil.rmtree(public_dir)
        shutil.copytree(artifact_root / "receipts" / "phase_a", public_dir)
        public_summary = {
            "kind": "paper2_recirculation_phase_a_public_summary_v1",
            "status": scientific_outcome,
            "phase_a": (
                json.loads((public_dir / "phase_a_summary.json").read_text(encoding="utf-8"))
                if (public_dir / "phase_a_summary.json").is_file()
                else None
            ),
            "cost": {
                "completed_measurements": phase_a_status["completed_measurements"],
                "phase_a_elapsed_seconds": phase_a_status["phase_a_elapsed_seconds"],
                "actual_total_a100_hours": phase_a_status["actual_total_a100_hours"],
                "expected_total_a100_hours_at_checkpoint": phase_a_status[
                    "expected_total_a100_hours_at_checkpoint"
                ],
                "actual_to_expected_multiplier": phase_a_status[
                    "actual_to_expected_multiplier"
                ],
                "cost_ceiling_a100_hours": phase_a_status["cost_ceiling_a100_hours"],
            },
            "public_receipts": file_receipts(public_dir),
            "private_receipts": file_receipts(artifact_root / "private" / "phase_a"),
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "phase_b_training_authorized": False,
            "confirm_scored": False,
            "eval_e_scored": False,
            "strategy_key_resolved": False,
        }
        atomic_json(public_dir / "summary.json", public_summary)
        status.update(
            status=scientific_outcome,
            scientific_outcome=scientific_outcome,
            elapsed_seconds=time.perf_counter() - started,
            phase_a_summary={
                "bytes": (public_dir / "summary.json").stat().st_size,
                "sha256": sha256_file(public_dir / "summary.json"),
            },
            publication_status="pending",
        )
        atomic_json(runner_status_path, status)
        status["export"] = archive_artifacts(artifact_root, export_path)
        atomic_json(runner_status_path, status)

        try:
            force_add_public_receipts(root=root, public_dir=public_dir)
            changed = subprocess.run(
                ["git", "diff", "--cached", "--quiet"], cwd=root, check=False
            ).returncode
            if changed:
                run(["git", "commit", "-m", "Bank recirculation Phase A sweep"], cwd=root)
                push_ref = os.environ.get(
                    "RECIRCULATION_PUSH_REF", "codex/bicameral-stage0"
                )
                run(["git", "push", "origin", f"HEAD:{push_ref}"], cwd=root)
            status.update(
                publication_status="published",
                git_head=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
            )
        except Exception as publication_error:
            status.update(
                publication_status="failed_after_scientific_completion",
                publication_exception_type=type(publication_error).__name__,
                publication_exception=str(publication_error),
            )
        atomic_json(runner_status_path, status)
        status["export"] = archive_artifacts(artifact_root, export_path)
        atomic_json(runner_status_path, status)
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        status.update(
            status="execution_failed",
            elapsed_seconds=time.perf_counter() - started,
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
            child_log_tail=(
                child_log.read_text(encoding="utf-8", errors="replace")[-30000:]
                if child_log.is_file()
                else ""
            ),
        )
        atomic_json(runner_status_path, status)
        try:
            status["export"] = archive_artifacts(artifact_root, export_path)
            atomic_json(runner_status_path, status)
        except Exception as archive_error:
            print(f"recirculation_phase_a_export_failed={archive_error}", flush=True)
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
