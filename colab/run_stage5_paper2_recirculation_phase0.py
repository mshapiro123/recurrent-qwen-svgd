"""Execute, persist, and publish the governed recirculation Phase-0 gates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


KIND = "paper2_recirculation_phase0_runner_v1"
STAGE = "stage5_paper2_recirculation_20260827"
PANEL = Path(
    "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
)
PANEL_SHA256 = "2e7e1d2be75ef8b7a536fe9a5554b3bf7883d54b1472e5b76ef50380c8270642"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def run(command: list[str], *, cwd: Path) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def file_receipts(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    drive_root = Path(
        os.environ.get(
            "RECIRCULATION_DRIVE_ROOT",
            f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{STAGE}",
        )
    )
    receipts_dir = drive_root / "receipts" / "phase0"
    private_dir = drive_root / "private" / "phase0"
    public_dir = root / "outputs" / "stage5" / STAGE / "phase0"
    scratch_root = Path(
        os.environ.get(
            "RECIRCULATION_SCRATCH_ROOT",
            "/mnt/local-scratch/recurrent-qwen-svgd-stage/recirculation_phase0",
        )
    )
    if not scratch_root.parent.exists():
        scratch_root = Path("/content/local-scratch/recirculation_phase0")
    model_cache = scratch_root / "hf_cache"
    status_path = drive_root / "receipts" / "status.json"
    child_log = drive_root / "receipts" / "phase0.log"
    started = time.perf_counter()
    status: dict[str, Any] = {
        "kind": KIND,
        "status": "preflight",
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
        "phase_b_training_authorized": False,
    }
    atomic_json(status_path, status)
    try:
        panel = root / PANEL
        if not panel.is_file() or sha256_file(panel) != PANEL_SHA256:
            raise RuntimeError("frozen 1,024-row DEV panel identity changed")
        lock = root / "training" / "paper2_recirculation_phase0_lock.json"
        lock_payload = json.loads(lock.read_text(encoding="utf-8"))
        for authority in lock_payload["authorities"]:
            path = root / "docs" / authority["filename"]
            observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if observed != {"bytes": authority["bytes"], "sha256": authority["sha256"]}:
                raise RuntimeError(f"authority identity changed: {authority['filename']}")
        receipts_dir.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)
        model_cache.mkdir(parents=True, exist_ok=True)
        status.update(
            status="running_phase0",
            panel_sha256=PANEL_SHA256,
            lock_sha256=sha256_file(lock),
            scratch=str(scratch_root),
        )
        atomic_json(status_path, status)
        run_tee(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_recirculation_phase0",
                "--lock",
                str(lock),
                "--panel",
                str(panel),
                "--output_dir",
                str(receipts_dir),
                "--private_dir",
                str(private_dir),
                "--model_cache",
                str(model_cache),
                "--generation_batch_size",
                os.environ.get("RECIRCULATION_GENERATION_BATCH_SIZE", "8"),
                "--nll_batch_size",
                os.environ.get("RECIRCULATION_NLL_BATCH_SIZE", "32"),
            ],
            cwd=root,
            log_path=child_log,
        )
        phase0_status = json.loads(
            (receipts_dir / "phase0_status.json").read_text(encoding="utf-8")
        )
        phase0_outcome = phase0_status.get("status")
        if phase0_outcome not in {
            "phase0_pass_awaiting_relay",
            "cost_ceiling_stop",
        }:
            raise RuntimeError(f"Phase 0 ended without a registered outcome: {phase0_outcome}")
        if public_dir.exists():
            shutil.rmtree(public_dir)
        shutil.copytree(receipts_dir, public_dir)
        summary = {
            "kind": "paper2_recirculation_phase0_public_summary_v1",
            "status": phase0_outcome,
            "runtime": phase0_status["runtime"],
            "identity_qwen": phase0_status["identity_qwen"],
            "identity_gemma": phase0_status["identity_gemma"],
            "battery_anchor": phase0_status["battery_anchor"],
            "qwen_timing": phase0_status["qwen_timing"],
            "gemma_anchor": phase0_status["gemma_anchor"],
            "cost_projection": phase0_status["cost_projection"],
            "public_receipts": file_receipts(public_dir),
            "private_receipts": file_receipts(private_dir),
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
        }
        atomic_json(public_dir / "summary.json", summary)
        push_ref = os.environ.get(
            "RECIRCULATION_PUSH_REF", "codex/bicameral-stage0"
        )
        run(["git", "add", str(public_dir.relative_to(root))], cwd=root)
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=root, check=False
        )
        if result.returncode:
            run(["git", "commit", "-m", "Bank recirculation Phase 0 gates"], cwd=root)
            run(["git", "push", "origin", f"HEAD:{push_ref}"], cwd=root)
        status.update(
            status=(
                "complete_awaiting_phase0_relay"
                if phase0_outcome == "phase0_pass_awaiting_relay"
                else "cost_ceiling_stop_awaiting_relay"
            ),
            elapsed_seconds=time.perf_counter() - started,
            phase0_summary={
                "bytes": (public_dir / "summary.json").stat().st_size,
                "sha256": sha256_file(public_dir / "summary.json"),
            },
            git_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
        )
        atomic_json(status_path, status)
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        status.update(
            status="failed",
            elapsed_seconds=time.perf_counter() - started,
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
            child_log_tail=(
                child_log.read_text(encoding="utf-8", errors="replace")[-20000:]
                if child_log.is_file()
                else ""
            ),
        )
        atomic_json(status_path, status)
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
