"""Run exactly the next credit-worthy Stage 5 full ARC assessment.

This is a narrow Colab entrypoint for low-credit sessions. It does not ask the
planner for a fresh action and it does not chain follow-up jobs. It runs the
full balanced ARC-Easy/ARC-Challenge assessment for a known passed proxy-gate
summary, pushes the resulting text artifacts, then optionally disconnects the
Colab runtime.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_SUMMARY = (
    "outputs/stage5/"
    "stage5_arc_agi_colab_continue_20260621_232031_plan_arc_mix_probe/"
    "summary.json"
)
RUN_ID = os.environ.get("STAGE5_FULL_ASSESS_ONCE_RUN_ID") or os.environ.get(
    "STAGE5_RECOVERY_FULL_ASSESS_RUN_ID",
    time.strftime("stage5_full_assessment_once_%Y%m%d_%H%M%S"),
)
SOURCE_SUMMARY = os.environ.get("STAGE5_FULL_ASSESS_SOURCE_SUMMARY") or os.environ.get(
    "STAGE5_RECOVERY_FULL_ASSESS_SOURCE_SUMMARY",
    DEFAULT_SOURCE_SUMMARY,
)
AUTO_DISCONNECT = os.environ.get("STAGE5_FULL_ASSESS_AUTO_DISCONNECT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
ALLOW_CPU = os.environ.get("STAGE5_FULL_ASSESS_ALLOW_CPU", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def mask(value: str) -> str:
    masked = value
    for key in ("GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            masked = masked.replace(token, "****")
    return masked


def run(cmd: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    printable = mask(" ".join(map(str, cmd)))
    print("$", printable, flush=True)
    try:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        message = f"command not found: {cmd[0]} ({exc})\n"
        print(message, end="", flush=True)
        proc = subprocess.CompletedProcess(cmd, 127, message, None)
        if check:
            raise RuntimeError(f"failed: {printable}") from exc
        return proc
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    proc = subprocess.CompletedProcess(cmd, process.wait(), "".join(chunks), None)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {printable}")
    return proc


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["STAGE5_RECOVERY_FULL_ASSESS_RUN_ID"] = RUN_ID
    env["STAGE5_RECOVERY_FULL_ASSESS_SOURCE_SUMMARY"] = SOURCE_SUMMARY
    env.setdefault("STAGE5_RECOVERY_FULL_ASSESS_PUSH", "1")
    return env


def cuda_runtime_status() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency/environment guard
        return False, f"torch import failed: {exc}"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is false"
    try:
        return True, torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - unusual CUDA state
        return True, f"cuda available, device name unavailable: {exc}"


def require_cuda_runtime() -> None:
    if ALLOW_CPU:
        print("CPU fallback allowed by STAGE5_FULL_ASSESS_ALLOW_CPU=1.", flush=True)
        return
    available, detail = cuda_runtime_status()
    if not available:
        raise RuntimeError(
            "Refusing to run full ARC assessment without CUDA. "
            "Attach an A100/GPU runtime or set STAGE5_FULL_ASSESS_ALLOW_CPU=1 for an intentional CPU run. "
            f"Detail: {detail}"
        )
    print(f"CUDA runtime OK: {detail}", flush=True)


def disconnect_if_requested() -> None:
    if not AUTO_DISCONNECT:
        return
    try:
        from google.colab import runtime  # type: ignore

        print("Disconnecting Colab runtime to conserve A100 credits...", flush=True)
        runtime.unassign()
    except Exception as exc:  # pragma: no cover - Colab only
        print(f"Runtime disconnect skipped/failed: {exc}", flush=True)


def run_assessment() -> int:
    source = ROOT / SOURCE_SUMMARY
    if not source.exists():
        raise FileNotFoundError(f"Missing source summary: {source}")
    require_cuda_runtime()
    run(["git", "status", "-sb"], check=False)
    run(["git", "log", "--oneline", "-5"], check=False)
    run(["nvidia-smi"], check=False)
    print(f"RUN_ID={RUN_ID}", flush=True)
    print(f"SOURCE_SUMMARY={SOURCE_SUMMARY}", flush=True)
    run([sys.executable, "colab/run_stage5_recovery_full_assessment.py"], env=child_env())
    return 0


def main() -> int:
    try:
        return run_assessment()
    finally:
        disconnect_if_requested()


if __name__ == "__main__":
    raise SystemExit(main())
