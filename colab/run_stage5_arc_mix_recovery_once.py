"""Run exactly one low-credit Stage 5 ARC-mix recovery proxy gate.

Use this after a full balanced ARC assessment reports
``needs_competence_recovery``. It runs one bounded competence-preserving
ARC/Opus mix arm, writes the normal ARC-mix summary artifacts, pushes safe text
outputs through the delegated runner, and optionally disconnects Colab.

It deliberately does not chain a full balanced assessment. If the proxy gate is
positive, launch the full assessment as a separate, explicit A100 decision.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE_SUMMARY = "outputs/stage5/stage5_full_assessment_once_20260622_005522/summary.json"
RUN_ID = os.environ.get("STAGE5_ARC_MIX_ONCE_RUN_ID") or os.environ.get(
    "STAGE5_ARC_MIX_RUN_ID",
    time.strftime("stage5_arc_mix_recovery_once_%Y%m%d_%H%M%S"),
)
SOURCE_SUMMARY = os.environ.get("STAGE5_ARC_MIX_ONCE_SOURCE_SUMMARY") or os.environ.get(
    "STAGE5_ARC_MIX_SOURCE_SUMMARY",
    DEFAULT_SOURCE_SUMMARY,
)
AUTO_DISCONNECT = os.environ.get("STAGE5_ARC_MIX_ONCE_AUTO_DISCONNECT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
ALLOW_CPU = os.environ.get("STAGE5_ARC_MIX_ONCE_ALLOW_CPU", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
PREFLIGHT_ONLY = os.environ.get("STAGE5_ARC_MIX_ONCE_PREFLIGHT_ONLY", "0").strip().lower() in {
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


def run(
    cmd: list[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
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
    env["STAGE5_ARC_MIX_RUN_ID"] = RUN_ID
    env["STAGE5_ARC_MIX_SOURCE_SUMMARY"] = SOURCE_SUMMARY
    env.setdefault("STAGE5_ARC_MIX_ARMS", "arc_mix_response_w01_lr2e6")
    env.setdefault("STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT", "2")
    env.setdefault("STAGE5_ARC_MIX_ARC_EASY_REPEAT", "4")
    env.setdefault("STAGE5_ARC_MIX_ARC_EVAL_LIMIT", "128")
    env.setdefault("STAGE5_ARC_MIX_OPUS_LIMIT", "3000")
    env.setdefault("STAGE5_ARC_MIX_PUSH", "1")
    return env


def read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def preflight_source_summary(source: Path) -> tuple[str | None, Path]:
    from colab.run_stage5_balanced_arc_mix_gate import selected_checkpoint

    payload = read_json(source)
    checkpoint = selected_checkpoint(payload)
    status = payload.get("status")
    print(f"source_status={status}", flush=True)
    print(f"resume_checkpoint={checkpoint}", flush=True)
    print(f"checkpoint_exists={checkpoint.exists()}", flush=True)
    return status, checkpoint


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
        print("CPU fallback allowed by STAGE5_ARC_MIX_ONCE_ALLOW_CPU=1.", flush=True)
        return
    available, detail = cuda_runtime_status()
    if not available:
        raise RuntimeError(
            "Refusing to run ARC-mix recovery without CUDA. "
            "Attach an A100/GPU runtime or set STAGE5_ARC_MIX_ONCE_ALLOW_CPU=1 for an intentional CPU run. "
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


def run_recovery_gate() -> int:
    source = ROOT / SOURCE_SUMMARY
    if not source.exists():
        raise FileNotFoundError(f"Missing source summary: {source}")
    preflight_source_summary(source)
    if PREFLIGHT_ONLY:
        print("Preflight-only mode complete; no training launched.", flush=True)
        return 0
    require_cuda_runtime()
    run(["git", "status", "-sb"], check=False)
    run(["git", "log", "--oneline", "-5"], check=False)
    run(["nvidia-smi"], check=False)
    env = child_env()
    print(f"RUN_ID={env['STAGE5_ARC_MIX_RUN_ID']}", flush=True)
    print(f"SOURCE_SUMMARY={env['STAGE5_ARC_MIX_SOURCE_SUMMARY']}", flush=True)
    print(f"ARMS={env['STAGE5_ARC_MIX_ARMS']}", flush=True)
    print(f"ARC_EVAL_LIMIT={env['STAGE5_ARC_MIX_ARC_EVAL_LIMIT']}", flush=True)
    print(f"OPUS_LIMIT={env['STAGE5_ARC_MIX_OPUS_LIMIT']}", flush=True)
    run([sys.executable, "colab/run_stage5_balanced_arc_mix_gate.py"], env=env)
    return 0


def main() -> int:
    try:
        return run_recovery_gate()
    finally:
        disconnect_if_requested()


if __name__ == "__main__":
    raise SystemExit(main())
