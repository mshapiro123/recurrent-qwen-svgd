"""Single-runtime Stage 5 Colab continuation wrapper.

This is the preferred "keep the A100 moving" entrypoint after the repo has
been cloned or pulled in Colab. It runs focused smoke tests for the planner and
Gate 1 path, executes the bounded next-action loop, writes the progress ledger,
and commits Stage 5 outputs when they changed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_COLAB_CONTINUE_RUN_ID") or time.strftime(
    "stage5_arc_agi_colab_continue_%Y%m%d_%H%M%S"
)
SAFE_OUTPUT_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
BINARY_OUTPUT_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
MAX_COMMIT_ARTIFACT_BYTES = int(os.environ.get("STAGE5_ARC_AGI_COMMIT_MAX_ARTIFACT_BYTES", "25000000"))


def focused_test_paths() -> list[str]:
    return [
        "tests/test_stage5_autopilot.py",
        "tests/test_arc_agi_training_signal.py",
        "tests/test_stage5_next_plan.py",
        "tests/test_stage5_next_action.py",
        "tests/test_stage5_arc_agi_candidate_distill_gate.py",
        "tests/test_stage5_arc_agi_candidate_gate.py",
        "tests/test_stage5_curriculum_particle_autopilot.py",
        "tests/test_stage5_gate1_assessment.py",
        "tests/test_stage5_gate2_assessment.py",
        "tests/test_stage5_recipe_control_assessment.py",
        "tests/test_stage5_recipe_selector_conversion.py",
        "tests/test_stage5_release_gate.py",
        "tests/test_stage5_selector_replication.py",
        "tests/test_stage5_benchmark_assessment.py",
        "tests/test_stage5_claim_packet.py",
        "tests/test_stage5_arc_agi_baseline_registry.py",
        "tests/test_stage5_arc_agi_sota_comparison.py",
        "tests/test_stage5_sft_gates.py",
        "tests/test_stage5_progress_ledger.py",
        "tests/test_stage5_benchmark_suite.py",
        "tests/test_lora.py",
        "tests/test_stage5_dense_sft_control.py",
    ]


def continuation_profile() -> str:
    return os.environ.get("STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE", "gate").strip().lower()


def default_max_actions() -> str:
    explicit = os.environ.get("STAGE5_ARC_AGI_COLAB_CONTINUE_MAX_ACTIONS")
    if explicit:
        return explicit
    return {
        "single": "1",
        "gate": "3",
        "same_recipe": "6",
        "claim": "10",
    }.get(continuation_profile(), "3")


def default_env() -> dict[str, str]:
    return {
        "STAGE5_ARC_AGI_NEXT_ACTION_RUN_ID": RUN_ID,
        "STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE": "1",
        "STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS": default_max_actions(),
        "STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT": "0",
        "STAGE5_ARC_AGI_AUTOPILOT_TRACE_SFT_GATE_ARMS": (
            "grid_only,symbolic_program_trace_covered,symbolic_state_trace_covered"
        ),
        "STAGE5_ARC_AGI_NEXT_PLAN_TRACE_SFT_GATE_ARMS": (
            "grid_only,symbolic_program_trace_covered,symbolic_state_trace_covered"
        ),
        "DRIVE_BACKUP_DIR": "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts",
    }


def stage5_output_paths() -> list[str]:
    return ["outputs/stage5", "outputs/hf_exports"]


def is_safe_output_artifact(path: Path, *, max_bytes: int = MAX_COMMIT_ARTIFACT_BYTES) -> bool:
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    if suffix in BINARY_OUTPUT_SUFFIXES:
        return False
    if suffix not in SAFE_OUTPUT_SUFFIXES:
        return False
    try:
        return path.stat().st_size <= max_bytes
    except OSError:
        return False


def committable_stage5_files() -> list[str]:
    files: list[str] = []
    for root_name in stage5_output_paths():
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if is_safe_output_artifact(path):
                files.append(path.relative_to(ROOT).as_posix())
    return sorted(files)


def git_add_files(paths: list[str]) -> None:
    chunk_size = 100
    for index in range(0, len(paths), chunk_size):
        run(["git", "add", "-f", *paths[index : index + chunk_size]])


def post_action_commands() -> list[list[str]]:
    """No-GPU follow-ups that make the committed run directory self-explaining."""

    return [
        [sys.executable, "colab/assess_stage5_release_gate.py"],
        [sys.executable, "colab/summarize_stage5_progress.py"],
    ]


def mask_command(cmd: list[str]) -> str:
    printable = " ".join(map(str, cmd))
    for key in ("GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            printable = printable.replace(token, "****")
    return printable


def run(cmd: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("$", mask_command(cmd), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    proc = subprocess.CompletedProcess(cmd, process.wait(), "".join(chunks), None)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {mask_command(cmd)}")
    return proc


def mount_drive_if_available() -> None:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as exc:  # pragma: no cover - Colab only
        print(f"Drive mount skipped/failed: {exc}")


def ensure_git_identity() -> None:
    run(["git", "config", "user.email", "colab-runner@local"], check=False)
    run(["git", "config", "user.name", "Colab Runner"], check=False)


def commit_stage5_outputs() -> None:
    run(["git", "status", "-sb"], check=False)
    files = committable_stage5_files()
    if not files:
        print("No safe Stage 5 output artifacts to commit.")
        return
    print(f"safe_stage5_artifacts={len(files)}")
    git_add_files(files)
    status = run(["git", "diff", "--cached", "--quiet"], check=False)
    if status.returncode == 0:
        print("No Stage 5 outputs to commit.")
        return
    run(["git", "commit", "-m", f"Record Stage 5 continuation {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    os.environ.update({key: value for key, value in default_env().items() if key not in os.environ})
    mount_drive_if_available()
    ensure_git_identity()
    run(["nvidia-smi"], check=False)
    run([sys.executable, "-m", "pytest", "-q", *focused_test_paths()])
    print("RUN_ID", RUN_ID)
    print("CONTINUATION_PROFILE", continuation_profile())
    run([sys.executable, "colab/run_stage5_next_action.py"])
    for command in post_action_commands():
        run(command, check=False)
    commit_stage5_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
