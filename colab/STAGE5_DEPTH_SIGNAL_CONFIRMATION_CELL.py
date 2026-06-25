"""Colab cell: run depth-routing recovery then expanded hard-content benchmark.

This is the bounded one-shot action after the re-entry repair diagnostics:

1. run Stage 4 re-entry recovery SFT with learned loop control,
2. publish its recovery and post-reentry health summaries,
3. run an expanded debiased benchmark focused on hard content signal, and
4. publish the benchmark suite plus assessment.

The run avoids gated GPQA by default and uses an open ARC-Challenge hard slice
as the third benchmark leg. Set explicit env vars before launch to override.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_DEPTH_SIGNAL_CONFIRMATION_CELL_VERSION = "depth_signal_confirmation_v1"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


def secret(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


def redact(text: str) -> str:
    out = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            out = out.replace(token, "****")
    return out


def run(cmd: list[str], *, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        print(redact(proc.stdout), flush=True)
    if check and proc.returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(redact(proc.stdout or "").splitlines()[-180:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def ensure_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-8"])


def env_default(name: str, value: str) -> None:
    os.environ.setdefault(name, value)


def print_current_pointer(label: str) -> str:
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    value = pointer.read_text(encoding="utf-8").strip() if pointer.exists() else ""
    print(f"{label}: {value}", flush=True)
    return value


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN/GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print("HF token loaded", flush=True)
else:
    print("HF token not found; public Hub access only.", flush=True)


try:
    print(f"cell_version={STAGE5_DEPTH_SIGNAL_CONFIRMATION_CELL_VERSION}", flush=True)
    run(["nvidia-smi"], cwd=Path("/content"))
    ensure_repo()

    # Stage 4 recovery: keep this bounded for L4/T4, but emphasize loop-control
    # enough to test whether hard/deep examples start routing deeper.
    run_id = time.strftime("stage5_depth_signal_recovery_%Y%m%d_%H%M%S")
    env_default("STAGE5_REENTRY_RECOVERY_RUN_ID", run_id)
    env_default("STAGE5_REENTRY_RECOVERY_CHILD_RUN_ID", f"{run_id}_curriculum_sft")
    env_default("STAGE5_REENTRY_RECOVERY_STEPS", "100")
    env_default("STAGE5_REENTRY_RECOVERY_LR", "5e-6")
    env_default("STAGE5_REENTRY_RECOVERY_HALT_TARGET_NLL_WEIGHT", "6.0")
    env_default("STAGE5_REENTRY_RECOVERY_LOOP_CONTROL_CE_WEIGHT", "5.0")
    env_default("STAGE5_REENTRY_RECOVERY_REQUIRE_TARGET_LOOP_GRADIENT", "0")
    env_default("STAGE5_REENTRY_RECOVERY_REENTRY_ADAPTER_MODE", "spectral")
    env_default("STAGE5_REENTRY_RECOVERY_OPTIMIZER_MODULES", "bridge,reentry,halt,lora")
    env_default("STAGE5_REENTRY_RECOVERY_DISCONNECT", "0")

    print("=== Stage 4: depth-routing recovery ===", flush=True)
    exec((ROOT / "colab" / "STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py").read_text(encoding="utf-8"))
    recovery_summary = print_current_pointer("recovery_summary")

    # Stage 5 benchmark: open hard fallback plus ARC-Challenge power-up.
    env_default("STAGE5_DEBIASED_BENCHMARK_SUITE_PROFILE", "depth_signal_confirmation")
    env_default("STAGE5_DEBIASED_BENCHMARKS", "arc_easy,arc_challenge,open_hard_arc_challenge")
    env_default("STAGE5_DEBIASED_SCORE_TARGETS", "content_question_only,cyclic_label_aggregated")
    env_default("STAGE5_DEBIASED_ARC_EASY_LIMIT", "128")
    env_default("STAGE5_DEBIASED_ARC_CHALLENGE_LIMIT", "256")
    env_default("STAGE5_DEBIASED_OPEN_HARD_ARC_CHALLENGE_LIMIT", "256")
    env_default("STAGE5_DEBIASED_USE_LEARNED_LOOP_CONTROL", "1")
    env_default("STAGE5_DEBIASED_ASSESS_REQUIRED_BENCHMARKS", "arc_challenge,open_hard_arc_challenge")
    env_default("STAGE5_DEBIASED_BENCHMARK_DISCONNECT", "0")
    env_default("STAGE5_BENCHMARK_ASSESS_ALLOWED_NEGATIVE_DELTA", "0")
    env_default("STAGE5_BENCHMARK_ASSESS_NEGATIVE_EVIDENCE_MIN_ABS_DELTA", "2")
    env_default("STAGE5_BENCHMARK_ASSESS_NEGATIVE_EVIDENCE_SIGN_TEST_P_THRESHOLD", "0.10")

    print("=== Stage 5: depth-signal benchmark ===", flush=True)
    exec((ROOT / "colab" / "STAGE5_DEBIASED_BENCHMARK_SUITE_CELL.py").read_text(encoding="utf-8"))
    assessment_summary = print_current_pointer("assessment_summary")
    print("depth_signal_confirmation_complete=true", flush=True)
    print("recovery_summary:", recovery_summary, flush=True)
    print("assessment_summary:", assessment_summary, flush=True)

finally:
    if os.environ.get("STAGE5_DEPTH_SIGNAL_CONFIRMATION_DISCONNECT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        print("Disconnecting Colab runtime to conserve credits.", flush=True)
        try:
            runtime.unassign()
        except Exception as exc:
            print("runtime.unassign failed:", repr(exc), flush=True)
    else:
        print("Leaving runtime attached because STAGE5_DEPTH_SIGNAL_CONFIRMATION_DISCONNECT=0.", flush=True)
