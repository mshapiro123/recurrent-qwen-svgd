"""Colab cell: run a bounded debiased Stage 5 benchmark suite.

This is a measurement-only GPU action. It compares base Qwen against the
current recurrent checkpoint with bare-label, content-question-only, and
cyclic-label-aggregated MCQ scoring, then assesses the cyclic aggregate.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_DEBIASED_BENCHMARK_SUITE_CELL_VERSION = "debiased_benchmark_suite_v1"
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


def mask(text: str, token: str | None) -> str:
    return text.replace(token, "****") if token else text


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    shown = mask(" ".join(map(str, cmd)), GH_TOKEN)
    print("$", shown, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = mask(proc.stdout or "", GH_TOKEN)
    if output:
        print(output, flush=True)
    if proc.returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(output.splitlines()[-160:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def current_source_summary() -> Path:
    override = os.environ.get("STAGE5_DEBIASED_BENCHMARK_SOURCE_SUMMARY", "").strip()
    if override:
        return resolve_path(override)
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    value = pointer.read_text(encoding="utf-8").strip()
    assert value, "config/stage5_current_source_summary.txt is empty."
    return resolve_path(value)


def payload_has_checkpoint(payload: dict) -> bool:
    for key in ("checkpoint", "phase1_checkpoint", "final_checkpoint", "tuned_checkpoint", "resume_checkpoint"):
        if payload.get(key):
            return True
    for key_path in (
        ("metadata", "checkpoint"),
        ("metadata", "recovered_checkpoint"),
        ("compact", "final_checkpoint"),
        ("autopilot_compact", "final_checkpoint"),
        ("selected_checkpoint", "checkpoint"),
    ):
        cursor = payload
        for key in key_path:
            if not isinstance(cursor, dict):
                cursor = {}
                break
            cursor = cursor.get(key)
        if cursor:
            return True
    return False


def benchmark_source_summary(start: Path) -> Path:
    """Follow scoring-policy/debias wrappers to the checkpoint-bearing summary."""

    seen: set[Path] = set()
    path = start
    for _depth in range(8):
        path = resolve_path(path)
        if path in seen:
            raise RuntimeError(f"Cycle while resolving benchmark source summary: {path}")
        seen.add(path)
        payload = read_json(path)
        kind = payload.get("kind")
        if payload_has_checkpoint(payload):
            return path
        if kind == "stage5_mcq_scoring_policy" and payload.get("source_summary"):
            path = resolve_path(payload["source_summary"])
            continue
        if kind == "stage5_mcq_debias_pair_assessment":
            source_summaries = payload.get("source_summaries") or {}
            next_summary = source_summaries.get("arc_challenge") or source_summaries.get("arc_easy")
            if next_summary:
                path = resolve_path(next_summary)
                continue
        if kind == "stage5_mcq_debias_diagnostic":
            next_summary = payload.get("nested_source_summary") or payload.get("source_summary")
            if next_summary:
                path = resolve_path(next_summary)
                continue
        return path
    raise RuntimeError(f"Could not resolve checkpoint-bearing source summary from {start}")


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN/GITHUB_TOKEN in Colab secrets."

HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

try:
    gpu_check = shutil.which("nvidia-smi")
    assert gpu_check, "Attach an A100/H100/L4/T4 GPU runtime before running this benchmark action."
    run(["nvidia-smi"], cwd=Path("/content"))

    drive.mount("/content/drive", force_remount=False)
    authed = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", authed])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", authed, str(ROOT)], cwd=Path("/content"))
        run(["git", "remote", "set-url", "origin", authed])

    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage5_benchmark_suite.py",
            "tests/test_stage5_benchmark_assessment.py",
            "tests/test_mcq_debias.py",
        ]
    )

    current_summary = current_source_summary()
    source_summary = benchmark_source_summary(current_summary)
    source_payload = read_json(source_summary)
    adjacent_adapter = source_summary.parent / "recurrent_adapter_checkpoint.pt"
    assert payload_has_checkpoint(source_payload) or adjacent_adapter.exists(), (
        "Resolved benchmark source summary does not expose a checkpoint path "
        f"and has no adjacent adapter checkpoint: {path_for_cli(source_summary)}"
    )
    print("current_summary:", path_for_cli(current_summary), flush=True)
    print("benchmark_source_summary:", path_for_cli(source_summary), flush=True)

    env = os.environ.copy()
    env.setdefault(
        "STAGE5_BENCHMARK_SUITE_RUN_ID",
        "stage5_debiased_benchmark_suite_" + time.strftime("%Y%m%d_%H%M%S"),
    )
    env["STAGE5_BENCHMARK_SOURCE_SUMMARY"] = path_for_cli(source_summary)
    env["STAGE5_BENCHMARKS"] = os.environ.get("STAGE5_DEBIASED_BENCHMARKS", "arc_challenge,gpqa_lite")
    env["STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT"] = os.environ.get("STAGE5_DEBIASED_ARC_CHALLENGE_LIMIT", "128")
    env["STAGE5_BENCHMARK_ARC_EASY_LIMIT"] = os.environ.get("STAGE5_DEBIASED_ARC_EASY_LIMIT", "128")
    env["STAGE5_BENCHMARK_GPQA_LIMIT"] = os.environ.get("STAGE5_DEBIASED_GPQA_LIMIT", "16")
    env["STAGE5_BENCHMARK_SCORE_TARGETS"] = "label,content_question_only,cyclic_label_aggregated"
    env["STAGE5_BENCHMARK_AGGREGATES"] = "mean"
    env["STAGE5_BENCHMARK_PUSH"] = "1"
    env["MODEL_NAME"] = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    env["DTYPE"] = os.environ.get("DTYPE", "bfloat16")
    env["ADAPTER_DTYPE"] = os.environ.get("ADAPTER_DTYPE", "float32")
    env["DEVICE"] = os.environ.get("DEVICE", "cuda")
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], env=env)

    benchmark_summary = (ROOT / "config" / "stage5_current_source_summary.txt").read_text(encoding="utf-8").strip()
    print("benchmark_summary:", benchmark_summary, flush=True)

    assess_env = os.environ.copy()
    assess_env.setdefault(
        "STAGE5_BENCHMARK_ASSESS_RUN_ID",
        "stage5_debiased_benchmark_assessment_" + time.strftime("%Y%m%d_%H%M%S"),
    )
    assess_env["STAGE5_BENCHMARK_ASSESS_SCORE_TARGET"] = "cyclic_label_aggregated"
    assess_env["STAGE5_BENCHMARK_ASSESS_AGGREGATE"] = "permutation_mean"
    assess_env["STAGE5_BENCHMARK_ASSESS_MIN_ARC_EXAMPLES"] = env["STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT"]
    assess_env["STAGE5_BENCHMARK_ASSESS_MIN_GPQA_EXAMPLES"] = env["STAGE5_BENCHMARK_GPQA_LIMIT"]
    assess_env["STAGE5_BENCHMARK_ASSESS_PUSH"] = "1"
    run([sys.executable, "colab/assess_stage5_benchmark_suite.py", "--summary_json", benchmark_summary], env=assess_env)

    assessment_summary = (ROOT / "config" / "stage5_current_source_summary.txt").read_text(encoding="utf-8").strip()
    print("assessment_summary:", assessment_summary, flush=True)
    assessment_md = ROOT / assessment_summary.replace("summary.json", "summary.md")
    if assessment_md.exists():
        print(assessment_md.read_text(encoding="utf-8"), flush=True)

finally:
    print("Disconnecting Colab runtime to conserve credits.", flush=True)
    try:
        runtime.unassign()
    except Exception as exc:
        print("runtime.unassign failed:", repr(exc), flush=True)
