"""Colab cell: benchmark the latest traced capability-ladder SFT checkpoint.

This is the restart-safe follow-up for a local-HF trace/SFT chain that already
finished training but did not finish or publish the post-SFT benchmark. It does
not regenerate traces and does not rerun SFT.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_TRACED_SFT_BENCHMARK_CELL_VERSION = "traced_sft_benchmark_v1"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DEFAULT_SOURCE_SUMMARY = (
    "outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_194543/summary.json"
)


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


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


def redact(text: str) -> str:
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            text = text.replace(token, "****")
    return text


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd or ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        chunks.append(safe)
    returncode = process.wait()
    if returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join("".join(chunks).splitlines()[-180:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(returncode, cmd, output="".join(chunks))


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def source_summary() -> str:
    explicit = os.environ.get("STAGE5_TRACED_SFT_BENCHMARK_SOURCE_SUMMARY", "").strip()
    if explicit:
        return path_for_cli(resolve_path(explicit))
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if pointer.exists():
        raw = pointer.read_text(encoding="utf-8").strip()
        if raw and resolve_path(raw).exists():
            return path_for_cli(resolve_path(raw))
    return DEFAULT_SOURCE_SUMMARY


print(
    f"STAGE5_TRACED_SFT_BENCHMARK_CELL_VERSION={STAGE5_TRACED_SFT_BENCHMARK_CELL_VERSION}",
    flush=True,
)
try:
    run(["nvidia-smi"], cwd=Path("/content"))

    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
        run(["git", "fetch", "origin", "main"], cwd=ROOT)
        run(["git", "checkout", "main"], cwd=ROOT)
        run(["git", "reset", "--hard", "origin/main"], cwd=ROOT)
    else:
        run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))

    run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
    run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

    if HF_TOKEN:
        from huggingface_hub import HfApi, login

        login(token=HF_TOKEN, add_to_git_credential=False)
        who = HfApi(token=HF_TOKEN).whoami()
        print("HF auth OK:", who.get("name") or who.get("email") or "authenticated user", flush=True)
    else:
        print("HF auth skipped; downloads will use anonymous Hub access.", flush=True)

    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage5_benchmark_suite.py",
            "tests/test_stage5_traced_sft_assessment.py",
        ],
        cwd=ROOT,
    )

    selected_source = source_summary()
    print("traced_sft_benchmark_source_summary:", selected_source, flush=True)
    assert resolve_path(selected_source).exists(), f"Missing source summary: {selected_source}"

    benchmark_env = os.environ.copy()
    benchmark_env.update(
        {
            "STAGE5_BENCHMARK_SOURCE_SUMMARY": selected_source,
            "STAGE5_BENCHMARK_SUITE_RUN_ID": os.environ.get("STAGE5_BENCHMARK_SUITE_RUN_ID")
            or time.strftime("stage5_local_hf_traced_sft_scale64_benchmark_%Y%m%d_%H%M%S"),
            "STAGE5_BENCHMARKS": os.environ.get(
                "STAGE5_TRACED_SFT_BENCHMARKS",
                "arc_easy,arc_challenge",
            ),
            "STAGE5_BENCHMARK_ARC_EASY_LIMIT": os.environ.get(
                "STAGE5_TRACED_SFT_BENCHMARK_ARC_EASY_LIMIT",
                "128",
            ),
            "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": os.environ.get(
                "STAGE5_TRACED_SFT_BENCHMARK_ARC_CHALLENGE_LIMIT",
                "128",
            ),
            "STAGE5_BENCHMARK_SCORE_TARGETS": os.environ.get(
                "STAGE5_TRACED_SFT_BENCHMARK_SCORE_TARGETS",
                "content_question_only,cyclic_label_aggregated",
            ),
            "STAGE5_BENCHMARK_AGGREGATES": os.environ.get(
                "STAGE5_TRACED_SFT_BENCHMARK_AGGREGATES",
                "mean",
            ),
            "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL": "1",
            "STAGE5_BENCHMARK_PUSH": "1",
            "MODEL_NAME": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
            "DTYPE": os.environ.get("DTYPE", "bfloat16"),
            "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
            "DEVICE": os.environ.get("DEVICE", "cuda"),
        }
    )
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], cwd=ROOT, env=benchmark_env)

    benchmark_summary = (ROOT / "config" / "stage5_current_source_summary.txt").read_text(
        encoding="utf-8"
    ).strip()
    print("benchmark_summary:", benchmark_summary, flush=True)

    assessment_env = os.environ.copy()
    assessment_env.update(
        {
            "STAGE5_TRACED_SFT_ASSESS_RUN_ID": os.environ.get("STAGE5_TRACED_SFT_ASSESS_RUN_ID")
            or time.strftime("stage5_local_hf_traced_sft_scale64_assessment_%Y%m%d_%H%M%S"),
            "STAGE5_TRACED_SFT_ASSESS_PUSH": "1",
        }
    )
    run(
        [sys.executable, "colab/assess_stage5_traced_sft.py", "--summary_json", benchmark_summary],
        cwd=ROOT,
        env=assessment_env,
    )

    assessment_summary = (ROOT / "config" / "stage5_current_source_summary.txt").read_text(
        encoding="utf-8"
    ).strip()
    print("assessment_summary:", assessment_summary, flush=True)
    assessment_md = resolve_path(assessment_summary.replace("summary.json", "summary.md"))
    if assessment_md.exists():
        print(assessment_md.read_text(encoding="utf-8"), flush=True)
finally:
    if os.environ.get("STAGE5_TRACED_SFT_BENCHMARK_DISCONNECT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        print("Disconnecting Colab runtime to conserve credits.", flush=True)
        runtime.unassign()
