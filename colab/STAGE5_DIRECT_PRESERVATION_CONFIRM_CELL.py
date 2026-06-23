"""Colab cell: confirm direct-route preservation on larger ARC slices.

This intentionally bypasses the generic planner. It consumes the latest
``stage5_direct_preservation_probe`` summary, resolves that summary's selected
checkpoint, then runs a no-training loop-1 benchmark on ARC-Easy and
ARC-Challenge. The benchmark runner publishes the result and updates
``config/stage5_current_source_summary.txt``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_DIRECT_PRESERVATION_CONFIRM_CELL_VERSION = "direct_preservation_confirm_v2"

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DEFAULT_SOURCE_SUMMARY = "outputs/stage5/stage5_direct_preservation_loop1_20260622_232720/summary.json"


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


DISCONNECT_WHEN_DONE = env_bool("STAGE5_DIRECT_CONFIRM_DISCONNECT", False)
ALLOW_UNPASSED_SOURCE = env_bool("STAGE5_DIRECT_CONFIRM_ALLOW_UNPASSED_SOURCE", False)


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


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True):
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
        run(["git", "fetch", "origin", "main"], cwd=ROOT)
        run(["git", "checkout", "main"], cwd=ROOT)
        run(["git", "reset", "--hard", "origin/main"], cwd=ROOT)
    else:
        run(["git", "clone", clone_url, str(ROOT)])
    run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
    run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pointer_value(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def candidate_source_summaries() -> list[str]:
    candidates = [
        os.environ.get("STAGE5_DIRECT_CONFIRM_SOURCE_SUMMARY", "").strip(),
        os.environ.get("STAGE5_BENCHMARK_SOURCE_SUMMARY", "").strip(),
        pointer_value(ROOT / "config" / "stage5_latest_direct_preservation_summary.txt"),
        pointer_value(ROOT / "config" / "stage5_current_source_summary.txt"),
        DEFAULT_SOURCE_SUMMARY,
    ]
    return [candidate for candidate in candidates if candidate]


def selected_source_summary() -> Path:
    checked: list[str] = []
    for candidate in candidate_source_summaries():
        path = resolve_path(candidate)
        checked.append(path_for_cli(path))
        if not path.exists():
            continue
        payload = read_json(path)
        if payload.get("kind") == "stage5_direct_preservation_probe":
            return path
    raise FileNotFoundError(
        "No stage5_direct_preservation_probe summary found. Checked: " + ", ".join(checked)
    )


def checkpoint_from_probe(payload: dict) -> str | None:
    best = payload.get("best_checkpoint")
    if isinstance(best, dict) and best.get("checkpoint"):
        return str(best["checkpoint"])
    for key in ("checkpoint", "phase1_checkpoint", "resume_checkpoint"):
        if payload.get(key):
            return str(payload[key])
    return None


def disconnect(reason: str) -> None:
    if not DISCONNECT_WHEN_DONE:
        return
    try:
        print(f"Disconnecting Colab runtime to conserve credits: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped: {exc}", flush=True)


try:
    print(
        f"STAGE5_DIRECT_PRESERVATION_CONFIRM_CELL_VERSION={STAGE5_DIRECT_PRESERVATION_CONFIRM_CELL_VERSION}",
        flush=True,
    )
    run(["nvidia-smi"], cwd=Path("/content"), check=False)
    sync_repo()
    os.chdir(ROOT)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

    source = selected_source_summary()
    source_payload = read_json(source)
    checkpoint = checkpoint_from_probe(source_payload)
    assert checkpoint, f"Direct-preservation source has no checkpoint: {path_for_cli(source)}"
    if not ALLOW_UNPASSED_SOURCE:
        assert source_payload.get("passed") is True, (
            "Direct-preservation source did not pass; refusing confirmation. "
            f"source={path_for_cli(source)} status={source_payload.get('status')}"
        )

    run_id = os.environ.get("STAGE5_DIRECT_CONFIRM_RUN_ID") or time.strftime(
        "stage5_direct_preservation_confirm_loop1_%Y%m%d_%H%M%S"
    )
    env = os.environ.copy()
    env.update(
        {
            "STAGE5_BENCHMARK_SUITE_RUN_ID": run_id,
            "STAGE5_BENCHMARK_SOURCE_SUMMARY": path_for_cli(source),
            "STAGE5_BENCHMARK_CHECKPOINT": checkpoint,
            "STAGE5_BENCHMARKS": "arc_easy,arc_challenge",
            "STAGE5_BENCHMARK_ARC_EASY_LIMIT": os.environ.get("STAGE5_DIRECT_CONFIRM_ARC_EASY_LIMIT", "256"),
            "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": os.environ.get(
                "STAGE5_DIRECT_CONFIRM_ARC_CHALLENGE_LIMIT", "256"
            ),
            "STAGE5_BENCHMARK_MAX_LOOPS": "1",
            "STAGE5_BENCHMARK_NUM_TRAJECTORIES": "1",
            "STAGE5_BENCHMARK_SCORE_TARGETS": os.environ.get(
                "STAGE5_DIRECT_CONFIRM_SCORE_TARGETS",
                "content_question_only,cyclic_label_aggregated",
            ),
            "STAGE5_BENCHMARK_AGGREGATES": "mean",
            "STAGE5_BENCHMARK_INCLUDE_LOOP_DIAGNOSTICS": "1",
            "STAGE5_BENCHMARK_PUSH": "1",
            "DTYPE": os.environ.get("DTYPE", "bfloat16"),
            "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
            "DEVICE": os.environ.get("DEVICE", "cuda"),
        }
    )
    print("direct_preservation_confirmation_run_id:", run_id, flush=True)
    print("direct_preservation_confirmation_source:", path_for_cli(source), flush=True)
    print("direct_preservation_confirmation_checkpoint:", checkpoint, flush=True)
    print("recommended_runtime: T4/L4/G4 is sufficient; A100 is not required.", flush=True)
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], cwd=ROOT, env=env)

    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if pointer.exists():
        print("current_source_summary:", pointer.read_text(encoding="utf-8").strip(), flush=True)
    disconnect("direct preservation confirmation finished")
except Exception:
    disconnect("direct preservation confirmation errored")
    raise
