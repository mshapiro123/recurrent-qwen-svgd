"""Colab cell: resume depth-router SFT from a passed direct-preservation checkpoint.

This target is deliberately narrower than the local-HF trace/SFT chain. It
does not generate more 7B traces. It reuses the latest gate-ready
capability-ladder trace curriculum, starts from the latest passed
direct-preservation checkpoint, trains only the lightweight halting/router
parameters by default, then runs a bounded ARC sanity benchmark.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_CELL_VERSION = "depth_router_after_direct_preserve_v1"
STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_TARGET = "traced_sft_depth_router_after_direct_preserve"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DEFAULT_TRACE_COLLECTION = (
    "outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json"
)


def secret(*names):
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


def env_flag(name, default):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def redact(text):
    text = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            text = text.replace(token, "****")
    return text


def run(cmd, *, cwd=None, env=None, check=True):
    printable = redact(" ".join(map(str, cmd)))
    print("$", printable, flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd or ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        chunks.append(safe)
    stdout = "".join(chunks)
    returncode = process.wait()
    if check and returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(stdout.splitlines()[-180:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(returncode, cmd, output=stdout)
    return subprocess.CompletedProcess(cmd, returncode, stdout, None)


def read_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def resolve_repo_path(value):
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path):
    try:
        return Path(path).relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def current_pointer_path():
    return ROOT / "config" / "stage5_current_source_summary.txt"


def latest_direct_pointer_path():
    return ROOT / "config" / "stage5_latest_direct_preservation_summary.txt"


def is_gate_ready_trace_collection(path):
    try:
        payload = read_json(path)
    except Exception:
        return False
    return (
        payload.get("kind") == "stage5_capability_ladder_trace_collection"
        and payload.get("status") == "trace_curriculum_gate_ready"
        and isinstance(payload.get("gate"), dict)
        and payload["gate"].get("go") is True
    )


def resolve_trace_collection_summary():
    explicit = (
        os.environ.get("STAGE5_DEPTH_ROUTER_TRACE_SOURCE_SUMMARY")
        or os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_SOURCE_SUMMARY")
        or ""
    ).strip()
    candidates = []
    if explicit:
        candidates.append(resolve_repo_path(explicit))
    candidates.append(resolve_repo_path(DEFAULT_TRACE_COLLECTION))
    root = ROOT / "outputs" / "stage5"
    if root.exists():
        candidates.extend(
            sorted(root.glob("**/summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        )
    for candidate in candidates:
        if candidate.exists() and is_gate_ready_trace_collection(candidate):
            return candidate
    raise RuntimeError("No gate-ready capability-ladder trace collection summary found.")


def direct_preservation_checkpoint(payload):
    best = payload.get("best_checkpoint")
    if isinstance(best, dict) and best.get("checkpoint"):
        return str(best["checkpoint"])
    for key in ("checkpoint", "phase1_checkpoint", "resume_checkpoint"):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def is_passed_direct_preservation(path):
    try:
        payload = read_json(path)
    except Exception:
        return False
    return (
        payload.get("kind") == "stage5_direct_preservation_probe"
        and payload.get("passed") is True
        and bool(direct_preservation_checkpoint(payload))
    )


def resolve_direct_preservation_summary():
    explicit = os.environ.get("STAGE5_DEPTH_ROUTER_DIRECT_SOURCE_SUMMARY", "").strip()
    candidates = []
    if explicit:
        path = resolve_repo_path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Explicit direct-preservation summary is missing: {path}")
        if not is_passed_direct_preservation(path):
            raise RuntimeError(f"Explicit direct-preservation summary has not passed: {path}")
        return path
    for pointer in (latest_direct_pointer_path(), current_pointer_path()):
        if pointer.exists():
            raw = pointer.read_text(encoding="utf-8").strip()
            if raw:
                candidates.append(resolve_repo_path(raw))
    root = ROOT / "outputs" / "stage5"
    if root.exists():
        candidates.extend(
            sorted(root.glob("**/summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        )
    for candidate in candidates:
        if candidate.exists() and is_passed_direct_preservation(candidate):
            return candidate
    raise RuntimeError(
        "No passed direct-preservation summary found. Run traced_sft_direct_preservation_probe first."
    )


def int_dict_max_key(payload, default):
    values = []
    if isinstance(payload, dict):
        for key in payload:
            try:
                values.append(int(key))
            except (TypeError, ValueError):
                pass
    return max(values) if values else default


def mode_rows_from_counts(mode_counts):
    if not isinstance(mode_counts, dict):
        return ""
    parts = []
    for mode, count in sorted(mode_counts.items()):
        try:
            n = int(count)
        except (TypeError, ValueError):
            continue
        if n > 0:
            parts.append(f"{mode}={n}")
    return ",".join(parts)


def derive_sft_env(trace_summary, direct_summary):
    trace_payload = read_json(trace_summary)
    direct_payload = read_json(direct_summary)
    curriculum = trace_payload.get("curriculum") if isinstance(trace_payload.get("curriculum"), dict) else {}
    counts = curriculum.get("counts") if isinstance(curriculum.get("counts"), dict) else {}
    collection = trace_payload.get("collection") if isinstance(trace_payload.get("collection"), dict) else {}
    work_dir = str(curriculum.get("work_dir") or "").replace("\\", "/")
    summary_json = str(curriculum.get("summary_json") or "").replace("\\", "/")
    if not work_dir or not summary_json:
        raise RuntimeError(f"Trace collection summary is missing curriculum work_dir/summary_json: {trace_summary}")

    positive_rows = int(counts.get("positive_sft_rows") or counts.get("typed_records") or 0)
    min_trace_rows = int(os.environ.get("STAGE5_DEPTH_ROUTER_MIN_TRACE_ROWS", "48"))
    if positive_rows < min_trace_rows:
        raise RuntimeError(f"Trace collection has only {positive_rows} rows; need at least {min_trace_rows}.")

    checkpoint = resolve_repo_path(direct_preservation_checkpoint(direct_payload))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing direct-preservation checkpoint: {checkpoint}")

    target_loop_counts = collection.get("target_loop_counts")
    if not isinstance(target_loop_counts, dict):
        target_loop_counts = counts.get("target_loop_counts")
    max_loops = int(os.environ.get("STAGE5_DEPTH_ROUTER_MAX_LOOPS", str(int_dict_max_key(target_loop_counts, 3))))
    run_id = os.environ.get("STAGE5_DEPTH_ROUTER_RUN_ID") or time.strftime(
        "stage5_depth_router_after_direct_preserve_%Y%m%d_%H%M%S"
    )
    env = os.environ.copy()
    env.update(
        {
            "MODEL_NAME": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
            "STAGE5_CURRICULUM_SFT_RUN_ID": run_id,
            "STAGE5_CURRICULUM_WORK_DIR": work_dir,
            "STAGE5_CURRICULUM_SUMMARY_JSON": summary_json,
            "STAGE5_CURRICULUM_RESUME_FROM": path_for_cli(checkpoint),
            "STAGE5_CURRICULUM_MIN_POSITIVE_ROWS": str(positive_rows),
            "STAGE5_CURRICULUM_MIN_MODE_ROWS": os.environ.get(
                "STAGE5_DEPTH_ROUTER_MIN_MODE_ROWS",
                mode_rows_from_counts(counts.get("mode_counts")),
            ),
            "STAGE5_CURRICULUM_MAX_LOOPS": str(max_loops),
            "STAGE5_CURRICULUM_PHASE1_STEPS": os.environ.get("STAGE5_DEPTH_ROUTER_STEPS", "100"),
            "STAGE5_CURRICULUM_PHASE1_LR": os.environ.get("STAGE5_DEPTH_ROUTER_LR", "5e-5"),
            "STAGE5_CURRICULUM_PHASE1_BETA": os.environ.get("STAGE5_DEPTH_ROUTER_BETA", "0.12"),
            "STAGE5_CURRICULUM_HALT_TARGET_NLL_WEIGHT": os.environ.get(
                "STAGE5_DEPTH_ROUTER_HALT_TARGET_NLL_WEIGHT",
                "5.0",
            ),
            "STAGE5_CURRICULUM_USE_LEARNED_LOOP_CONTROL": "1",
            "STAGE5_CURRICULUM_LOOP_CONTROL_CE_WEIGHT": os.environ.get(
                "STAGE5_DEPTH_ROUTER_LOOP_CONTROL_CE_WEIGHT",
                "4.0",
            ),
            "STAGE5_CURRICULUM_OPTIMIZER_MODULES": os.environ.get(
                "STAGE5_DEPTH_ROUTER_OPTIMIZER_MODULES",
                "halt",
            ),
            "STAGE5_CURRICULUM_DEPTH_HINT_STYLE": os.environ.get(
                "STAGE5_DEPTH_ROUTER_DEPTH_HINT_STYLE",
                "natural",
            ),
            "STAGE5_CURRICULUM_ALLOW_ANSWER_LINE_VERIFICATION": "1",
            "STAGE5_CURRICULUM_ALLOW_NO_DRIVE_BACKUP": "1",
            "STAGE5_CURRICULUM_SFT_COMMIT_CHECKPOINTS": os.environ.get(
                "STAGE5_DEPTH_ROUTER_COMMIT_CHECKPOINTS",
                "0",
            ),
            "STAGE5_CURRICULUM_SFT_PUSH": "1",
            "DTYPE": os.environ.get("DTYPE", "bfloat16"),
            "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
            "DEVICE": os.environ.get("DEVICE", "cuda"),
        }
    )
    print(
        json.dumps(
            {
                "trace_summary": path_for_cli(trace_summary),
                "direct_summary": path_for_cli(direct_summary),
                "resume_checkpoint": path_for_cli(checkpoint),
                "positive_rows": positive_rows,
                "target_loop_counts": target_loop_counts,
                "max_loops": max_loops,
                "run_id": run_id,
                "optimizer_modules": env["STAGE5_CURRICULUM_OPTIMIZER_MODULES"],
                "loop_control_ce_weight": env["STAGE5_CURRICULUM_LOOP_CONTROL_CE_WEIGHT"],
                "halt_target_nll_weight": env["STAGE5_CURRICULUM_HALT_TARGET_NLL_WEIGHT"],
            },
            indent=2,
        ),
        flush=True,
    )
    return env


print(
    "STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_CELL_VERSION="
    f"{STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_CELL_VERSION}",
    flush=True,
)

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
run(["nvidia-smi"], cwd=Path("/content"), check=False)

run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_stage5_curriculum_sft.py",
        "tests/test_curriculum_sft_gate.py",
        "tests/test_stage5_benchmark_suite.py",
        "tests/test_stage5_traced_sft_assessment.py",
    ],
    cwd=ROOT,
)

trace_summary = resolve_trace_collection_summary()
direct_summary = resolve_direct_preservation_summary()
sft_env = derive_sft_env(trace_summary, direct_summary)
run([sys.executable, "colab/run_stage5_curriculum_sft.py"], cwd=ROOT, env=sft_env)

if env_flag("STAGE5_DEPTH_ROUTER_RUN_BENCHMARK", "1"):
    pointer = current_pointer_path()
    source_summary = pointer.read_text(encoding="utf-8").strip()
    benchmark_env = os.environ.copy()
    benchmark_env.update(
        {
            "STAGE5_BENCHMARK_SOURCE_SUMMARY": source_summary,
            "STAGE5_BENCHMARK_SUITE_RUN_ID": os.environ.get("STAGE5_DEPTH_ROUTER_BENCHMARK_RUN_ID")
            or time.strftime("stage5_depth_router_after_direct_preserve_benchmark_%Y%m%d_%H%M%S"),
            "STAGE5_BENCHMARKS": os.environ.get("STAGE5_DEPTH_ROUTER_BENCHMARKS", "arc_easy,arc_challenge"),
            "STAGE5_BENCHMARK_ARC_EASY_LIMIT": os.environ.get("STAGE5_DEPTH_ROUTER_ARC_EASY_LIMIT", "128"),
            "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": os.environ.get(
                "STAGE5_DEPTH_ROUTER_ARC_CHALLENGE_LIMIT",
                "128",
            ),
            "STAGE5_BENCHMARK_SCORE_TARGETS": os.environ.get(
                "STAGE5_DEPTH_ROUTER_SCORE_TARGETS",
                "content_question_only,cyclic_label_aggregated",
            ),
            "STAGE5_BENCHMARK_AGGREGATES": "mean",
            "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL": "1",
            "STAGE5_BENCHMARK_PUSH": "1",
            "DTYPE": os.environ.get("DTYPE", "bfloat16"),
            "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
            "DEVICE": os.environ.get("DEVICE", "cuda"),
        }
    )
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], cwd=ROOT, env=benchmark_env)
    if env_flag("STAGE5_DEPTH_ROUTER_RUN_ASSESSMENT", "1"):
        assessment_source = current_pointer_path().read_text(encoding="utf-8").strip()
        assessment_env = os.environ.copy()
        assessment_env["STAGE5_TRACED_SFT_ASSESS_PUSH"] = "1"
        run(
            [sys.executable, "colab/assess_stage5_traced_sft.py", "--summary_json", assessment_source],
            cwd=ROOT,
            env=assessment_env,
        )

print("current_source_summary:", current_pointer_path().read_text(encoding="utf-8").strip(), flush=True)
run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)

if env_flag("STAGE5_DEPTH_ROUTER_DISCONNECT", "1"):
    print("Disconnecting Colab runtime to conserve credits.", flush=True)
    runtime.unassign()
