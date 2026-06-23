"""Colab cell: run the 7B capability ladder and build trace jobs.

This is a high-memory GPU target. It scores the ARC capability ladder with
Qwen 0.5B/1.5B/3B/7B, keeps the local scored rows alive, then immediately
builds provider-neutral trace-generation jobs from that result. By default it
does not call paid teacher APIs and does not train recurrent weights. If
explicit provider and SFT flags are set, it can continue through teacher trace
collection and bounded recurrent Phase 1 SFT in the same runtime.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_CELL_VERSION = "capability_ladder_7b_trace_chain_v1"
STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_TARGET = "capability_ladder_7b_trace_chain"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MIN_TRACE_ROWS_DEFAULT = 16


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


def redact(text):
    text = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            text = text.replace(token, "****")
    return text


def run(cmd, cwd=None, env=None, check=True):
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
        print("\n".join(stdout.splitlines()[-160:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(returncode, cmd, output=stdout)
    return subprocess.CompletedProcess(cmd, returncode, stdout, None)


def env_flag(name, default):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y"}


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


def current_source_summary():
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if not pointer.exists():
        return None
    raw = pointer.read_text(encoding="utf-8").strip()
    return resolve_repo_path(raw) if raw else None


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


def trace_collection_summary_paths():
    roots = [ROOT / "outputs" / "stage5"]
    drive_root = Path("/content/drive/MyDrive/recurrent-qwen-svgd/stage5_capability_ladder_trace_collection")
    if drive_root.exists():
        roots.append(drive_root)
    for root in roots:
        if root.exists():
            yield from sorted(root.glob("**/summary.json"), key=lambda item: item.stat().st_mtime, reverse=True)


def resolve_trace_collection_summary():
    explicit = (
        os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_SOURCE_SUMMARY")
        or os.environ.get("STAGE5_CURRENT_A100_SOURCE_SUMMARY")
        or ""
    ).strip()
    if explicit:
        path = resolve_repo_path(explicit)
        if not is_gate_ready_trace_collection(path):
            raise RuntimeError(f"Explicit source summary is not a gate-ready trace collection: {path}")
        return path

    pointer = current_source_summary()
    if pointer and is_gate_ready_trace_collection(pointer):
        return pointer

    for candidate in trace_collection_summary_paths():
        if is_gate_ready_trace_collection(candidate):
            return candidate
    raise RuntimeError("No gate-ready capability-ladder trace collection summary found for SFT.")


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


def derive_sft_env(summary_path):
    payload = read_json(summary_path)
    curriculum = payload.get("curriculum") if isinstance(payload.get("curriculum"), dict) else {}
    counts = curriculum.get("counts") if isinstance(curriculum.get("counts"), dict) else {}
    collection = payload.get("collection") if isinstance(payload.get("collection"), dict) else {}
    drive_backup = payload.get("drive_backup") if isinstance(payload.get("drive_backup"), dict) else {}

    work_dir = str(curriculum.get("work_dir") or "").replace("\\", "/")
    summary_json = str(curriculum.get("summary_json") or "").replace("\\", "/")
    if not work_dir or not summary_json:
        raise RuntimeError(f"Trace collection summary is missing curriculum work_dir/summary_json: {summary_path}")

    positive_rows = int(counts.get("positive_sft_rows") or counts.get("typed_records") or 0)
    min_trace_rows = int(os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_MIN_TRACE_ROWS", str(MIN_TRACE_ROWS_DEFAULT)))
    allow_tiny = env_flag("STAGE5_TRACED_CAPABILITY_SFT_ALLOW_TINY", "0")
    if positive_rows < min_trace_rows and not allow_tiny:
        raise RuntimeError(
            f"Trace collection has only {positive_rows} positive SFT rows; default floor is {min_trace_rows}. "
            "Collect more traces or set STAGE5_TRACED_CAPABILITY_SFT_ALLOW_TINY=1 for a deliberate smoke run."
        )

    target_loop_counts = collection.get("target_loop_counts")
    if not isinstance(target_loop_counts, dict):
        target_loop_counts = counts.get("target_loop_counts")
    max_loops = int(os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_MAX_LOOPS", str(int_dict_max_key(target_loop_counts, 4))))
    steps_default = min(150, max(50, positive_rows * 4))
    phase1_steps = int(os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_PHASE1_STEPS", str(steps_default)))
    min_mode_rows = os.environ.get(
        "STAGE5_TRACED_CAPABILITY_SFT_MIN_MODE_ROWS",
        mode_rows_from_counts(counts.get("mode_counts")),
    ).strip()

    env = os.environ.copy()
    env.update(
        {
            "MODEL_NAME": os.environ.get(
                "MODEL_NAME",
                os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_MODEL_NAME", DEFAULT_MODEL_NAME),
            ),
            "STAGE5_CURRICULUM_SFT_RUN_ID": os.environ.get("STAGE5_CURRICULUM_SFT_RUN_ID")
            or time.strftime("stage5_traced_capability_ladder_sft_%Y%m%d_%H%M%S"),
            "STAGE5_CURRICULUM_WORK_DIR": work_dir,
            "STAGE5_CURRICULUM_SUMMARY_JSON": summary_json,
            "STAGE5_CURRICULUM_MIN_POSITIVE_ROWS": str(positive_rows),
            "STAGE5_CURRICULUM_PHASE1_STEPS": str(phase1_steps),
            "STAGE5_CURRICULUM_MAX_LOOPS": str(max_loops),
            "DTYPE": os.environ.get("DTYPE", "bfloat16"),
            "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
            "DEVICE": os.environ.get("DEVICE", "cuda"),
            "STAGE5_CURRICULUM_SFT_PUSH": os.environ.get("STAGE5_CURRICULUM_SFT_PUSH", "1"),
        }
    )
    if min_mode_rows:
        env["STAGE5_CURRICULUM_MIN_MODE_ROWS"] = min_mode_rows
    drive_root = str(drive_backup.get("dest_root") or "").strip()
    if drive_root:
        env["STAGE5_CURRICULUM_INPUT_BACKUP_DIR"] = drive_root
    if env_flag("STAGE5_TRACED_CAPABILITY_SFT_ALLOW_NO_DRIVE_BACKUP", "0"):
        env["STAGE5_CURRICULUM_ALLOW_NO_DRIVE_BACKUP"] = "1"

    print(
        json.dumps(
            {
                "sft_source_summary": path_for_cli(summary_path),
                "positive_rows": positive_rows,
                "min_mode_rows": min_mode_rows,
                "max_loops": max_loops,
                "phase1_steps": phase1_steps,
                "model_name": env["MODEL_NAME"],
            },
            indent=2,
        ),
        flush=True,
    )
    return env


def prepare_provider_env(base_env):
    provider_env = base_env.copy()
    api_key_env = provider_env.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_API_KEY_ENV", "").strip()
    if not api_key_env:
        if secret("OPENAI_API_KEY"):
            api_key_env = "OPENAI_API_KEY"
        elif secret("OPENROUTER_API_KEY"):
            api_key_env = "OPENROUTER_API_KEY"
        else:
            api_key_env = "OPENAI_API_KEY"
    provider_env["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_API_KEY_ENV"] = api_key_env
    token = secret(api_key_env)
    if token:
        provider_env[api_key_env] = token
    if (
        api_key_env == "OPENROUTER_API_KEY"
        and "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BASE_URL" not in provider_env
    ):
        provider_env["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BASE_URL"] = "https://openrouter.ai/api/v1"

    provider_env.update(
        {
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER": "1",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BACKUP_DRIVE": "1" if backup_drive else "0",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_PUSH": "1",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COMMIT_RESPONSES": "1",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_GPU": "1",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_REFUSE_GPU": "0",
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_BACKUP_DRIVE": "1" if backup_drive else "0",
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_PUSH": "1",
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_ALLOW_GPU": "1",
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_REFUSE_GPU": "0",
        }
    )
    limit = os.environ.get("STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_TRACE_LIMIT", "").strip()
    if limit:
        provider_env["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT"] = limit
    return provider_env


print(
    f"STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_CELL_VERSION="
    f"{STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_CELL_VERSION}",
    flush=True,
)

run(["nvidia-smi"], cwd=Path("/content"))

backup_drive = env_flag("STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_BACKUP_DRIVE", "1")
if backup_drive and not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive", force_remount=True)

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

run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_stage5_capability_ladder_mcq_probe.py",
        "tests/test_merge_capability_score_rows.py",
        "tests/test_capability_ladder_curriculum.py",
        "tests/test_curriculum_sft_gate.py",
        "tests/test_capability_ladder_trace_jobs.py",
        "tests/test_stage5_next_plan.py::test_capability_ladder_mcq_probe_with_rows_recommends_trace_jobs_before_sft_gate",
    ],
    cwd=ROOT,
)

chain_arc_limit = os.environ.get("STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_ARC_LIMIT", "96")
chain_score_mode = os.environ.get("STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_SCORE_MODE", "content_question_only")

probe_env = os.environ.copy()
probe_env.update(
    {
        "STAGE5_CAPABILITY_LADDER_MODELS": (
            "qwen_0_5b=Qwen/Qwen2.5-0.5B-Instruct,"
            "qwen_1_5b=Qwen/Qwen2.5-1.5B-Instruct,"
            "qwen_3b=Qwen/Qwen2.5-3B-Instruct,"
            "qwen_7b=Qwen/Qwen2.5-7B-Instruct"
        ),
        "STAGE5_CAPABILITY_LADDER_MODEL_LADDER": (
            "qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3,qwen_7b:4"
        ),
        "STAGE5_CAPABILITY_LADDER_ARC_LIMIT": chain_arc_limit,
        "STAGE5_CAPABILITY_LADDER_SCORE_MODE": chain_score_mode,
        "STAGE5_CAPABILITY_LADDER_BACKUP_DRIVE": "1" if backup_drive else "0",
        "STAGE5_CAPABILITY_LADDER_PUSH": "1",
        "DTYPE": os.environ.get("DTYPE", "bfloat16"),
        "DEVICE": os.environ.get("DEVICE", "cuda"),
    }
)

print("=== 7B capability-ladder MCQ probe ===", flush=True)
run([sys.executable, "colab/run_stage5_capability_ladder_mcq_probe.py"], cwd=ROOT, env=probe_env)

print("=== Sync after probe commit ===", flush=True)
run(["git", "pull", "--rebase", "origin", "main"], cwd=ROOT)

trace_env = os.environ.copy()
trace_env.update(
    {
        "STAGE5_CAPABILITY_LADDER_TRACE_ALLOW_GPU": "1",
        "STAGE5_CAPABILITY_LADDER_TRACE_REFUSE_GPU": "0",
        "STAGE5_CAPABILITY_LADDER_TRACE_BACKUP_DRIVE": "1" if backup_drive else "0",
        "STAGE5_CAPABILITY_LADDER_TRACE_PUSH": "1",
    }
)

print("=== Build capability-ladder trace jobs ===", flush=True)
run([sys.executable, "colab/run_stage5_capability_ladder_trace_jobs.py"], cwd=ROOT, env=trace_env)

run_provider = env_flag("STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_RUN_PROVIDER", "0")
run_sft_default = "1" if run_provider else "0"
run_sft = env_flag("STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_RUN_SFT", run_sft_default)
if run_provider:
    print("=== Provider trace responses and trace collection ===", flush=True)
    provider_env = prepare_provider_env(os.environ.copy())
    run([sys.executable, "colab/run_stage5_capability_ladder_trace_responses.py"], cwd=ROOT, env=provider_env)
    run(["git", "pull", "--rebase", "origin", "main"], cwd=ROOT)
    run([sys.executable, "colab/run_stage5_capability_ladder_trace_collect.py"], cwd=ROOT, env=provider_env)
elif run_sft:
    print(
        "Skipping provider response generation because "
        "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_RUN_PROVIDER is not enabled. "
        "SFT will use an existing gate-ready trace collection if one is available.",
        flush=True,
    )

if run_sft:
    print("=== Bounded traced capability-ladder Phase 1 SFT ===", flush=True)
    trace_collection = resolve_trace_collection_summary()
    sft_env = derive_sft_env(trace_collection)
    run([sys.executable, "colab/run_stage5_curriculum_sft.py"], cwd=ROOT, env=sft_env)
else:
    print(
        "Trace jobs are ready. To continue in the same runtime, rerun with "
        "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_RUN_PROVIDER=1 plus a provider "
        "model override/map, or run target=capability_ladder_trace_response_collect_cpu.",
        flush=True,
    )

summary_pointer = ROOT / "config" / "stage5_current_source_summary.txt"
print("current_source_summary:", summary_pointer.read_text(encoding="utf-8").strip(), flush=True)
run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)

if env_flag("STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_DISCONNECT", "1"):
    print("Disconnecting Colab runtime to conserve credits.", flush=True)
    runtime.unassign()
else:
    print("Leaving Colab runtime connected for follow-up inspection.", flush=True)
