"""Colab cell: local-HF capability-ladder traces followed by bounded SFT.

This high-memory GPU target follows the current capability-ladder trace-job
summary, generates a bounded set of local Qwen 7B traces, collects verified
trace rows, and immediately starts deterministic recurrent Phase 1 SFT if the
trace collection gate passes. It is intended for overnight A100/H100/G4 runs
where avoiding another manual Colab handoff is worth more than splitting the
steps into separate cells.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_CELL_VERSION = "local_hf_trace_sft_chain_v1"
STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_TARGET = "capability_ladder_local_hf_trace_sft"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DEFAULT_TRACE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
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
        print("\n".join(stdout.splitlines()[-180:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(returncode, cmd, output=stdout)
    return subprocess.CompletedProcess(cmd, returncode, stdout, None)


def env_flag(name, default):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y"}


def attached_gpu_memory_mb():
    proc = run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        cwd=Path("/content"),
        check=False,
    )
    gpus = []
    if proc.returncode:
        return gpus
    for line in proc.stdout.splitlines():
        if not line.strip() or "," not in line:
            continue
        name, raw_memory = line.rsplit(",", 1)
        try:
            memory_mb = int(raw_memory.strip())
        except ValueError:
            continue
        gpus.append({"name": name.strip(), "memory_mb": memory_mb})
    return gpus


def require_enough_vram_for_local_hf():
    if env_flag("STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_SKIP_VRAM_CHECK", "0"):
        print("VRAM preflight skipped by override.", flush=True)
        return
    min_vram_mb = int(os.environ.get("STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_MIN_VRAM_MB", "20000"))
    gpus = attached_gpu_memory_mb()
    if not gpus:
        raise RuntimeError("No visible NVIDIA GPU for local-HF trace generation.")
    best = max(gpus, key=lambda item: int(item["memory_mb"]))
    print({"local_hf_trace_vram_preflight": {"gpus": gpus, "min_vram_mb": min_vram_mb}}, flush=True)
    if int(best["memory_mb"]) < min_vram_mb:
        raise RuntimeError(
            f"Local Qwen 7B trace generation requires at least {min_vram_mb} MB VRAM by default; "
            f"best visible GPU is {best['name']} with {best['memory_mb']} MB. "
            "Use an L4/A100/H100/high-memory runtime, reduce the model, or set "
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_SKIP_VRAM_CHECK=1 deliberately."
        )


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


def current_source_pointer():
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if pointer.exists():
        raw = pointer.read_text(encoding="utf-8").strip()
        if raw:
            return resolve_repo_path(raw)
    return None


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

    pointer = current_source_pointer()
    if pointer and is_gate_ready_trace_collection(pointer):
        return pointer

    for candidate in trace_collection_summary_paths():
        if is_gate_ready_trace_collection(candidate):
            return candidate
    raise RuntimeError("No gate-ready capability-ladder trace collection summary found after trace collection.")


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
                os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_MODEL_NAME", DEFAULT_STUDENT_MODEL),
            ),
            "STAGE5_CURRICULUM_SFT_RUN_ID": os.environ.get("STAGE5_CURRICULUM_SFT_RUN_ID")
            or time.strftime("stage5_local_hf_traced_capability_sft_%Y%m%d_%H%M%S"),
            "STAGE5_CURRICULUM_WORK_DIR": work_dir,
            "STAGE5_CURRICULUM_SUMMARY_JSON": summary_json,
            "STAGE5_CURRICULUM_MIN_POSITIVE_ROWS": str(positive_rows),
            "STAGE5_CURRICULUM_PHASE1_STEPS": str(phase1_steps),
            "STAGE5_CURRICULUM_MAX_LOOPS": str(max_loops),
            "DTYPE": os.environ.get("DTYPE", "bfloat16"),
            "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
            "DEVICE": os.environ.get("DEVICE", "cuda"),
            "STAGE5_CURRICULUM_ALLOW_ANSWER_LINE_VERIFICATION": "1",
            "STAGE5_CURRICULUM_SFT_COMMIT_CHECKPOINTS": os.environ.get(
                "STAGE5_CURRICULUM_SFT_COMMIT_CHECKPOINTS",
                "1",
            ),
            "STAGE5_CURRICULUM_SFT_PUSH": os.environ.get("STAGE5_CURRICULUM_SFT_PUSH", "1"),
        }
    )
    if min_mode_rows:
        env["STAGE5_CURRICULUM_MIN_MODE_ROWS"] = min_mode_rows
    drive_root = str(drive_backup.get("dest_root") or "").strip()
    if drive_root:
        env["STAGE5_CURRICULUM_INPUT_BACKUP_DIR"] = drive_root
    if env_flag("STAGE5_TRACED_CAPABILITY_SFT_ALLOW_NO_DRIVE_BACKUP", "1"):
        env["STAGE5_CURRICULUM_ALLOW_NO_DRIVE_BACKUP"] = "1"

    resume_from = os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_RESUME_FROM", "").strip()
    if resume_from:
        env["STAGE5_CURRICULUM_RESUME_FROM"] = resume_from
    else:
        # Favor the best currently-known learned-depth checkpoint when present.
        candidate = ROOT / "outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/phase1/phase1_step_400.pt"
        if candidate.exists():
            env["STAGE5_CURRICULUM_RESUME_FROM"] = path_for_cli(candidate)

    env["STAGE5_CURRICULUM_USE_TARGET_LOOP_CONTROL"] = os.environ.get(
        "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_USE_TARGET_LOOP_CONTROL",
        "0",
    )
    env["STAGE5_CURRICULUM_USE_LEARNED_LOOP_CONTROL"] = os.environ.get(
        "STAGE5_CURRICULUM_USE_LEARNED_LOOP_CONTROL",
        "1",
    )
    env["STAGE5_CURRICULUM_LOOP_CONTROL_CE_WEIGHT"] = os.environ.get(
        "STAGE5_CURRICULUM_LOOP_CONTROL_CE_WEIGHT",
        "8.0",
    )
    env["STAGE5_CURRICULUM_HALT_TARGET_NLL_WEIGHT"] = os.environ.get(
        "STAGE5_CURRICULUM_HALT_TARGET_NLL_WEIGHT",
        "10.0",
    )
    env["STAGE5_CURRICULUM_OPTIMIZER_MODULES"] = os.environ.get(
        "STAGE5_CURRICULUM_OPTIMIZER_MODULES",
        "halt",
    )
    env["STAGE5_CURRICULUM_PHASE1_LR"] = os.environ.get("STAGE5_CURRICULUM_PHASE1_LR", "2e-4")
    env["STAGE5_CURRICULUM_PHASE1_BETA"] = os.environ.get("STAGE5_CURRICULUM_PHASE1_BETA", "0.12")
    env["STAGE5_CURRICULUM_DEPTH_HINT_STYLE"] = os.environ.get(
        "STAGE5_CURRICULUM_DEPTH_HINT_STYLE",
        "natural",
    )

    print(
        json.dumps(
            {
                "sft_source_summary": path_for_cli(summary_path),
                "positive_rows": positive_rows,
                "min_mode_rows": min_mode_rows,
                "max_loops": max_loops,
                "phase1_steps": phase1_steps,
                "student_model": env["MODEL_NAME"],
                "resume_from": env.get("STAGE5_CURRICULUM_RESUME_FROM", ""),
                "optimizer_modules": env.get("STAGE5_CURRICULUM_OPTIMIZER_MODULES", ""),
            },
            indent=2,
        ),
        flush=True,
    )
    return env


print(
    "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_CELL_VERSION="
    f"{STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_CELL_VERSION}",
    flush=True,
)

run(["nvidia-smi"], cwd=Path("/content"), check=False)
require_enough_vram_for_local_hf()

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
    print("HF auth skipped; local 7B download may be rate limited.", flush=True)

run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_curriculum_job_responses.py",
        "tests/test_stage5_capability_ladder_trace_responses_runner.py",
        "tests/test_stage5_capability_ladder_trace_collect_runner.py",
        "tests/test_stage5_curriculum_sft.py",
        "tests/test_curriculum_sft_gate.py",
    ],
    cwd=ROOT,
)

trace_env = os.environ.copy()
min_trace_rows_for_sft = os.environ.get(
    "STAGE5_TRACED_CAPABILITY_SFT_MIN_TRACE_ROWS",
    str(MIN_TRACE_ROWS_DEFAULT),
)
trace_env.update(
    {
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BACKEND": "hf_local",
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_LOCAL_HF": "1",
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_MODEL_NAME": os.environ.get(
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_MODEL_NAME",
            DEFAULT_TRACE_MODEL,
        ),
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_DTYPE": os.environ.get(
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_DTYPE",
            "bfloat16",
        ),
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_DEVICE": os.environ.get(
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_DEVICE",
            "cuda",
        ),
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_TOP_P": os.environ.get(
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_TOP_P",
            "0.95",
        ),
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_STUDENT_LINEAGE": "1",
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT": os.environ.get(
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT",
            "32",
        ),
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MAX_TOKENS": os.environ.get(
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MAX_TOKENS",
            "1536",
        ),
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_TEMPERATURE": os.environ.get(
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_TEMPERATURE",
            "0.2",
        ),
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_TIMEOUT_SEC": os.environ.get(
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_TIMEOUT_SEC",
            "300",
        ),
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_FAIL_FAST": os.environ.get(
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_FAIL_FAST",
            "0",
        ),
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_GPU": "1",
        "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_ALLOW_GPU": "1",
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_REFUSE_GPU": "0",
        "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_REFUSE_GPU": "0",
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BACKUP_DRIVE": os.environ.get(
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BACKUP_DRIVE",
            "0",
        ),
        "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_BACKUP_DRIVE": os.environ.get(
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_BACKUP_DRIVE",
            "0",
        ),
        "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_MIN_POSITIVE_ROWS": os.environ.get(
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_MIN_POSITIVE_ROWS",
            min_trace_rows_for_sft,
        ),
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_PUSH": "1",
        "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_PUSH": "1",
        "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COMMIT_RESPONSES": "1",
    }
)

print("=== Local-HF trace responses ===", flush=True)
run([sys.executable, "colab/run_stage5_capability_ladder_trace_responses.py"], cwd=ROOT, env=trace_env)

print("=== Sync after trace response commit ===", flush=True)
run(["git", "pull", "--rebase", "origin", "main"], cwd=ROOT)

print("=== Trace collection/gate ===", flush=True)
run([sys.executable, "colab/run_stage5_capability_ladder_trace_collect.py"], cwd=ROOT, env=trace_env)

run_sft = env_flag("STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_RUN_SFT", "1")
if run_sft:
    print("=== Bounded traced capability-ladder recurrent SFT ===", flush=True)
    trace_collection = resolve_trace_collection_summary()
    sft_env = derive_sft_env(trace_collection)
    run([sys.executable, "colab/run_stage5_curriculum_sft.py"], cwd=ROOT, env=sft_env)

    run_benchmark = env_flag("STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_RUN_BENCHMARK", "1")
    if run_benchmark:
        print("=== Bounded post-SFT recurrent-vs-base benchmark ===", flush=True)
        summary_pointer = ROOT / "config" / "stage5_current_source_summary.txt"
        source_summary = summary_pointer.read_text(encoding="utf-8").strip()
        benchmark_env = os.environ.copy()
        benchmark_env.update(
            {
                "STAGE5_BENCHMARK_SOURCE_SUMMARY": source_summary,
                "STAGE5_BENCHMARK_SUITE_RUN_ID": os.environ.get("STAGE5_BENCHMARK_SUITE_RUN_ID")
                or time.strftime("stage5_local_hf_traced_sft_benchmark_%Y%m%d_%H%M%S"),
                "STAGE5_BENCHMARKS": os.environ.get(
                    "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_BENCHMARKS",
                    "arc_challenge",
                ),
                "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": os.environ.get(
                    "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_ARC_CHALLENGE_LIMIT",
                    "96",
                ),
                "STAGE5_BENCHMARK_ARC_EASY_LIMIT": os.environ.get(
                    "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_ARC_EASY_LIMIT",
                    "96",
                ),
                "STAGE5_BENCHMARK_SCORE_TARGETS": os.environ.get(
                    "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_SCORE_TARGETS",
                    "content_question_only,cyclic_label_aggregated",
                ),
                "STAGE5_BENCHMARK_AGGREGATES": os.environ.get(
                    "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_AGGREGATES",
                    "mean",
                ),
                "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL": os.environ.get(
                    "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL",
                    "1",
                ),
                "STAGE5_BENCHMARK_PUSH": "1",
                "DTYPE": os.environ.get("DTYPE", "bfloat16"),
                "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
                "DEVICE": os.environ.get("DEVICE", "cuda"),
            }
        )
        run([sys.executable, "colab/run_stage5_benchmark_suite.py"], cwd=ROOT, env=benchmark_env)
    else:
        print("Post-SFT benchmark skipped by STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_RUN_BENCHMARK=0.", flush=True)
else:
    print("SFT skipped by STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_RUN_SFT=0.", flush=True)

summary_pointer = ROOT / "config" / "stage5_current_source_summary.txt"
print("current_source_summary:", summary_pointer.read_text(encoding="utf-8").strip(), flush=True)
run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)

if env_flag("STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_DISCONNECT", "1"):
    print("Disconnecting Colab runtime to conserve credits.", flush=True)
    runtime.unassign()
else:
    print("Leaving Colab runtime connected for follow-up inspection.", flush=True)
