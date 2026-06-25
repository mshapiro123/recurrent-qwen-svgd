"""Colab cell: train recurrent Phase 1 on traced capability-ladder rows.

This GPU target follows the latest gate-ready capability-ladder trace
collection, derives the SFT row/mode/depth gates from that artifact, and runs a
bounded deterministic recurrent Phase 1 SFT. It is the bridge from provider
trace collection to actual recurrent training; it does not run Phase 2/SVGD.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL_VERSION = "traced_capability_ladder_sft_v1"
STAGE5_TRACED_CAPABILITY_LADDER_SFT_TARGET = "traced_capability_ladder_sft"
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
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(redact(proc.stdout), flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {printable}")
    return proc


def read_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def current_source_pointer():
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if pointer.exists():
        value = pointer.read_text(encoding="utf-8").strip()
        if value:
            return resolve_repo_path(value)
    return None


def trace_collection_summary_paths():
    roots = [ROOT / "outputs" / "stage5"]
    drive_root = Path("/content/drive/MyDrive/recurrent-qwen-svgd/stage5_capability_ladder_trace_collection")
    if drive_root.exists():
        roots.append(drive_root)
    for root in roots:
        if not root.exists():
            continue
        yield from sorted(root.glob("**/summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)


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
    raise RuntimeError(
        "No gate-ready stage5_capability_ladder_trace_collection summary found. "
        "Run capability_ladder_trace_response_collect_cpu or capability_ladder_trace_collect_cpu first."
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


def derive_training_env(summary_path):
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
    allow_tiny = os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_ALLOW_TINY", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
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

    run_id = os.environ.get("STAGE5_CURRICULUM_SFT_RUN_ID") or time.strftime(
        "stage5_traced_capability_ladder_sft_%Y%m%d_%H%M%S"
    )
    env = os.environ.copy()
    env.update(
        {
            "MODEL_NAME": os.environ.get("MODEL_NAME", os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_MODEL_NAME", DEFAULT_MODEL_NAME)),
            "STAGE5_CURRICULUM_SFT_RUN_ID": run_id,
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
                "STAGE5_TRACED_CAPABILITY_SFT_COMMIT_CHECKPOINTS",
                "0",
            ),
        }
    )
    if min_mode_rows:
        env["STAGE5_CURRICULUM_MIN_MODE_ROWS"] = min_mode_rows
    drive_root = str(drive_backup.get("dest_root") or "").strip()
    if drive_root:
        env["STAGE5_CURRICULUM_INPUT_BACKUP_DIR"] = drive_root
    if os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_ALLOW_NO_DRIVE_BACKUP", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        env["STAGE5_CURRICULUM_ALLOW_NO_DRIVE_BACKUP"] = "1"

    print(
        json.dumps(
            {
                "source_summary": str(summary_path),
                "work_dir": work_dir,
                "summary_json": summary_json,
                "positive_rows": positive_rows,
                "min_mode_rows": min_mode_rows,
                "max_loops": max_loops,
                "phase1_steps": phase1_steps,
                "drive_backup_root": drive_root,
                "model_name": env["MODEL_NAME"],
            },
            indent=2,
        ),
        flush=True,
    )
    return env


print(f"STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL_VERSION={STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL_VERSION}", flush=True)

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
run(["git", "log", "--oneline", "-5"], cwd=ROOT)
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

if HF_TOKEN:
    from huggingface_hub import HfApi, login

    login(token=HF_TOKEN, add_to_git_credential=False)
    who = HfApi(token=HF_TOKEN).whoami()
    print("HF auth OK:", who.get("name") or who.get("email") or "authenticated user", flush=True)

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive", force_remount=True)

run(["nvidia-smi"], cwd=ROOT, check=False)
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_stage5_curriculum_sft.py",
        "tests/test_curriculum_sft_gate.py",
        "tests/test_stage5_a100_go_no_go.py",
        "tests/test_stage5_next_plan.py",
    ],
    cwd=ROOT,
)

source_summary = resolve_trace_collection_summary()
env = derive_training_env(source_summary)
run([sys.executable, "colab/run_stage5_curriculum_sft.py"], cwd=ROOT, env=env)

if os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_DISCONNECT", "1") == "1":
    print("Disconnecting Colab runtime to conserve credits.", flush=True)
    runtime.unassign()
