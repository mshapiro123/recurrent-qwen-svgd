"""Colab cell: run the 7B capability ladder and build trace jobs.

This is a high-memory GPU target. It scores the ARC capability ladder with
Qwen 0.5B/1.5B/3B/7B, keeps the local scored rows alive, then immediately
builds provider-neutral trace-generation jobs from that result. It does not
call paid teacher APIs and does not train recurrent weights.
"""

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_CELL_VERSION = "capability_ladder_7b_trace_chain_v1"
STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_TARGET = "capability_ladder_7b_trace_chain"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


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

summary_pointer = ROOT / "config" / "stage5_current_source_summary.txt"
print("current_source_summary:", summary_pointer.read_text(encoding="utf-8").strip(), flush=True)
run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)

if env_flag("STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_DISCONNECT", "1"):
    print("Disconnecting Colab runtime to conserve credits.", flush=True)
    runtime.unassign()
else:
    print("Leaving Colab runtime connected for follow-up inspection.", flush=True)
