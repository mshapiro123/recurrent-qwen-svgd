import os
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from google.colab import drive, runtime, userdata

STAGE5_DIRECT_PRESERVATION_PROBE_CELL_VERSION = "direct_preservation_probe_v1"

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")

DEFAULT_SOURCE_SUMMARY = (
    "outputs/stage5/"
    "stage5_arc_agi_next_action_20260622_181850_plan_conservative_direct_preservation/"
    "answer_prior_diagnosis.json"
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


def env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


DRIVE_BACKUP = env_bool("STAGE5_DIRECT_PRESERVE_DRIVE_BACKUP", False)
DISCONNECT_WHEN_DONE = env_bool("STAGE5_DIRECT_PRESERVE_DISCONNECT", False)
CHAIN_CONFIRM_WHEN_PASSED = env_bool("STAGE5_DIRECT_PRESERVE_CHAIN_CONFIRM", False)
CHAIN_DEPTH_ROUTER_WHEN_CONFIRMED = env_bool("STAGE5_DIRECT_PRESERVE_CHAIN_DEPTH_ROUTER", False)
SWEEP_SPEC = os.environ.get("STAGE5_DIRECT_PRESERVE_SWEEP", "").strip()
RUN_ID = os.environ.get("STAGE5_DIRECT_PRESERVE_RUN_ID") or time.strftime(
    "stage5_direct_preservation_loop1_%Y%m%d_%H%M%S"
)
SOURCE_SUMMARY = os.environ.get("STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY") or DEFAULT_SOURCE_SUMMARY
CURRENT_STAGE = "startup"


def set_stage(name):
    global CURRENT_STAGE
    CURRENT_STAGE = str(name)
    print(f"stage={CURRENT_STAGE}", flush=True)


def printable_cmd(cmd):
    text = redact(" ".join(map(str, cmd)))
    return text


def redact(value):
    text = str(value)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            text = text.replace(token, "****")
    return text


def run(cmd, *, cwd=None, env=None, check=True):
    print("$", printable_cmd(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


def sanitize_run_id_part(value):
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return text.strip("_") or "attempt"


SWEEP_ENV_KEYS = {
    "lr": "STAGE5_DIRECT_PRESERVE_LR",
    "learning_rate": "STAGE5_DIRECT_PRESERVE_LR",
    "steps": "STAGE5_DIRECT_PRESERVE_MAX_STEPS",
    "max_steps": "STAGE5_DIRECT_PRESERVE_MAX_STEPS",
    "distill": "STAGE5_DIRECT_PRESERVE_DISTILL_WEIGHT",
    "distill_weight": "STAGE5_DIRECT_PRESERVE_DISTILL_WEIGHT",
    "temperature": "STAGE5_DIRECT_PRESERVE_DISTILL_TEMPERATURE",
    "distill_temperature": "STAGE5_DIRECT_PRESERVE_DISTILL_TEMPERATURE",
    "beta": "STAGE5_DIRECT_PRESERVE_BETA",
    "min_margin": "STAGE5_DIRECT_PRESERVE_MIN_BASE_MARGIN",
    "min_base_margin": "STAGE5_DIRECT_PRESERVE_MIN_BASE_MARGIN",
    "save_every": "STAGE5_DIRECT_PRESERVE_SAVE_EVERY",
    "seed": "STAGE5_DIRECT_PRESERVE_SEED",
}


def parse_sweep_spec():
    if not SWEEP_SPEC:
        return [{"name": "main", "run_id": RUN_ID, "overrides": {}}]
    attempts = []
    for raw_attempt in SWEEP_SPEC.split(";"):
        raw_attempt = raw_attempt.strip()
        if not raw_attempt:
            continue
        name, _, raw_params = raw_attempt.partition(":")
        name = sanitize_run_id_part(name)
        overrides = {}
        if raw_params.strip():
            for raw_pair in raw_params.split(","):
                raw_pair = raw_pair.strip()
                if not raw_pair:
                    continue
                key, sep, value = raw_pair.partition("=")
                if not sep:
                    raise ValueError(f"Bad STAGE5_DIRECT_PRESERVE_SWEEP item {raw_pair!r}; expected key=value.")
                env_key = SWEEP_ENV_KEYS.get(key.strip().lower())
                if not env_key:
                    raise ValueError(f"Unknown STAGE5_DIRECT_PRESERVE_SWEEP key {key!r}.")
                overrides[env_key] = value.strip()
        attempts.append({"name": name, "run_id": f"{RUN_ID}_{name}", "overrides": overrides})
    if not attempts:
        raise ValueError("STAGE5_DIRECT_PRESERVE_SWEEP was set but contained no attempts.")
    return attempts


def direct_preservation_env(attempt):
    env = os.environ.copy()
    env.update(
        {
            "STAGE5_DIRECT_PRESERVE_RUN_ID": attempt["run_id"],
            "STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY": SOURCE_SUMMARY,
            "STAGE5_DIRECT_PRESERVE_ARC_TRAIN_LIMIT": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_ARC_TRAIN_LIMIT", "512"
            ),
            "STAGE5_DIRECT_PRESERVE_ARC_EVAL_LIMIT": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_ARC_EVAL_LIMIT", "128"
            ),
            "STAGE5_DIRECT_PRESERVE_MAX_STEPS": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_MAX_STEPS", "75"
            ),
            "STAGE5_DIRECT_PRESERVE_SAVE_EVERY": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_SAVE_EVERY", "25"
            ),
            "STAGE5_DIRECT_PRESERVE_MIN_BASE_MARGIN": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_MIN_BASE_MARGIN", "1.0"
            ),
            "STAGE5_DIRECT_PRESERVE_LR": os.environ.get("STAGE5_DIRECT_PRESERVE_LR", "5e-7"),
            "STAGE5_DIRECT_PRESERVE_BETA": os.environ.get("STAGE5_DIRECT_PRESERVE_BETA", "0.02"),
            "STAGE5_DIRECT_PRESERVE_DISTILL_WEIGHT": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_DISTILL_WEIGHT", "1.0"
            ),
            "STAGE5_DIRECT_PRESERVE_DISTILL_TEMPERATURE": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_DISTILL_TEMPERATURE", "2.0"
            ),
        }
    )
    env.update(attempt["overrides"])
    return env


def sync_repo():
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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_from_probe(payload):
    best = payload.get("best_checkpoint")
    if isinstance(best, dict) and best.get("checkpoint"):
        return str(best["checkpoint"])
    for key in ("checkpoint", "phase1_checkpoint", "resume_checkpoint"):
        if payload.get(key):
            return str(payload[key])
    return None


def safe_stage_and_push(run_dir):
    pointer = ROOT / "config" / "stage5_latest_direct_preservation_summary.txt"
    current_pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    summary_rel = f"{run_dir.relative_to(ROOT).as_posix()}/summary.json"
    pointer.write_text(f"{summary_rel}\n", encoding="utf-8")
    current_pointer.write_text(f"{summary_rel}\n", encoding="utf-8")
    suffixes = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".log", ".csv"}
    files = [path for path in run_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]
    summary = run_dir / "summary.json"
    if summary.exists():
        try:
            checkpoint = checkpoint_from_probe(read_json(summary))
        except Exception as exc:
            checkpoint = ""
            print(f"checkpoint_publish_probe_failed={exc}", flush=True)
        if checkpoint:
            checkpoint_path = ROOT / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
            if checkpoint_path.exists() and ROOT in checkpoint_path.resolve().parents:
                files.append(checkpoint_path)
                print(f"staged_selected_checkpoint={checkpoint_path.relative_to(ROOT).as_posix()}", flush=True)
            else:
                print(f"selected_checkpoint_not_visible_for_publish={checkpoint}", flush=True)
    files.append(pointer)
    files.append(current_pointer)
    files = sorted(set(files), key=lambda path: path.as_posix())
    if not files:
        print("No lightweight output files to commit.", flush=True)
        return
    rels = [str(path.relative_to(ROOT)) for path in files]
    run(["git", "add", "-f", *rels], cwd=ROOT)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, stdout=subprocess.PIPE)
    if not status.stdout.strip():
        print("No git changes to commit.", flush=True)
        return
    try:
        run(["git", "commit", "-m", f"Record Stage 5 direct preservation probe {run_dir.name} [skip ci]"], cwd=ROOT)
        run(["git", "fetch", "origin", "main"], cwd=ROOT)
        run(["git", "rebase", "origin/main"], cwd=ROOT)
        run(["git", "push", "origin", "main"], cwd=ROOT)
    except Exception as exc:
        print(f"WARNING: GitHub publish failed; local result files remain in the Colab runtime: {exc}", flush=True)


def maybe_chain_confirmation(run_dir):
    if not CHAIN_CONFIRM_WHEN_PASSED:
        print("direct_preservation_chain_confirm=disabled", flush=True)
        return False
    summary = run_dir / "summary.json"
    payload = read_json(summary)
    if payload.get("passed") is not True:
        print(
            f"direct_preservation_chain_confirm=skipped status={payload.get('status')} passed={payload.get('passed')}",
            flush=True,
        )
        return False
    checkpoint = checkpoint_from_probe(payload)
    assert checkpoint, f"Direct-preservation probe passed but exposed no checkpoint: {summary}"
    confirm_run_id = os.environ.get("STAGE5_DIRECT_CONFIRM_RUN_ID") or f"{RUN_ID}_confirm"
    benchmark_env = os.environ.copy()
    benchmark_env.update(
        {
            "STAGE5_BENCHMARK_SUITE_RUN_ID": confirm_run_id,
            "STAGE5_BENCHMARK_SOURCE_SUMMARY": summary.relative_to(ROOT).as_posix(),
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
    set_stage("direct_preservation_confirmation")
    print("direct_preservation_confirmation_run_id:", confirm_run_id, flush=True)
    print("direct_preservation_confirmation_source:", summary.relative_to(ROOT).as_posix(), flush=True)
    print("direct_preservation_confirmation_checkpoint:", checkpoint, flush=True)
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], cwd=ROOT, env=benchmark_env)

    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    benchmark_summary = pointer.read_text(encoding="utf-8").strip()
    assess_env = os.environ.copy()
    assess_env.update(
        {
            "STAGE5_BENCHMARK_ASSESS_RUN_ID": os.environ.get("STAGE5_DIRECT_CONFIRM_ASSESS_RUN_ID")
            or f"{confirm_run_id}_assessment",
            "STAGE5_BENCHMARK_ASSESS_SCORE_TARGET": os.environ.get(
                "STAGE5_DIRECT_CONFIRM_ASSESS_SCORE_TARGET",
                "content_question_only",
            ),
            "STAGE5_BENCHMARK_ASSESS_MIN_ARC_EXAMPLES": os.environ.get(
                "STAGE5_DIRECT_CONFIRM_ASSESS_MIN_ARC_EXAMPLES",
                "128",
            ),
            "STAGE5_BENCHMARK_ASSESS_PUSH": "1",
        }
    )
    set_stage("direct_preservation_confirmation_assessment")
    run(
        [sys.executable, "colab/assess_stage5_benchmark_suite.py", "--summary_json", benchmark_summary],
        cwd=ROOT,
        env=assess_env,
    )
    assessment_summary = pointer.read_text(encoding="utf-8").strip()
    print("direct_preservation_confirmation_assessment:", assessment_summary, flush=True)
    try:
        assessment_payload = read_json(ROOT / assessment_summary)
    except Exception as exc:
        print(f"direct_preservation_confirmation_assessment_read_failed={exc}", flush=True)
        return False
    return assessment_payload.get("passed") is True


def maybe_chain_depth_router(run_dir, *, confirmation_passed):
    if not CHAIN_DEPTH_ROUTER_WHEN_CONFIRMED:
        print("direct_preservation_chain_depth_router=disabled", flush=True)
        return
    if not confirmation_passed:
        print("direct_preservation_chain_depth_router=skipped confirmation_passed=False", flush=True)
        return
    summary = run_dir / "summary.json"
    depth_env = os.environ.copy()
    depth_env.update(
        {
            "STAGE5_DEPTH_ROUTER_DIRECT_SOURCE_SUMMARY": summary.relative_to(ROOT).as_posix(),
            "STAGE5_DEPTH_ROUTER_TRACE_SOURCE_SUMMARY": os.environ.get(
                "STAGE5_DIRECT_DEPTH_ROUTER_TRACE_SOURCE_SUMMARY",
                "outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json",
            ),
            "STAGE5_DEPTH_ROUTER_RUN_ID": os.environ.get("STAGE5_DIRECT_DEPTH_ROUTER_RUN_ID")
            or f"{RUN_ID}_depth_router",
            "STAGE5_DEPTH_ROUTER_DISCONNECT": "0",
        }
    )
    set_stage("direct_preservation_depth_router")
    run([sys.executable, "colab/STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_CELL.py"], cwd=ROOT, env=depth_env)


def disconnect(reason):
    if not DISCONNECT_WHEN_DONE:
        return
    try:
        print(f"Disconnecting Colab runtime to conserve credits: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped: {exc}", flush=True)


def write_failure_summary(exc_type, exc, tb):
    try:
        run_dir = ROOT / "outputs" / "stage5" / f"{RUN_ID}_failure"
        run_dir.mkdir(parents=True, exist_ok=True)
        traceback_lines = traceback.format_exception(exc_type, exc, tb)
        payload = {
            "kind": "stage5_direct_preservation_probe_failure",
            "status": "failed",
            "run_id": f"{RUN_ID}_failure",
            "stage": CURRENT_STAGE,
            "source_summary": SOURCE_SUMMARY,
            "exception_type": getattr(exc_type, "__name__", str(exc_type)),
            "exception": redact(exc),
            "traceback_tail": [
                redact(line.rstrip())
                for line in "".join(traceback_lines).splitlines()[-120:]
            ],
            "target": os.environ.get("STAGE5_CURRENT_A100_TARGET", ""),
        }
        summary = run_dir / "summary.json"
        summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("failure_summary:", summary.relative_to(ROOT).as_posix(), flush=True)
        safe_stage_and_push(run_dir)
    except Exception as hook_exc:
        print(f"failure_summary_hook_failed: {hook_exc}", flush=True)


def write_attempt_failure_summary(run_dir, attempt, exc):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "stage5_direct_preservation_probe",
        "status": "direct_route_attempt_failed",
        "passed": False,
        "run_id": attempt["run_id"],
        "attempt": {
            "name": attempt["name"],
            "run_id": attempt["run_id"],
            "overrides": attempt["overrides"],
        },
        "stage": CURRENT_STAGE,
        "source_summary": SOURCE_SUMMARY,
        "exception_type": type(exc).__name__,
        "exception": redact(exc),
        "best_checkpoint": {},
        "next_step": (
            "Inspect the child process output for this attempt before spending more GPU. "
            "The sweep stopped because the attempt failed before a normal summary was available."
        ),
    }
    summary = run_dir / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage 5 Direct Preservation Attempt Failure",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Attempt: `{attempt['name']}`",
        f"- Status: `{payload['status']}`",
        f"- Stage: `{payload['stage']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Exception: `{payload['exception_type']}: {payload['exception']}`",
        "",
        payload["next_step"],
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("attempt_failure_summary:", summary.relative_to(ROOT).as_posix(), flush=True)


def failure_excepthook(exc_type, exc, tb):
    write_failure_summary(exc_type, exc, tb)
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = failure_excepthook


try:
    set_stage("gpu_preflight")
    assert shutil.which("nvidia-smi"), "This cell is intended for a GPU runtime; nvidia-smi was not found."
    run(["nvidia-smi"], check=False)

    set_stage("drive_optional")
    if DRIVE_BACKUP:
        drive.mount("/content/drive", force_remount=False)
    else:
        print("Drive backup disabled; using GitHub as primary artifact store.", flush=True)
    set_stage("repo_sync")
    sync_repo()
    os.chdir(ROOT)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)

    set_stage("install_dependencies")
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

    attempts = parse_sweep_spec()
    print("direct_preservation_probe_run_id:", RUN_ID, flush=True)
    print("direct_preservation_source_summary:", SOURCE_SUMMARY, flush=True)
    print(
        "direct_preservation_attempts:",
        json.dumps(
            [
                {
                    "name": attempt["name"],
                    "run_id": attempt["run_id"],
                    "overrides": attempt["overrides"],
                }
                for attempt in attempts
            ],
            sort_keys=True,
        ),
        flush=True,
    )

    selected_run_dir = None
    attempt_results = []
    for index, attempt in enumerate(attempts, start=1):
        attempt_run_id = attempt["run_id"]
        attempt_env = direct_preservation_env(attempt)
        set_stage(f"direct_preservation_probe_{index}_{attempt['name']}")
        print(
            "direct_preservation_attempt:",
            json.dumps(
                {
                    "index": index,
                    "name": attempt["name"],
                    "run_id": attempt_run_id,
                    "overrides": attempt["overrides"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        attempt_run_dir = ROOT / "outputs" / "stage5" / attempt_run_id
        try:
            run([sys.executable, "colab/run_stage5_direct_preservation_probe.py"], cwd=ROOT, env=attempt_env)
            if not (attempt_run_dir / "summary.json").exists():
                raise RuntimeError(f"Direct-preservation attempt returned without summary: {attempt_run_dir}")
        except Exception as exc:
            set_stage(f"direct_preservation_attempt_failed_{index}_{attempt['name']}")
            write_attempt_failure_summary(attempt_run_dir, attempt, exc)
            safe_stage_and_push(attempt_run_dir)
            raise

        assert attempt_run_dir.exists(), f"Expected run_dir was not created: {attempt_run_dir}"
        set_stage(f"backup_and_publish_{index}_{attempt['name']}")
        if DRIVE_BACKUP:
            drive_dst = DRIVE_ARTIFACT_ROOT / "stage5" / attempt_run_id
            drive_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(attempt_run_dir, drive_dst, dirs_exist_ok=True)
            print(f"backed_up_run_dir={attempt_run_dir} -> {drive_dst}", flush=True)
        else:
            print(f"drive_backup_skipped={attempt_run_id}", flush=True)

        safe_stage_and_push(attempt_run_dir)
        attempt_payload = read_json(attempt_run_dir / "summary.json")
        attempt_results.append(
            {
                "name": attempt["name"],
                "run_id": attempt_run_id,
                "status": attempt_payload.get("status"),
                "passed": attempt_payload.get("passed"),
                "summary": (attempt_run_dir / "summary.json").relative_to(ROOT).as_posix(),
            }
        )
        if attempt_payload.get("passed") is True:
            selected_run_dir = attempt_run_dir
            print(f"direct_preservation_selected_attempt={attempt_run_id}", flush=True)
            break
        print(
            f"direct_preservation_attempt_not_passed={attempt_run_id} status={attempt_payload.get('status')}",
            flush=True,
        )

    if selected_run_dir is None:
        print(
            "direct_preservation_no_attempt_passed:",
            json.dumps(attempt_results, sort_keys=True),
            flush=True,
        )
        disconnect("direct preservation attempts finished without a pass")
    else:
        confirmation_passed = maybe_chain_confirmation(selected_run_dir)
        maybe_chain_depth_router(selected_run_dir, confirmation_passed=confirmation_passed)
        disconnect("direct preservation probe finished")
except Exception:
    disconnect("direct preservation probe errored")
    raise
