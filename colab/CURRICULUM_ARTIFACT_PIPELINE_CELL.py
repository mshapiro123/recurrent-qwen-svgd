import json, os, shutil, subprocess, sys
from pathlib import Path
from google.colab import drive, runtime, userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
WORK_DIR = os.environ.get("STAGE5_CURRICULUM_ARTIFACT_WORK_DIR", "data/curriculum/claim_direct_deep_001")

# CPU/network workflow. Leave this False until the model map and provider secret are set.
RUN_PROVIDER_RESPONSES = os.environ.get("STAGE5_CURRICULUM_RUN_PROVIDER_RESPONSES", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
PROVIDER_BACKEND = "openai_compatible"  # or "command"
PROVIDER_COMMAND = "python scripts/my_provider_runner.py"
PROVIDER_LIMIT_RAW = os.environ.get("STAGE5_CURRICULUM_PROVIDER_LIMIT", "none").strip().lower()
PROVIDER_LIMIT = None if PROVIDER_LIMIT_RAW in {"", "none", "all"} else int(PROVIDER_LIMIT_RAW)
MIN_POSITIVE_ROWS = 2000
MIN_MODE_ROWS = "direct=1000,deep_narrow=1000"

API_KEY_ENV = "OPENAI_API_KEY"
BASE_URL = "https://api.openai.com/v1"
MAX_TOKENS = 2048
TEMPERATURE = 0.2
JSON_MODE = True
EXTRA_HEADERS = {}  # Example for OpenRouter: {"HTTP-Referer": "https://colab.research.google.com", "X-Title": "recurrent-qwen-curriculum"}
EXTRA_BODY_JSON = {}  # Provider-specific controls, e.g. {"top_p": 0.95}; may not override model/messages/stream.
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 2.0
RETRY_BACKOFF = 2.0

MOUNT_DRIVE = True
BACKUP_TO_DRIVE = True
DISCONNECT_RUNTIME_WHEN_DONE = False
REFUSE_GPU_RUNTIME = True
ALLOW_GPU_RUNTIME_FOR_CPU_API_CELL = False

MODEL_MAP = {
    "opus-strong": "replace-with-opus-compatible-model-id",
    "glm-strong": "replace-with-glm-compatible-model-id",
    "weak-reference": "replace-with-cheap-reference-model-id",
}

PIPELINE_ARGS = [
    "--work_dir",
    WORK_DIR,
    "--seed_models",
    "opus-strong,glm-strong",
    "--solver_models",
    "opus-strong,glm-strong",
    "--judge_models",
    "opus-strong,glm-strong",
    "--domains",
    "math,science",
    "--difficulties",
    "medium,hard",
    "--target_steps",
    "1,2,5,9",
    "--count_per_combo",
    "122",
    "--reference_model",
    "weak-reference",
    "--reference_samples",
    "3",
    "--min_reference_samples",
    "1",
    "--min_natural_agree",
    "2",
    "--min_distinct_agree",
    "2",
    "--require_programmatic_answer_check",
]


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
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
provider_key = secret(API_KEY_ENV)
if provider_key:
    os.environ[API_KEY_ENV] = provider_key


def redacted(text):
    text = str(text)
    if GH_TOKEN:
        text = text.replace(GH_TOKEN, "****")
    if provider_key:
        text = text.replace(provider_key, "****")
    return text


def run(cmd, cwd=None, env=None, check=True):
    printable = redacted(" ".join(map(str, cmd)))
    print("$", printable, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(redacted(proc.stdout), flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {printable}")
    return proc


def attached_gpu_names():
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def refuse_gpu_runtime_for_cpu_api_work():
    names = attached_gpu_names()
    if not names:
        return
    print("GPU runtime detected:", names, flush=True)
    if REFUSE_GPU_RUNTIME and not ALLOW_GPU_RUNTIME_FOR_CPU_API_CELL:
        print(
            "Refusing to run CPU/API curriculum pipeline on an attached GPU runtime. "
            "Switch to CPU/cheap runtime, or set ALLOW_GPU_RUNTIME_FOR_CPU_API_CELL=True deliberately.",
            flush=True,
        )
        if DISCONNECT_RUNTIME_WHEN_DONE:
            runtime.unassign()
        raise RuntimeError("CPU/API curriculum pipeline refused attached GPU runtime.")


def sync_repo():
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        try:
            run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
            run(["git", "fetch", "origin", "main"], cwd=ROOT)
            run(["git", "checkout", "main"], cwd=ROOT)
            pull = run(["git", "pull", "--ff-only", "origin", "main"], cwd=ROOT, check=False)
            if pull.returncode == 0:
                return
            print("Existing clone could not fast-forward; recloning cleanly.", flush=True)
        except Exception as exc:
            print(f"Existing clone refresh failed; recloning cleanly: {exc}", flush=True)
        shutil.rmtree(ROOT)
    run(["git", "clone", clone_url, str(ROOT)])


def run_pipeline():
    return run(
        [sys.executable, "training/run_curriculum_pipeline_from_artifacts.py", *PIPELINE_ARGS],
        cwd=ROOT,
    )


def summary_payload():
    path = ROOT / WORK_DIR / "summary.json"
    assert path.exists(), f"missing summary: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def response_pairs(summary):
    pending = summary.get("pending_responses") or []
    if pending:
        pairs = []
        for item in pending:
            remaining = int(item.get("remaining_rows") or 0)
            if remaining <= 0:
                continue
            jobs = Path(item["jobs_jsonl"])
            responses = Path(item["responses_jsonl"])
            print(
                "response pair pending: "
                f"{item.get('name')} has usable_job_ids={item.get('usable_job_ids', item.get('existing_rows', 0))}/"
                f"{item.get('expected_rows')} with remaining={remaining}",
                flush=True,
            )
            pairs.append((jobs, responses))
        return pairs

    artifacts = summary["artifacts"]
    status = summary["status"]
    mapping = {
        "pending_seed_responses": [("jobs_seed", "responses_seed")],
        "pending_ground_truth_responses": [("jobs_ground_truth", "responses_ground_truth")],
        "pending_reference_attempt_responses": [("jobs_reference_attempts", "responses_reference_attempts")],
        "pending_method_or_perturbation_responses": [
            ("jobs_methods", "responses_methods"),
            ("jobs_perturbation", "responses_perturbation"),
        ],
        "pending_judgment_responses": [
            ("jobs_naturalness", "responses_naturalness"),
            ("jobs_distinctness", "responses_distinctness"),
            ("jobs_depth", "responses_depth"),
            ("jobs_error_detection", "responses_error_detection"),
        ],
    }
    pairs = []
    for jobs_key, responses_key in mapping.get(status, []):
        jobs = Path(artifacts[jobs_key]["path"])
        responses = Path(artifacts[responses_key]["path"])
        job_lines = int(artifacts[jobs_key]["lines"])
        response_lines = int(artifacts[responses_key]["lines"])
        if job_lines > 0 and response_lines < job_lines:
            print(
                f"response pair pending: {responses_key} has {response_lines}/{job_lines} rows",
                flush=True,
            )
            pairs.append((jobs, responses))
    return pairs


def write_model_map():
    if RUN_PROVIDER_RESPONSES and any(value.startswith("replace-with-") for value in MODEL_MAP.values()):
        raise AssertionError("Fill MODEL_MAP with concrete provider model ids before RUN_PROVIDER_RESPONSES=True.")
    path = ROOT / WORK_DIR / "model_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(MODEL_MAP, indent=2), encoding="utf-8")
    return path


def run_provider_pair(jobs, responses, model_map_path):
    cmd = [
        sys.executable,
        "training/run_curriculum_job_responses.py",
        "--jobs_jsonl",
        str(jobs),
        "--output_jsonl",
        str(responses),
        "--report_json",
        str(responses.with_suffix(".report.json")),
        "--backend",
        PROVIDER_BACKEND,
        "--resume",
        "--fail_fast",
    ]
    if PROVIDER_LIMIT is not None:
        cmd += ["--limit", str(PROVIDER_LIMIT)]
    if PROVIDER_BACKEND == "command":
        cmd += [
            "--command",
            PROVIDER_COMMAND,
            "--model_map_json",
            str(model_map_path),
        ]
    elif PROVIDER_BACKEND == "openai_compatible":
        cmd += [
            "--api_key_env",
            API_KEY_ENV,
            "--base_url",
            BASE_URL,
            "--model_map_json",
            str(model_map_path),
            "--max_tokens",
            str(MAX_TOKENS),
            "--temperature",
            str(TEMPERATURE),
            "--max_retries",
            str(MAX_RETRIES),
            "--retry_sleep_sec",
            str(RETRY_SLEEP_SEC),
            "--retry_backoff",
            str(RETRY_BACKOFF),
        ]
        for key, value in EXTRA_HEADERS.items():
            cmd += ["--extra_header", f"{key}={value}"]
        if EXTRA_BODY_JSON:
            cmd += ["--extra_body_json", json.dumps(EXTRA_BODY_JSON)]
        if JSON_MODE:
            cmd.append("--json_mode")
    run(cmd, cwd=ROOT)


def backup_work_dir():
    if not BACKUP_TO_DRIVE:
        return
    if not Path("/content/drive/MyDrive").exists():
        print("Drive not mounted; skipping backup.", flush=True)
        return
    src = ROOT / WORK_DIR
    dst = Path("/content/drive/MyDrive/recurrent-qwen-svgd/curriculum_runs") / Path(WORK_DIR).name
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"backed up {src} -> {dst}", flush=True)


def run_sft_gate(summary):
    if summary.get("status") != "complete":
        print("SFT gate skipped because curriculum pipeline is not complete.", flush=True)
        return None
    output_json = Path(WORK_DIR) / "curriculum_sft_gate.json"
    output_md = Path(WORK_DIR) / "curriculum_sft_gate.md"
    cmd = [
        sys.executable,
        "training/check_curriculum_sft_gate.py",
        "--summary_json",
        str(Path(WORK_DIR) / "summary.json"),
        "--output_json",
        str(output_json),
        "--output_md",
        str(output_md),
        "--min_positive_rows",
        str(MIN_POSITIVE_ROWS),
        "--fail_on_no_go",
    ]
    if MIN_MODE_ROWS:
        cmd += ["--min_mode_rows", MIN_MODE_ROWS]
    run(cmd, cwd=ROOT)
    gate = json.loads((ROOT / output_json).read_text(encoding="utf-8"))
    print("sft_gate_status:", gate["status"], flush=True)
    print("sft_gate_go:", gate["go"], flush=True)
    return gate


refuse_gpu_runtime_for_cpu_api_work()
sync_repo()
run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
run(["nvidia-smi"], cwd=ROOT, check=False)
if MOUNT_DRIVE and not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive", force_remount=True)

run_pipeline()
summary = summary_payload()
print("status:", summary["status"], flush=True)
print("next_action:", summary["next_action"], flush=True)
print("counts:", summary.get("counts", {}), flush=True)

if RUN_PROVIDER_RESPONSES:
    if PROVIDER_BACKEND == "openai_compatible":
        assert provider_key, f"Missing provider key in Colab secret/env {API_KEY_ENV}."
    model_map_path = write_model_map()
    pairs = response_pairs(summary)
    assert pairs, f"No provider response pairs for status={summary['status']}"
    for jobs, responses in pairs:
        run_provider_pair(jobs, responses, model_map_path)
    run_pipeline()
    summary = summary_payload()
    print("post-provider status:", summary["status"], flush=True)
    print("post-provider next_action:", summary["next_action"], flush=True)
else:
    print("Provider calls skipped. Set RUN_PROVIDER_RESPONSES=True after MODEL_MAP/API key are configured.", flush=True)

run_sft_gate(summary)
backup_work_dir()

if DISCONNECT_RUNTIME_WHEN_DONE:
    print("Disconnecting runtime after CPU curriculum cell.", flush=True)
    runtime.unassign()
